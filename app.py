import json
from configparser import ConfigParser
import web

from server.loaders import DataLoader
from server.emoclass import EmoClassification as ec
from server.emovis import EmoDiscretization as emodis
from server.emovis import FeatureContribution as fcontrib

render = web.template.render('templates/')

urls = (
    '/', 'Index',
    '/load_channels', 'LoadChannels',
    '/process_dataset', 'ProcessDataset',
    '/retrain_classifier', 'Retraining'
)

app = web.application(urls, globals())

# Reading file of server connection
conf = ConfigParser()
conf.read('server/server_config.cfg')
data_folder = conf.get('general', 'data_folder')
models_folder = conf.get('general', 'models_folder')

# variables
devMode = True


class Index(object):
    def GET(self):
        return render.index()


class LoadChannels(object):
    def POST(self):
        data = json.loads(web.data())
        dataset_folder = conf.get('dataset', data["dataset"] + '_folder')
        web.header('Content-Type', 'application/json')
        return json.dumps(DataLoader.load_channels(data["dataset"], dataset_folder))


class ProcessDataset(object):
    """
    data = {
        'dataset': idDataset,
        'fselector': id fSelector,
        'classifier': idClassifier,
        'winSize': float,
        'winIni': int,
        'sampleSize': int,
        'channels': Array,
        'testSize': int
    };
    """
    def POST(self):
        data = json.loads(web.data())
        dataset_folder = conf.get('dataset', data["dataset"] + '_folder')
        if not devMode:
            DataLoader.convert_dataset(data["dataset"], dataset_folder, data_folder)
        predicted_vals, ground_vals, original_features, emotion_names, metrics = ec.train_and_evaluate_classifier(data, data_folder)
        #predicted_vals, emotion_names = emodis.discretize_by_quadrant(predicted_vals, data["nClasses"])
        #featuresContrib, emo_names, feat_names = fcontrib.compute_contribution(original_features, predicted_vals, emotion_names)
        featuresContrib, emo_names, feat_names = fcontrib.compute_contribution(original_features, ground_vals,
                                                                               emotion_names)
        res = {
            'class': ground_vals, # predicted_vals,
            'class_gt': ground_vals,
            'features': {'fcs': featuresContrib, 'emo_names': emo_names, 'feat_names': feat_names},
            'emo_names': emo_names,
            'metrics': metrics
        }
        web.header('Content-Type', 'application/json')
        return json.dumps(res)


class Retraining(object):
    """
    data = {
        'dataset': idDataset,
        'classifier': idClassifier,
        'winSize': float,
        'winIni': int,
        'sampleSize': int,
        'features': Array
    };
    """

    def POST(self):
        data = json.loads(web.data())
        predicted_vals, ground_vals, original_features, emotion_names, metrics = ec.train_and_evaluate_classifier(data, data_folder)
        #featuresContrib, emo_names, feat_names = fcontrib.compute_contribution(original_features, ground_vals,
        #                                                                       emotion_names)
        res = {
            'class': ground_vals,  # predicted_vals,
            'class_gt': ground_vals,
            #'features': {'fcs': featuresContrib, 'emo_names': emo_names, 'feat_names': feat_names},
            #'emo_names': emo_names,
            'metrics': metrics
        }
        web.header('Content-Type', 'application/json')
        return json.dumps(res)


if __name__ == "__main__":
    app.run()
