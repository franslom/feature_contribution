import pickle
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import KFold
import server.emovis.EmoDiscretization as EDisc
import server.emoclass.ClassifiersManager as ClfMan
import server.emoclass.FeatureManager as FMan

nClasses = None


def classify_2dim(data, clf_valence, clf_arousal):
    valence = clf_valence.predict(data)
    arousal = clf_arousal.predict(data)
    return [{'valence': valence[i], 'arousal': arousal[i]} for i in range(len(valence))]


def classify_1dim(data, clf):
    pred_vals = clf.predict(data)
    return [EDisc.get_centroid_emotion(pred_vals[i], nClasses) for i in range(len(pred_vals))]


def train_and_test_by_scale(features, all_df_y, id_classifier, test_size):
    n_folds = round(1 / (test_size / 100.0))
    kf = KFold(n_splits=n_folds)
    max_acc_aro = 0
    max_acc_val = 0
    best_clf_val = None
    best_clf_aro = None
    for train_index, test_index in kf.split(features):
        X_train, X_test = features[train_index], features[test_index]
        y_train_v, y_test_v = all_df_y['valence'][train_index], all_df_y['valence'][test_index]
        y_train_a, y_test_a = all_df_y['arousal'][train_index], all_df_y['arousal'][test_index]
        clf_val = ClfMan.train_classifier(id_classifier, X_train, y_train_v, saveClf=False)
        clf_aro = ClfMan.train_classifier(id_classifier, X_train, y_train_a, saveClf=False)
        acc_val = ClfMan.test_classifier(clf_val, X_test, y_test_v)
        acc_aro = ClfMan.test_classifier(clf_val, X_test, y_test_a)
        if acc_val > max_acc_val:
            max_acc_val = acc_val
            best_clf_val = clf_val
        if acc_aro > max_acc_aro:
            max_acc_aro = acc_aro
            best_clf_aro = clf_aro

    predicted_vals = classify_2dim(features, best_clf_val, best_clf_aro)
    return predicted_vals, best_clf_val, best_clf_aro


def train_and_test_by_level(features, all_df_y, id_classifier, test_size):
    n_folds = round(1 / (test_size / 100.0))
    kf = KFold(n_splits=n_folds)
    max_acc_aro = 0
    max_acc_val = 0
    best_clf_val = None
    best_clf_aro = None
    tmp_labs, _ = EDisc.discretize_by_level(all_df_y.to_dict('records'), nClasses)
    all_df_y = pd.DataFrame.from_records(tmp_labs)

    for train_index, test_index in kf.split(features):
        X_train, X_test = features[train_index], features[test_index]
        y_train_v, y_test_v = all_df_y['val_lvl'][train_index], all_df_y['val_lvl'][test_index]
        y_train_a, y_test_a = all_df_y['aro_lvl'][train_index], all_df_y['aro_lvl'][test_index]
        """
        print(np.count_nonzero(np.isnan(X_train)), np.count_nonzero(np.isnan(X_test)))
        print(np.count_nonzero(np.isnan(y_train_v)), np.count_nonzero(np.isnan(y_test_v)))
        print(np.count_nonzero(np.isnan(y_train_a)), np.count_nonzero(np.isnan(y_test_a)))
        """
        clf_val = ClfMan.train_classifier(id_classifier, X_train, y_train_v, saveClf=False)
        clf_aro = ClfMan.train_classifier(id_classifier, X_train, y_train_a, saveClf=False)
        acc_val = ClfMan.test_classifier(clf_val, X_test, y_test_v)
        acc_aro = ClfMan.test_classifier(clf_val, X_test, y_test_a)
        if acc_val > max_acc_val:
            max_acc_val = acc_val
            best_clf_val = clf_val
        if acc_aro > max_acc_aro:
            max_acc_aro = acc_aro
            best_clf_aro = clf_aro

    # predict values for all
    valence = best_clf_val.predict(features)
    arousal = best_clf_aro.predict(features)
    """
    print("testing arousal with all")
    ClfMan.test_classifier(best_clf_aro, features, all_df_y['aro_lvl'])
    ClfMan.test_classifier(best_clf_val, features, all_df_y['val_lvl'])
    """
    predicted_vals = [{'valence': EDisc.get_centroid_level(valence[i], nClasses), 'arousal': EDisc.get_centroid_level(arousal[i], nClasses)} for i in range(len(valence))]
    return predicted_vals, best_clf_val, best_clf_aro


def train_and_test_by_emotion(features, all_df_y, id_classifier, test_size):
    n_folds = round(1 / (test_size / 100.0))
    kf = KFold(n_splits=n_folds)
    max_acc = 0
    best_clf = None
    for train_index, test_index in kf.split(features):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = all_df_y['emotion'][train_index], all_df_y['emotion'][test_index]
        clf_val = ClfMan.train_classifier(id_classifier, X_train, y_train, saveClf=False)
        acc_val = ClfMan.test_classifier(clf_val, X_test, y_test)
        if acc_val > max_acc:
            max_acc = acc_val
            best_clf = clf_val
    predicted_vals = classify_1dim(features, best_clf)
    return predicted_vals, best_clf, None


modeClassification = {
    'AVs': train_and_test_by_scale,
    'AVl': train_and_test_by_level,
    'AV': train_and_test_by_level,
    'emo': train_and_test_by_emotion
}


# Loading selected features
def get_dataset(x_filename, y_filename, channels=None, features=None):
    feature_names = ['Unnamed: 0']
    if features is None:
        for channel in channels:
            tmp = channel["channel"].split("_")
            id_signal, id_ch = tmp[0], tmp[1]
            if len(channel["features"]) == 0:
                feature_names += FMan.get_feature_list(id_signal, id_ch)
            else:
                feature_names += [id_signal + "_" + id_ch + "_" + fname for fname in channel["features"]]
    else:
        feature_names += features

    print(feature_names)

    X = pd.read_csv(x_filename, index_col=0, usecols=feature_names)
    X = FMan.preprocess_features(X)
    y = pd.read_csv(y_filename, index_col=0)
    print("X = ", X.shape, "y = ", y.shape)
    return X, y


def train_and_evaluate_classifier(data_in, data_folder):
    global nClasses
    nClasses = data_in["nClasses"]
    id_clf = data_in["classifier"]
    if "features" in data_in.keys():
        tmp_features = {}
        for id_feature in data_in["features"]:
            vals = id_feature.split("_")
            id_ch = vals[0] + "_" + vals[1]
            if id_ch not in tmp_features.keys():
                tmp_features[id_ch] = []
            tmp_features[id_ch].append("_".join(vals[2:]))
        channels = [{"channel": ch, "features": tmp_features[ch]} for ch in tmp_features]
    else:
        channels = [{"channel": ch, "features": []} for ch in data_in["channels"]]

    print("channels:", channels)
    start_time = time.time()

    data_in["channels"] = channels
    x_filename = FMan.extract_features(data_in, data_folder)
    y_filename = data_folder + 'all_df_y'
    features_or, all_df_y = get_dataset(x_filename, y_filename, channels)
    # lo que devuelve la aplicacion del fselector son los componentes principales, no podemos especificar si son
    # features especificos de la lista inicial
    features = FMan.select_features(data_in["fselector"], features_or)
    print('features for model: ', features.shape)

    new_labels, _ = EDisc.discretize_by_level(all_df_y.to_dict('records'), nClasses)
    all_df_y = pd.DataFrame.from_records(new_labels)
    new_labels, quad_names = EDisc.discretize_by_quadrant(all_df_y.to_dict('records'), nClasses)
    all_df_y = pd.DataFrame.from_records(new_labels)

    # Evaluation of the model
    res_quad = ClfMan.evaluate_model(features, all_df_y["emotion"], id_clf)
    res_aro = ClfMan.evaluate_model(features, all_df_y["aro_lvl"], id_clf)
    res_val = ClfMan.evaluate_model(features, all_df_y["val_lvl"], id_clf)

    duration = (time.time() - start_time)

    return None, all_df_y.to_dict('records'), features_or, quad_names, {"res_quad": res_quad, "res_aro": res_aro, "res_val": res_val, "duration": duration}


"""
def start_classification(data_in, data_folder):
    global nClasses
    nClasses = data_in["nClasses"]
    channels = [{"channel": ch, "features": []} for ch in data_in["channels"]]
    data_in["channels"] = channels

    x_filename = FMan.extract_features(data_in, data_folder)
    y_filename = data_folder + 'all_df_y'

    features_or, all_df_y = get_dataset(x_filename, y_filename, channels)
    features = FMan.select_features(data_in["fselector"], features_or)
    print('fselector', features.shape)

    labels_gt, quad_names, metrics = discretize_and_evaluate(features, all_df_y, nClasses, data_in["classifier"])
    return None, labels_gt, features_or, quad_names, metrics


def retrain_classifier(data_in, data_folder):
    global nClasses
    nClasses = data_in["nClasses"]

    x_filename = data_folder + "all_features_x"
    y_filename = data_folder + 'all_df_y'
    listFeatures = data_in["features"]

    features, all_df_y = get_dataset(x_filename, y_filename, features=listFeatures)
    print('features', features.shape)

    labels_gt, quad_names, metrics = discretize_and_evaluate(features, all_df_y, nClasses, data_in["classifier"])
    return None, labels_gt, features, quad_names, metrics
"""


if __name__ == "__main__":
    data = {'channels': ['GSR_GSR'], 'winSize': 63, 'winIni': 0, 'sampleSize': 128, 'fselector': "",
            'classifier': "svm", 'nClasses': 9, 'testSize': 20, 'mode': "AVl"}
    out_folder = '../../datasets/data_files/'
    pred, ground, features = train_and_evaluate_classifier(data, out_folder)
    print(features.shape)
    print(len(pred), len(ground))
    print(pred)
    print(ground)