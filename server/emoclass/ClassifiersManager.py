import numpy as np
import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from joblib import dump, load
from numpy import mean, isnan
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate


classifiers = {
    'lda': LinearDiscriminantAnalysis(),
    'qda': QuadraticDiscriminantAnalysis(),
    'rf': RandomForestClassifier(n_estimators=100, max_depth=5, oob_score=True, min_samples_split=20),
    'ab': AdaBoostClassifier(n_estimators=100),
    'knn': KNeighborsClassifier(n_neighbors=9),
    'svm': svm.SVC()
}


def evaluator(classifier, ypred, ytest):
    errors = abs(ypred - ytest)
    val = np.count_nonzero(errors)
    print('=== Classifier: ' + classifier + ' ===')
    #print("M A E: ", np.mean(errors))
    print(np.count_nonzero(errors), len(ytest))
    print('Accuracy: ', accuracy_score(ytest, ypred))


def evaluate_model(X, y, idClf):
    model = classifiers[idClf]
    #cv = StratifiedKFold(n_splits=10)
    cv = KFold(n_splits=10, shuffle=True, random_state=1)
    scores = cross_validate(model, X, y, scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'], cv=cv, n_jobs=-1)
    return {
        'acc_mean': mean(scores['test_accuracy']),
        'acc_min': scores['test_accuracy'].min(),
        'acc_max': scores['test_accuracy'].max(),
        #'prec_mean': mean(scores['test_precision_macro']),
        #'prec_min': scores['test_precision_macro'].min(),
        #'prec_max': scores['test_precision_macro'].max(),
        #'rec_mean': mean(scores['test_recall_macro']),
        #'rec_min': scores['test_recall_macro'].min(),
        #'rec_max': scores['test_recall_macro'].max(),
        #'f1_mean': mean(scores['test_f1_macro']),
        #'f1_min': scores['test_f1_macro'].min(),
        #'f1_max': scores['test_f1_macro'].max()
        'prec_mean': mean(scores['test_precision_weighted']),
        'prec_min': scores['test_precision_weighted'].min(),
        'prec_max': scores['test_precision_weighted'].max(),
        'rec_mean': mean(scores['test_recall_weighted']),
        'rec_min': scores['test_recall_weighted'].min(),
        'rec_max': scores['test_recall_weighted'].max(),
        'f1_mean': mean(scores['test_f1_weighted']),
        'f1_min': scores['test_f1_weighted'].min(),
        'f1_max': scores['test_f1_weighted'].max(),
    }
    

def train_classifier(idClassifier,  Xtrain, ytrain, saveClf=True, nameClf='', folder=''):
    clf = classifiers[idClassifier]
    clf.fit(Xtrain, ytrain)
    if saveClf:
        if nameClf == '':
            nameClf = idClassifier
        dump(clf, folder + nameClf + '.joblib')
    return clf


def test_classifier(clf, Xtest, ytest):
    ypred = clf.predict(Xtest)
    errors = abs(ypred - ytest)
    print('Errors: ', str(np.count_nonzero(errors)) + ' / ' + str(len(ytest)))
    acc = accuracy_score(ytest, ypred)
    print('Accuracy: ', acc)
    print('ConfusionMatrix')
    print(confusion_matrix(ytest, ypred))
    return acc


def load_classifier(folder, nameClf):
    return load(folder + nameClf + '.joblib')


if __name__ == "__main__":
    folder = '../models/'
    data_folder = '../../datasets/data_files/'
    y = pickle.load(open('../../datasets/data_files/all_df_y', 'rb'))
    X = pickle.load(open('../../datasets/data_files/all_features_x', 'rb'))
    print(X.shape, y.shape)

    idClf = 'knn'
    train_classifier(idClf, X.values.tolist(), y['valence'].values.tolist(), folder=folder, nameClf=idClf + '_valence')
    train_classifier(idClf, X.values.tolist(), y['arousal'].values.tolist(), folder=folder, nameClf=idClf + '_arousal')
    clf = load_classifier(folder, idClf + '_valence')
    test_classifier(clf, X, y['valence'])
    print(clf.predict(X[0:3]))
    clf = load_classifier(folder, idClf + '_arousal')
    test_classifier(clf, X, y['arousal'])
    print(clf.predict(X[0:3]))
