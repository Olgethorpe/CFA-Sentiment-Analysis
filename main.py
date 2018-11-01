from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import enum
import numpy as np
import pathlib
import pandas as pd
import itertools
import os

class Vectorizers(enum.Enum):
    """Store the vectorizers"""
    V1 = TfidfVectorizer()
    V2 = TfidfVectorizer(ngram_range=(2, 2))
    V3 = TfidfVectorizer(ngram_range=(3, 3))
    V4 = CountVectorizer()
    V5 = CountVectorizer(ngram_range=(2, 2))
    V6 = CountVectorizer(ngram_range=(3, 3))

CV = 5
C = [pow(10, x) for x in range(-4, 4)]
alpha = np.linspace(0, 1, 11)[1:]
kernel = ('linear', 'poly', 'rbf')
gamma = [pow(10, x) for x in range(-3, 3)]
degree = list(range(1, 6))
n_estimators = [x*200 for x in range(1, 11)]
# Stores all the unfitted models
class Models(enum.Enum):
    M1 = GridSearchCV(
        estimator=Pipeline([
            ('sampling', SMOTE()),
            ('classification', LogisticRegression())
        ]), 
        param_grid={
            'classification__C': C
        }, 
        cv=CV, return_train_score=True, verbose=3, n_jobs=-1)
    M2 = GridSearchCV(
        estimator=Pipeline([
            ('sampling', SMOTE()),
            ('classification', MultinomialNB())
        ]), 
        param_grid={
            'classification__alpha': alpha
        }, 
        cv=CV, return_train_score=True, n_jobs=-1, verbose=3)
    M3 = GridSearchCV(
        estimator=Pipeline([
            ('sampling', SMOTE()),
            ('classification', RandomForestClassifier(max_depth=200))
        ]),
        param_grid={
            'classification__n_estimators': n_estimators,
        }, 
        cv=CV, return_train_score=True, verbose=3, n_jobs=2)
    M4 = GridSearchCV(
        estimator=Pipeline([
            ('sampling', SMOTE()),
            ('classification', SVC())
        ]), 
        param_grid={
            'classification__C': C,
            'classification__kernel': kernel,
            'classification__gamma': gamma,
            'classification__degree': degree
        },
        cv=CV, return_train_score=True, n_jobs=-1, verbose=3)
    M5 = GridSearchCV(
        estimator=Pipeline([
            ('sampling', SMOTE()),
            ('classification', Perceptron(max_iter=1000))
        ]),
        param_grid={
            'classification__alpha': alpha
        }, 
        cv=CV, return_train_score=True, n_jobs=-1, verbose=3)
    M6 = GridSearchCV(
        estimator=Pipeline([
            ('sampling', SMOTE()),
            ('classification', KNeighborsClassifier())
        ]), 
        param_grid={
    	    'classification__n_neighbors': [1, 5, 11, 21, 41, 61, 81, 101, 201, 401]
        }, 
        cv=CV, return_train_score=True, n_jobs=2, verbose=3)

class CFAUtils:
    """Utility functions to help with CFA analysis"""
    def powerset(self, iterable):
        """Return the powerset of the vectorizers without the empty set"""
        return list(filter(None, itertools.chain.from_iterable(
            itertools.combinations(iterable, r) for r in range(len(iterable) + 1))))

    def to_list(self, enum_class):
        """Return enumerated values in a slicable iterable"""
        return list(map(lambda x: x.value, enum_class))

class Data:
    """Data object to store and manipulate raw data."""
    def __init__(self, data_file=None):
        if data_file is not None:
            self.raw = data_file

    @property
    def raw(self):
        return self.__raw

    @raw.setter
    def raw(self, data_file):
        data = pd.read_csv(data_file, names=['X', 'Y'], index_col=0, skiprows=1)
        self.__raw = data

    @property
    def transformed(self):
        return self.__transformed

    @transformed.setter
    def transformed(self, data):
        self.__transformed = data

class Text(Data):
    """Text object to manipulate text data"""
    def fit_transform(self, vectorizers, folder):
        """Convert text features to numbers"""
        self.features_folder = folder
        self.vectorizer_names = vectorizers
        if not self.features_file.exists():
            self.vectorizers = vectorizers
            self.transformed = self.vectorizers.fit_transform(self.raw.X)
            np.savez(self.features_file, X = self.transformed, Y = self.raw.Y)
        else:
            self.transformed = np.load(self.features_file)

    @property
    def features_folder(self):
        """Return the location of the data feature combinations"""
        return self._features_folder

    @features_folder.setter
    def features_folder(self, val):
        self._features_folder = pathlib.Path(val).resolve()
        if not self._features_folder.exists():
            os.makedirs(self._features_folder)

    @property
    def features_file(self):
        """Return the location of the data feature file"""
        return self.features_folder.joinpath('{}{}'.format('+'.join(self.vectorizer_names), '.npz'))

    @property
    def vectorizer_names(self):
        """Return the names of the vectorizers used"""
        return self._vectorizer_names

    @vectorizer_names.setter
    def vectorizer_names(self, vectorizers):
        self._vectorizer_names = [vectorizer.name for vectorizer in vectorizers]

    @property
    def vectorizers(self):
        """Return the vectorizers"""
        return self.__vectorizers

    @vectorizers.setter
    def vectorizers(self, vectorizers):
        self.__vectorizers = FeatureUnion([(feature.name, feature.value) for feature in vectorizers])

def main():
    util = CFAUtils()
    dataCombinations = util.powerset(Vectorizers)
    dataCombinations = list(dataCombinations[0:7]) + list(dataCombinations[18:19])
    initFolderStructure()
    createData(dataCombinations)
    trainModels(dataCombinations)

def initFolderStructure():
    """Initialize the folder structure"""
    data_folder = pathlib.Path('Data').resolve()
    raw_data_folder = pathlib.Path('Data/Raw').resolve()
    model_folder = pathlib.Path('Models').resolve()
    smote_folder = pathlib.Path('Data/SMOTE_Data').resolve()
    test_folder = pathlib.Path('Data/Test_Data').resolve()
    if not data_folder.exists():
        os.makedirs(data_folder)
    if not raw_data_folder.exists():
        os.makedirs(raw_data_folder)
    if not model_folder.exists():
        os.makedirs(model_folder)
    if not smote_folder.exists():
        os.makedirs(smote_folder)
    if not test_folder.exists():
        os.makedirs(test_folder)
        os.makedirs(test_folder.joinpath('Test1'))
        os.makedirs(test_folder.joinpath('Test2'))

def createData(dataCombinations):
    """Create the data combinations"""
    data_file = pathlib.Path('Data/Raw/Tweets.csv').resolve()
    data = Text(data_file)
    for comb in dataCombinations:
        data.fit_transform(comb, 'Data/Data_Feature_Combinations')

def trainModels(dataCombinations, resultFolder=None):
    """Train all models for each combination"""
    modelFolder = pathlib.Path('Models').resolve()
    if resultFolder is not None:
        modelFolder.joinpath(resultFolder)
    for comb in dataCombinations[0:6]:
        names = '+'.join([v.name for v in comb])
        folder = modelFolder.joinpath(names)
        if not folder.exists():
            os.makedirs(folder)
        for model in Models:
            modelFile = folder.joinpath('{}.csv'.format(model.name))
            if not modelFile.exists():
                data = pathlib.Path('Data/SMOTE_Data/{}.npz'.format(names)).resolve()
                data = np.load(data)
                model.value.fit(data['X'].ravel()[0], data['Y'])
                data = pd.DataFrame.from_dict(model.value.cv_results_)
                if model == Models.M1:
                    data = data[['param_classification__C', 'mean_train_score', 'std_train_score', 'mean_test_score', 'std_test_score']]
                elif model == Models.M2 or model == Models.M5:
                    data = data[['param_classification__alpha', 'mean_train_score', 'std_train_score', 'mean_test_score', 'std_test_score']]
                elif model == Models.M3:
                    data = data[['param_classification__n_estimators', 'mean_train_score', 'std_train_score', 'mean_test_score', 'std_test_score']]
                elif model == model.M4:
                    data = data[['param_classification__C', 'param_classification__kernel', 'param_classification__gamma', 'param_classification__degree', 
                                 'mean_train_score', 'std_train_score', 'mean_test_score', 'std_test_score']]
                elif model == model.M6:
                    data = data[['param_classification__n_neighbors', 'mean_train_score', 'std_train_score', 'mean_test_score', 'std_test_score']]
                data.to_csv(modelFile)

if __name__ == '__main__':
    main()
