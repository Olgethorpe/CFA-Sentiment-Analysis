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
import joblib

class Vectorizers(enum.Enum):
    """Store the vectorizers"""
    V1 = TfidfVectorizer()
    V2 = TfidfVectorizer(ngram_range=(2, 2))
    V3 = TfidfVectorizer(ngram_range=(3, 3))
    V4 = CountVectorizer()
    V5 = CountVectorizer(ngram_range=(2, 2))
    V6 = CountVectorizer(ngram_range=(3, 3))

# Parameters to pass to models
CV = 5
C = [pow(10, x) for x in range(-4, 4)]
alpha = np.linspace(0, 1, 11)[1:]
kernel = ('linear', 'poly', 'rbf')
gamma = [pow(10, x) for x in range(-3, 3)]
degree = list(range(1, 6))
n_estimators = [x*200 for x in range(1, 11)]

class Models(enum.Enum):
    """Stores the unfitted models"""
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
        self.train_features_folder = folder
        self.test_features_folder = folder
        self.vectorizer_names = vectorizers
        if not self.train_features_file.exists():
            self.vectorizers = vectorizers
            self.transformed = self.vectorizers.fit_transform(self.raw.X)
            x_train, x_test, y_train, y_test = train_test_split(self.transformed, self.raw.Y, test_size=.5, random_state=1)
            np.savez(self.train_features_file, X=x_train, Y=y_train)
            if not self.test_features_file.exists():
                np.savez(self.test_features_file, X=x_test, Y=y_test)

    @property
    def train_features_folder(self):
        """Return the location of the training data feature combinations"""
        return self._features_folder

    @train_features_folder.setter
    def train_features_folder(self, val):
        self._features_folder = pathlib.Path(val).resolve()
        if not self._features_folder.exists():
            os.makedirs(self._features_folder)

    @property
    def test_features_folder(self):
        """Return the location of the testing data feature combinations"""
        return self._test_folder

    @test_features_folder.setter
    def test_features_folder(self, val):
        self._test_folder = self._features_folder.parents[0].joinpath('test')
        if not self._test_folder.exists():
            os.makedirs(self._test_folder)

    @property
    def train_features_file(self):
        """Return the location of the data feature file"""
        return self.train_features_folder.joinpath('{}{}'.format(self.vectorizer_names, '.npz'))

    @property
    def test_features_file(self):
        """Return the location of the data feature file"""
        return self.test_features_folder.joinpath('{}{}'.format(self.vectorizer_names, '.npz'))

    @property
    def vectorizer_names(self):
        """Return the names of the vectorizers used"""
        return self._vectorizer_names

    @vectorizer_names.setter
    def vectorizer_names(self, vectorizers):
        self._vectorizer_names = '+'.join([vectorizer.name for vectorizer in vectorizers])

    @property
    def vectorizers(self):
        """Return the vectorizers"""
        return self._vectorizers

    @vectorizers.setter
    def vectorizers(self, vectorizers):
        self._vectorizers = FeatureUnion([(feature.name, feature.value) for feature in vectorizers])

class CFA:
    """Combinatorial Fusion Analysis object"""
    def __init__(self, **kwargs):
        """Initialize the CFA object"""

def main():
    """Main definition of the program"""
    util = CFAUtils()
    data_combs = util.powerset(Vectorizers)
    data_combs = list(data_combs[0:7]) + list(data_combs[18:19])
    create_data(data_combs)
    #trainModels(data_combs) # Only for use when tuning
    train_tuned_models()

def create_data(data_combs):
    """Create the data combinations"""
    data_file = pathlib.Path('data/raw/tweets.csv').resolve()
    data = Text(data_file)
    for comb in data_combs:
        data.fit_transform(comb, 'data/train')

# TODO: fix when we have a ton of models
def train_tuned_models():
    """Train the models using the selected tuning parameters"""
    tuned_folder = pathlib.Path('models').resolve().joinpath('tuned')
    tuning_params = pd.read_csv(pathlib.Path('models').resolve().joinpath('tuning_params.csv'), index_col=0)
    if not tuned_folder.exists():
        os.makedirs(tuned_folder)
    tuning_params.apply(train, args=(tuned_folder,), axis=1)

def train(x, tuned_folder):
    """Train a model with some tuning parameters"""
    vectorizer_folder = tuned_folder.joinpath(x.name)
    data_file = np.load(pathlib.Path('data').resolve().joinpath('train', '{}.npz'.format(x.name)))
    if not vectorizer_folder.exists():
        os.makedirs(vectorizer_folder)
    for col_val, col in zip(x, x.index):
        if col == 'M1':
            m = LogisticRegression(C=col_val, solver='liblinear', multi_class='auto', verbose=3, n_jobs=-1, random_state=1)
        elif col == 'M2':
            m = MultinomialNB(alpha=col_val)
        elif col == 'M3':
            m = RandomForestClassifier(n_estimators=col_val, verbose=3, n_jobs=-1)
        elif col == 'M4':
            params = col_val.split(',')
            m = SVC(C=float(params[0]), kernel=params[1], gamma=float(params[2]), degree=float(params[3]), verbose=3)
        elif col == 'M5':
            m = Perceptron(alpha=col_val, max_iter=1000, verbose=3, n_jobs=-1)
        model_file = vectorizer_folder.joinpath('{}.joblib'.format(col))
        if not model_file.exists():
            m.fit(data_file['X'].ravel()[0], data_file['Y'])
            joblib.dump(m, model_file)

# TODO: fix when revisiting project to follow new folder structure
def trainModels(data_combs, resultFolder=None):
    """Train all models for each combination"""
    modelFolder = pathlib.Path('models').resolve()
    if resultFolder is not None:
        modelFolder.joinpath(resultFolder)
    for comb in data_combs[0:6]:
        names = '+'.join([v.name for v in comb])
        folder = modelFolder.joinpath(names)
        if not folder.exists():
            os.makedirs(folder)
        for model in Models:
            modelFile = folder.joinpath('{}.csv'.format(model.name))
            if not modelFile.exists():
                data = pathlib.Path('data/train/{}.npz'.format(names)).resolve()
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
