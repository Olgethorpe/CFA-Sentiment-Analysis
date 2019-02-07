from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
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

class Parameters(enum.Enum):
    """Parameters to pass to models"""
    CV = 5
    C1 = [pow(10, x) for x in range(-4, 4)]
    C2 = [pow(10, x) for x in range(-2, 1)]
    Alpha = np.linspace(0, 1, 11)[1:]
    Kernel = ('linear', 'poly', 'rbf')
    N_estimators = [x*200 for x in range(1, 11)]
    Degree = [1,2,4]
    Gamma = [pow(10, x) for x in range(-3, 0)]
    Scorer = {
        'F1': make_scorer(f1_score, average='macro'),
        'Accuracy': make_scorer(accuracy_score),
        'Precision': make_scorer(precision_score, average='macro'),
        'Recall': make_scorer(recall_score, average='macro'),
    }

class CVModels(enum.Enum):
    """'Stores the unfitted models'"""
    M1 = GridSearchCV(
        estimator=Pipeline([
            ('sampling', SMOTE()),
            ('classification', LogisticRegression(
                multi_class='ovr',
                solver='liblinear',
                random_state=42)
            )
        ]),
        param_grid={
            'classification__C': Parameters.C1
        },
        cv=Parameters.CV,
        return_train_score=True,
        verbose=10,
        n_jobs=-1,
        scoring=Parameters.Scorer,
        refit=False
    )
    M2 = GridSearchCV(
        estimator=Pipeline([
            ('sampling', SMOTE()),
            ('classification', MultinomialNB())
        ]),
        param_grid={
            'classification__alpha': Parameters.Alpha
        },
        cv=Parameters.CV,
        return_train_score=True,
        n_jobs=-1,
        verbose=10,
        scoring=Parameters.Scorer,
        refit=False
    )
    M3 = GridSearchCV(
        estimator=Pipeline([
            ('sampling', SMOTE()),
            ('classification', RandomForestClassifier(
                max_depth=200,
                random_state=42,
                n_jobs=7)
            )
        ]),
        param_grid={
            'classification__n_estimators': Parameters.N_estimators,
        },
        cv=Parameters.CV,
        return_train_score=True,
        n_jobs=-1,
        verbose=10,
        scoring=Parameters.Scorer,
        refit=False
    )
    M4 = GridSearchCV(
        estimator=Pipeline([
            ('sampling', SMOTE()),
            ('classification', SVC(random_state=42))
        ]),
        param_grid={
            'classification__C': Parameters.C2,
            'classification__kernel': Parameters.Kernel,
            'classification__gamma': Parameters.Gamma,
            'classification__degree': Parameters.Degree
        },
        cv=Parameters.CV,
        return_train_score=True,
        n_jobs=-1,
        verbose=10,
        scoring=Parameters.Scorer,
        refit=False
    )
    M5 = GridSearchCV(
        estimator=Pipeline([
            ('sampling', SMOTE()),
            ('classification', Perceptron(max_iter=1000, tol=1e-3))
        ]),
        param_grid={
            'classification__alpha': Parameters.Alpha
        },
        cv=Parameters.CV,
        return_train_score=True,
        n_jobs=-1,
        verbose=10,
        scoring=Parameters.Scorer,
        refit=False
    )

class CFAUtils:
    """Utility functions to help with CFA analysis"""
    def powerset(self, iterable):
        """Return the powerset of the vectorizers without the empty set"""
        return list(filter(None, itertools.chain.from_iterable(
            itertools.combinations(iterable, r) for r in range(len(iterable) + 1))))

    def to_list(self, enum_class):
        """Return enumerated values in a slicable iterable"""
        return list(map(lambda x: x.value, enum_class))

def main():
    """Main definition of the program"""
    util = CFAUtils()
    data_combs = util.powerset(Vectorizers)
    benchmark_folder = pathlib.Path('data/benchmark_data').resolve()
    os.makedirs(benchmark_folder, exist_ok=True)
    raw_data_file = pathlib.Path('data/raw/tweets.csv').resolve()
    raw_data = pd.read_csv(
        raw_data_file,
        index_col=0,
        names=['X', 'Y'],
        skiprows=1
    )
    X = raw_data['X']
    Y = raw_data['Y']
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=.2,
        random_state=42
    )
    for comb in data_combs:
        folder_name = ''
        for vectorizer in comb:
            folder_name += vectorizer.name
            if len(comb) > 1 and vectorizer != comb[-1]:
                folder_name += '+'
        union = FeatureUnion([(v.name, v.value) for v in comb])
        X_fitted_data = union.fit_transform(X_train)
        gen_model_data_folder = benchmark_folder.joinpath(folder_name)
        os.makedirs(gen_model_data_folder, exist_ok=True)
        for CV_m in CVModels:
            model_file = '{}.csv'.format(CV_m.name)
            gen_model_file = gen_model_data_folder.joinpath(model_file)
            if not gen_model_file.exists():
                CV_m.value.fit(X_fitted_data, Y_train)
                results = pd.DataFrame.from_dict(CV_m.value.cv_results_)
                results.to_csv(gen_model_file)

if __name__ == '__main__':
    main()
