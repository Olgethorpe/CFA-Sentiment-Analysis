from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import mutual_info_classif, chi2, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import *
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from scipy.stats import pearsonr
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

# Parameters to pass to models
CV = 5
C = [pow(10, x) for x in range(-4, 4)]
alpha = np.linspace(0, 1, 11)[1:]
kernel = ('linear', 'poly', 'rbf')
gamma = [pow(10, x) for x in range(-3, 3)]
nn_alpha = [pow(10, x) for x in range(-4, 1)]
degree = list(range(1, 6))
n_estimators = [x*200 for x in range(1, 11)]
nn_estimators = [(neuron,) * layer for layer in [x*100 for x in range(1, 11)] for neuron in [10, 50, 100, 150]]
scorer = {
    'F1': make_scorer(f1_score, average='macro'),
    'Accuracy': make_scorer(accuracy_score),
    'Precision': make_scorer(precision_score, average='macro'),
    'Recall': make_scorer(recall_score, average='macro'),
    'AUC': make_scorer(roc_auc_score, average='macro')
}

class Models(enum.Enum):
    """'Stores the unfitted models'"""
    M1 = GridSearchCV(
        estimator=Pipeline([
            ('sampling', SMOTE()),
            ('classification', LogisticRegression(multi_class='ovr', solver='liblinear', random_state=42))
        ]),
        param_grid={
            'classification__C': C
        },
        cv=CV, return_train_score=True, verbose=10, n_jobs=-1, scoring=scorer, refit=False)
    M2 = GridSearchCV(
        estimator=Pipeline([
            ('sampling', SMOTE()),
            ('classification', MultinomialNB())
        ]),
        param_grid={
            'classification__alpha': alpha
        },
        cv=CV, return_train_score=True, n_jobs=-1, verbose=10, scoring=scorer, refit=False)
    M3 = GridSearchCV(
        estimator=Pipeline([
            ('sampling', SMOTE()),
            ('classification', RandomForestClassifier(max_depth=200, random_state=42))
        ]),
        param_grid={
            'classification__n_estimators': n_estimators,
        },
        cv=CV, return_train_score=True, verbose=10, n_jobs=-1, scoring=scorer, refit=False)
    M4 = GridSearchCV(
        estimator=Pipeline([
            ('sampling', SMOTE()),
            ('classification', SVC(random_state=42))
        ]),
        param_grid={
            'classification__C': C,
            'classification__kernel': kernel,
            'classification__gamma': gamma,
            'classification__degree': degree
        },
        cv=CV, return_train_score=True, n_jobs=-1, verbose=10, scoring=scorer, refit=False)
    M5 = GridSearchCV(
        estimator=Pipeline([
            ('sampling', SMOTE()),
            ('classification', Perceptron(max_iter=1000))
        ]),
        param_grid={
            'classification__alpha': alpha
        },
        cv=CV, return_train_score=True, n_jobs=-1, verbose=10, scoring=scorer, refit=False)
    M6 = GridSearchCV(
        estimator=Pipeline([
            ('sampling', SMOTE()),
            ('classification', MLPClassifier())
        ]),
        param_grid={
            'classification__alpha': nn_alpha,
            'classification__learning_rate_init': nn_alpha,
            'classification__hidden_layer_sizes': nn_estimators
        },
        cv=CV, return_train_score=True, n_jobs=-1, verbose=10, scoring=scorer, refit=False)

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
    data_combs = list(data_combs[0:7]) + list(data_combs[18:19]) # Remove for all combinations
    benchmark_folder = pathlib.Path('data/benchmark_data').resolve()
    features_selected_folder = pathlib.Path('data/feature_selected_data').resolve()
    os.makedirs(benchmark_folder, exist_ok=True)
    os.makedirs(features_selected_folder, exist_ok=True)
    raw_data_file = pathlib.Path('data/raw/tweets.csv').resolve()
    raw_data = pd.read_csv(raw_data_file, index_col=0, names=['X', 'Y'], skiprows=1)
    X = raw_data['X']
    Y = raw_data['Y']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=42)
    for comb in data_combs:
        folder_name = ''
        for vectorizer in comb:
            folder_name += vectorizer.name
            if len(comb) > 1 and vectorizer != comb[-1]:
                folder_name += '+'
        union = FeatureUnion([(v.name, v.value) for v in comb])
        X_fitted_data = union.fit_transform(X_train)
        #print(X_fitted_data.shape)
        gen_model_data_folder = benchmark_folder.joinpath(folder_name)
        selected_features_model_data_folder = features_selected_folder.joinpath(folder_name)
        os.makedirs(gen_model_data_folder, exist_ok=True)
        os.makedirs(selected_features_model_data_folder, exist_ok=True)
        for m in Models:
            model_file = '{}.csv'.format(m.name)
            gen_model_file = gen_model_data_folder.joinpath(model_file)
            selected_features_model_file = selected_features_model_data_folder.joinpath(model_file)
            if not gen_model_file.exists():
                m.value.fit(X_fitted_data, Y_train)
                results = pd.DataFrame.from_dict(m.value.cv_results_)
                results.to_csv(gen_model_file)
            """
            Discuss about feature selection
            if not selected_features_model_file.exists():
                corrs = []
                chis = []
                infos = []
                for i in range(0, X_train.shape[0]):
                    x_corr_data = pd.DataFrame(X_fitted_data.getcol(i).todense())
                    corr = pearsonr(x_corr_data[0], Y_train)
                    chi = chi2(x_corr_data[0].values.reshape(-1, 1), Y_train.values.reshape(-1, 1))
                    corrs.append(corr)
                    chis.append((chi[0][0], chi[1][0]))
                corrs = pd.DataFrame(corrs, columns=['Pearson Coef', 'Pearson P-Val'])
                corrs['Pearson Coef'] = corrs['Pearson Coef'].apply(abs)
                chis = pd.DataFrame(chis, columns=['Chi^2', 'Chi P-Val'])
                all_vals = pd.concat([corrs, chis], axis=1)
                #all_vals.to_csv('test.csv')
                print(all_vals)
            """

if __name__ == '__main__':
    main()
