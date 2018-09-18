from enum import Enum
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from concurrent.futures import ProcessPoolExecutor

# Stores the different models
class Models(Enum):
    A = LogisticRegression()
    B = MultinomialNB()
    C = RandomForestClassifier()
    D = LinearSVC()
    E = Perceptron()

    # Returns all models in a list
    @classmethod
    def list(self):
        return list(map(lambda x: x.value, self))

	# Fits all models to some given data
    @classmethod
    def fit(self, Xtrain, Ytrain):
        with ProcessPoolExecutor() as executor:
            for x in self:
                executor.submit(x.value.fit(Xtrain, Ytrain))
