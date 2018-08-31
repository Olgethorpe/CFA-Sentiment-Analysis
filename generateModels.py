from pandas import read_csv
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from numpy import arange, split
from collections import OrderedDict 
from matplotlib import pyplot
from concurrent.futures import ThreadPoolExecutor

def main():
    models = OrderedDict([('Logisitic Regression', LogisticRegression()), ('Random Forest', RandomForestClassifier()), 
                          ('Gaussian Naive Bayes', GaussianNB()), ('Multinomial Naive Bayes', MultinomialNB()),
                          ('Bernoulli Naive Bayes', BernoulliNB()), ('Probability Calibration', CalibratedClassifierCV()),
                          ('Decision Tree', DecisionTreeClassifier()), ('Multi-Layer Perceptron', MLPClassifier())])
    modelAccuracies = OrderedDict()
    modelXvals = None
    dataFile = Path('Data/Tweets.csv')
    data = read_csv(dataFile)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data.text.values).toarray()
    Y = data.airline_sentiment.values
    dataPcts = [(x * 100 * .1) / 100 for x in range(1, 10)]
    with ThreadPoolExecutor() as executor:
        for key, model, count in zip(models.keys(), models.values(), range(len(models))):
            executor.submit(compute, X, Y, model, dataPcts, modelAccuracies, key, count)
    pyplot.figure(len(models) + 1)
    pyplot.ylabel('Accuracy')
    pyplot.xlabel('Train/Test Data Percentage Split')
    pyplot.title('Comparative Accuracy of All Models')
    for modelName, accuracy in modelAccuracies.items():
        pyplot.plot(accuracy.keys(), accuracy.values(), label = modelName)
    pyplot.legend()
    pyplot.savefig('Model Results/Comparative Accuracy.png')

def compute(X, Y, model, dataPcts, modelAccuracies, key, figCount):
    accuracies = OrderedDict()
    print('Started {} analysis'.format(key))
    for trainPct, testPct in zip(dataPcts, dataPcts[::-1]):
        trainVal = int(X.shape[0] * trainPct)
        testVal = int(X.shape[0] * testPct)
        accKey = '{}/{}'.format(int(trainPct * 100), int(testPct * 100))
        Xtrain = X[0:trainVal]
        Ytrain = Y[0:trainVal]
        Xtest = X[trainVal:]
        Ytest = Y[trainVal:]
        accuracies[accKey] = model.fit(Xtrain, Ytrain).score(Xtest, Ytest)
    modelAccuracies[key] = accuracies
    pyplot.figure(figCount)
    pyplot.plot(accuracies.keys(), accuracies.values())
    pyplot.ylabel('Accuracy')
    pyplot.xlabel('Train/Test Data Percentage Split')
    pyplot.title('{} Accuracy'.format(key))
    pyplot.savefig('Model Results/{} Accuracy.png'.format(key))
    figCount += 1
    print('Finished {} analysis'.format(key))

if __name__ == '__main__':
    main()