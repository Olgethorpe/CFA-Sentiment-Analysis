from pandas import read_csv
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from numpy import arange, split
from collections import OrderedDict
from matplotlib import pyplot

def main():
    models = {'Logisitic Regression': LogisticRegression(), 'Random Forest': RandomForestClassifier()}
    dataFile = Path('Data/Tweets.csv')
    data = read_csv(dataFile)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data.text.values).toarray()
    Y = data.airline_sentiment.values
    dataPcts = [(x * 100 * .1) / 100 for x in range(1, 10)]
    for key, model in models.items():
        accuracies = OrderedDict()
        for trainPct, testPct in zip(dataPcts, dataPcts[::-1]):
            trainVal = int(X.shape[0] * trainPct)
            testVal = int(X.shape[0] * testPct)
            accKey = '{}/{}'.format(int(trainPct * 100), int(testPct * 100))
            Xtrain = X[0:trainVal]
            Ytrain = Y[0:trainVal]
            Xtest = X[trainVal:]
            Ytest = Y[trainVal:]
            accuracies[accKey] = model.fit(Xtrain, Ytrain).score(Xtest, Ytest)
        pyplot.plot(accuracies.keys(), accuracies.values())
        pyplot.ylabel('Accuracy')
        pyplot.xlabel('Train/Test Data Percentage Split')
        pyplot.title('{} Accuracy'.format(key))
        pyplot.savefig('Model Results/{} Accuracy.png'.format(key))
        print('Finished {} analysis'.format(key))

if __name__ == '__main__':
    main()