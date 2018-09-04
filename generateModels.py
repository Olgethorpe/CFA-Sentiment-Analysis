from collections import OrderedDict
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path
from pandas import read_csv
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

vectorizers = OrderedDict([('Uni-gram', CountVectorizer())])# ('Bi-Gram', CountVectorizer(ngram_range = (1, 2)))])
models = OrderedDict([('Logistic Regression', LogisticRegression()), ('Multinomial Naive Bayes', MultinomialNB())])
                      #('Random Forest', RandomForestClassifier())])
dataFile = Path('Data/Tweets.csv')
data = read_csv(dataFile)
figNum = 0
iterations = 100
for vectorizerKey, vectorizer in vectorizers.items():
    X = vectorizer.fit_transform(data.text.values)
    Y = data.airline_sentiment.values
    for modelKey, model in models.items():
        T0Acc = OrderedDict()
        T1Acc= OrderedDict()
        T2Acc = OrderedDict()
        # Handles graphing and getting the models for the accuracy graphs
        for trainSize in [(x * 100 * .1) / 100 for x in range(1, 10)]:
            trainVal = int(trainSize * 100)
            testVal = int(100 - trainVal)
            testSize = round(1 - trainSize, 2)   
            t0 = 0 
            t1 = 0
            t2 = 0
            for iter in range(iterations):
                X_0Train, X_0Test, Y_0Train, Y_0Test = train_test_split(X, Y, test_size = testSize)
                X_1Test, X_2Test, Y_1Test, Y_2Test = train_test_split(X_0Test, Y_0Test, test_size = testSize)
                model.fit(X_0Train, Y_0Train)
                t0 += model.score(X_0Test, Y_0Test)
                t1 += model.score(X_1Test, Y_1Test)
                t2 += model.score(X_2Test, Y_2Test)
                print('Iteration {} of {} for {} has completed for size {}/{}'.format(iter, iterations, modelKey, trainVal, testVal))
            T0Acc['{}/{}'.format(trainVal, testVal)] = t0 / iterations
            T1Acc['{}/{}'.format(trainVal, testVal)] = t1 / iterations
            T2Acc['{}/{}'.format(trainVal, testVal)] = t2 / iterations
        pyplot.figure(figNum)
        pyplot.plot(T0Acc.keys(), T0Acc.values(), label = '{}'.format('T0 Avg Accuracy'))
        pyplot.plot(T1Acc.keys(), T1Acc.values(), label = '{}'.format('T1 Avg Accuracy'))
        pyplot.plot(T2Acc.keys(), T2Acc.values(), label = '{}'.format('T2 Avg Accuracy'))
        pyplot.legend()
        pyplot.ylabel('Accuracy')
        pyplot.xlabel('Train/Test Data Percentage Split') 
        pyplot.title('{} Accuracy'.format(modelKey))
        pyplot.savefig('Model Results/Avg Accuracy/{} Accuracy.png'.format(modelKey))
        print('Finished {} analysis for {}'.format(modelKey, vectorizerKey))
        figNum += 1