
#import sklearn

#import numpy
#import gensim
from EnsembleTextClassifier import EnsembleTextClassifier,VotingMethod
from TextDatasetFileParser import  TextDatasetFileParser
from random import shuffle


data = TextDatasetFileParser().parse('Datasets/ecig_sentiment_test.csv')
unlabeled_data_file = 'Datasets/review_text.csv'

text_classifier = EnsembleTextClassifier(voting_method=VotingMethod.maximum, unlabeled_data=unlabeled_data_file)
training_set_end = int(len(data) * 0.9)
classifiers = ["CNN", "Random Forest", "Regular Expressions", "Word2Vec"]
#classifiers = ["Random Forest", "Regular Expressions", "Word2Vec"]

shuffle(data)
text_classifier.train(data[0:training_set_end])

test_set = data[training_set_end:]

for i in range(0, len(text_classifier.classifiers)):
    print(classifiers[i] + ":")
    if classifiers[i] == 'CNN':
        print('here')
    text_classifier.classifiers[i].evaluate(test_set, True)
    print("")
