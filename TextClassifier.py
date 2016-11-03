from TextDatasetFileParser import TextDatasetFileParser
from RegexClassifier import RegexClassifier
from RandomForestTextClassifier import RandomForestTextClassifier
from random import shuffle
import sys

if len(sys.argv) < 2:
    sys.exit()

parser = TextDatasetFileParser()
data = parser.parse(sys.argv[1])
regexClassifier = RegexClassifier()
randomForest = RandomForestTextClassifier()

shuffle(data)

instances = data.copy()
data = data[1:int(len(instances) * 0.9)]

regexClassifier.train(data)
randomForest.train(data)

data = instances[int(len(instances) * 0.9) + 1:len(instances) - 1]

print("Regular Expression Classifier:")
regexClassifier.evaluate(data)
print("\nRandom Forest:")
randomForest.evaluate(data)