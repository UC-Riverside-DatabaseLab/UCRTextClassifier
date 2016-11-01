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
regexClassifierCorrect = 0
randomForestCorrect = 0

for instance in data:
    maxClass = None
    maxProbability = 0
    
    for classValue, probability in regexClassifier.classify(instance).items():
        if probability > maxProbability:
            maxClass = classValue
            maxProbability = probability
    
    if maxClass == instance.classValue:
        regexClassifierCorrect = regexClassifierCorrect + 1
    
    maxClass = None
    maxProbability = 0
    
    for classValue, probability in randomForest.classify(instance).items():
        if probability > maxProbability:
            maxClass = classValue
            maxProbability = probability
    
    if maxClass == instance.classValue:
        randomForestCorrect = randomForestCorrect + 1

print("Regex Classifier Accuracy: " + str(regexClassifierCorrect / len(data)))
print("Random Forest Accuracy: " + str(randomForestCorrect / len(data)))