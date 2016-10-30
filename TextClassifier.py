from ARFFDataset import ARFFDataset
from RegexClassifier import RegexClassifier
from RandomForestTextClassifier import RandomForestTextClassifier
from random import shuffle
import sys

if len(sys.argv) < 2:
    sys.exit()

data = ARFFDataset(sys.argv[1])
regexClassifier = RegexClassifier(0)
randomForest = RandomForestTextClassifier(0)

data.setClassAttribute(1)
shuffle(data.instances)

instances = data.instances.copy()
data.instances = data.instances[1:int(len(instances) * 0.9)]

regexClassifier.train(data)
randomForest.train(data)

data.instances = instances[int(len(instances) * 0.9) + 1:len(instances) - 1]
regexClassifierCorrect = 0
randomForestCorrect = 0

for instance in data.instances:
    maxClass = None
    maxProbability = 0
    
    for classValue, probability in regexClassifier.classify(instance).items():
        if probability > maxProbability:
            maxClass = classValue
            maxProbability = probability
    
    if maxClass == instance.values[data.classIndex]:
        regexClassifierCorrect = regexClassifierCorrect + 1
    
    maxClass = None
    maxProbability = 0
    
    for classValue, probability in randomForest.classify(instance).items():
        if probability > maxProbability:
            maxClass = classValue
            maxProbability = probability
    
    if maxClass == instance.values[data.classIndex]:
        randomForestCorrect = randomForestCorrect + 1

print("Regex Classifier Accuracy: " + str(regexClassifierCorrect / len(data.instances)))
print("Random Forest Accuracy: " + str(randomForestCorrect / len(data.instances)))