from enum import Enum
from random import shuffle
from statistics import mean
from statistics import median
from threading import Thread
from RandomForestTextClassifier import RandomForestTextClassifier
from RegexClassifier import RegexClassifier
from TextDatasetFileParser import TextDatasetFileParser
import sys

class VotingMethod(Enum):
    majority = 1
    average = 2
    median = 3
    product = 4

class ClassifierThread(Thread):
    def __init__(self, classifier, data):
        Thread.__init__(self)
        
        self.classifier = classifier
        self.data = data
    
    def run(self):
        self.classifier.train(data)

class TextClassifier(object):
    def __init__(self, votingMethod = VotingMethod.majority):
        self.classifiers = [RandomForestTextClassifier(), RegexClassifier()]
        self.votingMethod = votingMethod
    
    def train(self, data):
        self.classes = set()
        threads = []
        
        for instance in data:
            self.classes.add(instance.classValue)
        
        for classifier in self.classifiers:
            threads.append(ClassifierThread(classifier, data))
            threads[len(threads) - 1].start()
        
        for thread in threads:
            thread.join()
    
    def classify(self, instance):
        if self.votingMethod == VotingMethod.majority:
            return self.__majority(instance)
        elif self.votingMethod == VotingMethod.average:
            return self.__average(instance)
        elif self.votingMethod == VotingMethod.median:
            return self.__median(instance)
        elif self.votingMethod == VotingMethod.product:
            return self.__product(instance)
    
    def evaluate(self, classifier, testSet, verbose = False):
        correct = 0
        weightedCorrect = 0
        weightedTotal = 0
        confusionMatrix = {}
        columnWidth = {}
        
        for classValue in self.classes:
            confusionMatrix[classValue] = {}
            
            for prediction in self.classes:
                confusionMatrix[classValue][prediction] = 0
        
        for instance in testSet:
            maxClass = None
            maxProbability = 0
            weightedTotal = weightedTotal + instance.weight
            
            for classValue, probability in classifier.classify(instance).items():
                if probability > maxProbability:
                    maxClass = classValue
                    maxProbability = probability
            
            if maxClass == instance.classValue:
                correct = correct + 1
                weightedCorrect = weightedCorrect + instance.weight
            
            if instance.classValue not in confusionMatrix:
                confusionMatrix[instance.classValue] = {}
                
                for classValue in self.classes:
                    confusionMatrix[instance.classValue][classValue] = 0
                
                self.classes.add(instance.classValue)
                
                for classValue in self.classes:
                    confusionMatrix[classValue][instance.classValue] = 0
            
            confusionMatrix[instance.classValue][maxClass] = confusionMatrix[instance.classValue][maxClass] + 1
            
            if verbose and instance.classValue not in columnWidth:
                columnWidth[instance.classValue] = len(instance.classValue)
        
        accuracy = correct / len(testSet)
        weightedAccuracy = weightedCorrect / weightedTotal
        
        if verbose:
            classes = list(self.classes)
            
            classes.sort()
            print(("Accuracy: %0.2f" % (100 * accuracy)) + "%\n" + ("Weighted Accuracy: %0.2f" % (100 * weightedAccuracy)) + "%\nConfusion Matrix:")
            
            for classValue, distribution in confusionMatrix.items():
                for prediction, count in distribution.items():
                    if prediction not in columnWidth or (prediction in columnWidth and len(str(count)) > columnWidth[prediction]):
                        columnWidth[prediction] = len(str(count))
            
            for classValue in classes:
                row = ""
                
                for prediction in classes:
                    for i in range(0, columnWidth[prediction] - len(str(prediction)) + 1):
                        row = row + " "
                    
                    row = row + prediction
                
                print(row + " <- Classified As")
                break
            
            for classValue in classes:
                row = ""
                
                for prediction in classes:
                    for i in range(0, columnWidth[prediction] - len(str(confusionMatrix[classValue][prediction])) + 1):
                        row = row + " "
                    
                    row = row + str(confusionMatrix[classValue][prediction])
                
                print(row + " " + classValue)
        
        return {"accuracy" : accuracy, "weightedaccuracy" : weightedAccuracy, "confusionmatrix" : confusionMatrix}
    
    def __average(self, instance):
        distribution = {}

        for classifier in self.classifiers:
            for prediction, probability in self.__checkDistribution(classifier.classify(instance)).items():
                if prediction not in distribution:
                    distribution[prediction] = []
                
                distribution[prediction].append(probability)
        
        for prediction, probabilities in distribution.items():
            distribution[prediction] = mean(probabilities)
        
        return self.__normalizeDistribution(distribution)
    
    def __majority(self, instance):
        distribution = {}
        
        for classifier in self.classifiers:
            maxClass = None
            maxProbability = 0
            
            for prediction, probability in self.__checkDistribution(classifier.classify(instance)).items():
                if probability > maxProbability:
                    maxClass = prediction
                    maxProbability = probability
                elif probability == maxProbability:
                    maxClass = None
            
            if maxClass not in distribution:
                distribution[maxClass] = 0
            
            distribution[maxClass] = distribution[maxClass] + 1
        
        return self.__normalizeDistribution(distribution)
    
    def __median(self, instance):
        distribution = {}
        
        for classifier in self.classifiers:
            for prediction, probability in self.__checkDistribution(classifier.classify(instance)).items():
                if prediction not in distribution:
                    distribution[prediction] = []
                
                distribution[prediction].append(probability)
        
        for prediction, probabilities in distribution.items():
            distribution[prediction] = median(probabilities)
        
        return self.__normalizeDistribution(distribution)
    
    def __product(self, instance):
        distribution = {}
        
        for classifier in self.classifiers:
            for prediction, probability in self.__checkDistribution(classifier.classify(instance)).items():
                if prediction not in distribution:
                    distribution[prediction] = 1
                
                distribution[prediction] = distribution[prediction] * probability
        
        return self.__normalizeDistribution(distribution)

    def __checkDistribution(self, distribution):
        for classValue in self.classes:
            if classValue not in distribution:
                distribution[classValue] = 0

        return distribution
    
    def __normalizeDistribution(self, distribution):
        sumOfProbabilities = 0
        
        for classValue, probability in distribution.items():
            sumOfProbabilities = sumOfProbabilities + probability
        
        for classValue, probability in distribution.items():
            distribution[classValue] = probability / sumOfProbabilities
        
        return distribution

if len(sys.argv) < 2:
    sys.exit()

data = TextDatasetFileParser().parse(sys.argv[1])
textClassifier = TextClassifier(VotingMethod.product)
trainingSetRatio = 0.9
testSet = data[int(len(data) * trainingSetRatio):]
classifiers = ["Random Forest", "Regular Expression Classifier"]

shuffle(data)
textClassifier.train(data[0:int(len(data) * trainingSetRatio)])

for i in range(0, len(classifiers)):
    print(classifiers[i] + ":")
    textClassifier.evaluate(textClassifier.classifiers[i], testSet, True)
    print("")

print("Overall:")
textClassifier.evaluate(textClassifier, testSet, True)