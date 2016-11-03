from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

class RandomForestTextClassifier(object):
    def __init__(self, numTrees = 10, criterion = "gini", maxDepth = None, minSamplesSplit = 2, minSamplesLeaf = 1, minWeightFractionLeaf = 0.0, maxFeatures = "auto",
                 maxLeafNodes = None, minImpuritySplit = 1e-07, bootstrap = True, oobScore = False, numJobs = 1, randomState = None, verbose = 0, warmStart = False,
                 classWeight = None):
        self.randomForest = RandomForestClassifier(numTrees, criterion, maxDepth, minSamplesSplit, minSamplesLeaf, minWeightFractionLeaf, maxFeatures, maxLeafNodes,
                                                   minImpuritySplit, bootstrap, oobScore, numJobs, randomState, verbose, warmStart, classWeight)
        self.vectorizer = TfidfVectorizer()
    
    def train(self, data):
        trainingData = []
        trainingLabels = []
        trainingWeights = []
        
        for instance in data:
            trainingData.append(instance.text)
            trainingLabels.append(instance.classValue)
            trainingWeights.append(instance.weight)
        
        self.randomForest.fit(self.vectorizer.fit_transform(trainingData), trainingLabels, trainingWeights)
    
    def classify(self, instance):
        testData = []
        distribution = {}
        
        testData.append(instance.text)
        
        orderedDistribution = self.randomForest.predict_proba(self.vectorizer.transform(testData))
        
        for i in range(0, len(orderedDistribution[0])):
            if orderedDistribution[0, i] > 0:
                distribution[self.randomForest.classes_[i]] = orderedDistribution[0, i]
        
        return distribution
    
    def evaluate(self, testSet):
        correct = 0
        weightedCorrect = 0
        weightedTotal = 0
        confusionMatrix = {}
        columnWidth = {}
        
        for instance in testSet:
            maxClass = None
            maxProbability = 0
            weightedTotal = weightedTotal + instance.weight
            
            for classValue, probability in self.classify(instance).items():
                if probability > maxProbability:
                    maxClass = classValue
                    maxProbability = probability
            
            if maxClass == instance.classValue:
                correct = correct + 1
                weightedCorrect = weightedCorrect + instance.weight
            
            if instance.classValue not in confusionMatrix:
                confusionMatrix[instance.classValue] = {}
            
            for classValue, distribution in confusionMatrix.items():
                if instance.classValue not in distribution:
                    confusionMatrix[classValue][instance.classValue] = 0
            
            if maxClass in confusionMatrix[instance.classValue]:
                confusionMatrix[instance.classValue][maxClass] = confusionMatrix[instance.classValue][maxClass] + 1
            else:
                confusionMatrix[instance.classValue][maxClass] = 1
            
            if instance.classValue not in columnWidth:
                columnWidth[instance.classValue] = len(instance.classValue)
        
        accuracy = correct / len(testSet)
        weightedAccuracy = weightedCorrect / weightedTotal
        
        print(("Accuracy: %0.2f" % (100 * accuracy)) + "%\n" + ("Weighted Accuracy: %0.2f" % (100 * weightedAccuracy)) + "%\nConfusion Matrix:")
        
        for classValue, distribution in confusionMatrix.items():
            for prediction, count in distribution.items():
                if prediction not in columnWidth or (prediction in columnWidth and len(str(count)) > columnWidth[prediction]):
                    columnWidth[prediction] = len(str(count))
        
        for classValue, distribution in confusionMatrix.items():
            row = ""
            
            for prediction, count in distribution.items():
                for i in range(0, columnWidth[prediction] - len(str(prediction)) + 1):
                    row = row + " "
                
                row = row + prediction
            
            print(row + " <- Classified As")
            break
        
        for classValue, distribution in confusionMatrix.items():
            row = ""
            
            for prediction, count in distribution.items():
                for i in range(0, columnWidth[prediction] - len(str(count)) + 1):
                    row = row + " "
                
                row = row + str(confusionMatrix[classValue][prediction])
            
            print(row + " " + classValue)