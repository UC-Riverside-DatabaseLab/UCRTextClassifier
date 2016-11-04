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