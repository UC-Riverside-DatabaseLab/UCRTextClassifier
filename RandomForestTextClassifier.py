from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

class RandomForestTextClassifier(object):
    def __init__(self, stringAttributeIndex):
        self.stringAttributeIndex = stringAttributeIndex
        self.randomForest = RandomForestClassifier()
        self.vectorizer = TfidfVectorizer()
    
    def train(self, data):
        trainingData = []
        trainingLabels = []
        trainingWeights = []
        
        for instance in data.instances:
            trainingData.append(instance.values[self.stringAttributeIndex])
            trainingLabels.append(instance.values[data.classIndex])
            trainingWeights.append(instance.weight)
        
        self.randomForest.fit(self.vectorizer.fit_transform(trainingData), trainingLabels, trainingWeights)
    
    def classify(self, instance):
        testData = []
        distribution = {}
        
        testData.append(instance.values[self.stringAttributeIndex])
        
        orderedDistribution = self.randomForest.predict_proba(self.vectorizer.transform(testData))
        
        for i in range(0, len(orderedDistribution[0])):
            if orderedDistribution[0, i] > 0:
                distribution[self.randomForest.classes_[i]] = orderedDistribution[0, i]
        
        return distribution