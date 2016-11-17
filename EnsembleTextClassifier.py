import sys
from enum import Enum
from random import shuffle
from threading import Thread
from statistics import mean, median
from TextDatasetFileParser import TextDatasetFileParser
from AbstractTextClassifier import AbstractTextClassifier
from RandomForestTextClassifier import RandomForestTextClassifier
from RegexClassifier import RegexClassifier

class VotingMethod(Enum):
    average = 1
    majority = 2
    maximum = 3
    median = 4
    product = 5

class ClassifierThread(Thread):
    def __init__(self, classifier, data):
        Thread.__init__(self)

        self.classifier = classifier
        self.data = data

    def run(self):
        self.classifier.train(data)

class EnsembleTextClassifier(AbstractTextClassifier):
    def __init__(self, voting_method=VotingMethod.majority, use_weights=True):
        self.classifiers = [RandomForestTextClassifier(), RegexClassifier()]
        self.classifier_weights = []
        self.voting_method = voting_method
        self.use_weights = use_weights

        while len(self.classifier_weights) < len(self.classifiers):
            self.classifier_weights.append(1)

    def train(self, data):
        self.classes = set()
        threads = []

        for instance in data:
            self.classes.add(instance.class_value)

        if self.use_weights:
            training_set_size = int(0.9 * len(data))
            validation_set = data[training_set_size:]
            data = data[0:training_set_size]

        for classifier in self.classifiers:
            threads.append(ClassifierThread(classifier, data))
            threads[len(threads) - 1].start()

        for thread in threads:
            thread.join()

        if self.use_weights:
            for i in range(0, len(self.classifiers)):
                accuracy = self.classifiers[i].evaluate(validation_set)["weightedaccuracy"]
                self.classifier_weights[i] = accuracy

    def classify(self, instance):
        if self.voting_method == VotingMethod.average:
            return self.__average(instance)
        elif self.voting_method == VotingMethod.majority:
            return self.__majority(instance)
        elif self.voting_method == VotingMethod.maximum:
            return self.__maximum(instance)
        elif self.voting_method == VotingMethod.median:
            return self.__median(instance)
        elif self.voting_method == VotingMethod.product:
            return self.__product(instance)

    def __average(self, instance):
        distribution = {}

        for i in range(0, len(self.classifiers)):
            predictions = self.__check_distribution(self.classifiers[i].classify(instance))

            for prediction, probability in predictions.items():
                if prediction not in distribution:
                    distribution[prediction] = []

                distribution[prediction].append(probability * self.classifier_weights[i])

        for prediction, probabilities in distribution.items():
            distribution[prediction] = mean(probabilities)

        return self.__normalize_distribution(distribution)

    def __majority(self, instance):
        distribution = {}

        for i in range(0, len(self.classifiers)):
            max_class = None
            max_probability = 0
            predictions = self.__check_distribution(self.classifiers[i].classify(instance))

            for prediction, probability in predictions.items():
                if probability > max_probability:
                    max_class = prediction
                    max_probability = probability

            if max_class not in distribution:
                distribution[max_class] = 0

            distribution[max_class] = distribution[max_class] + self.classifier_weights[i]

        return self.__normalize_distribution(distribution)

    def __maximum(self, instance):
        distribution = {}

        for i in range(0, len(self.classifiers)):
            predictions = self.__check_distribution(self.classifiers[i].classify(instance))

            for prediction, probability in predictions.items():
                probability = probability * self.classifier_weights[i]

                if prediction not in distribution or probability > distribution[prediction]:
                    distribution[prediction] = probability

        return self.__normalize_distribution(distribution)

    def __median(self, instance):
        distribution = {}

        for i in range(0, len(self.classifiers)):
            predictions = self.__check_distribution(self.classifiers[i].classify(instance))

            for prediction, probability in predictions.items():
                if prediction not in distribution:
                    distribution[prediction] = []

                distribution[prediction].append(probability * self.classifier_weights[i])

        for prediction, probabilities in distribution.items():
            distribution[prediction] = median(probabilities)

        return self.__normalize_distribution(distribution)

    def __product(self, instance):
        distribution = {}

        for i in range(0, len(self.classifiers)):
            predictions = self.__check_distribution(self.classifiers[i].classify(instance))

            for prediction, probability in predictions.items():
                if prediction not in distribution:
                    distribution[prediction] = 1

                distribution[prediction] = distribution[prediction] * probability
                distribution[prediction] = distribution[prediction] * self.classifier_weights[i]

        return self.__normalize_distribution(distribution)

    def __check_distribution(self, distribution):
        for class_value in self.classes:
            if class_value not in distribution:
                distribution[class_value] = 0

        return distribution

    def __normalize_distribution(self, distribution):
        sum_of_probabilities = 0

        for class_value, probability in distribution.items():
            sum_of_probabilities = sum_of_probabilities + probability

        for class_value, probability in distribution.items():
            distribution[class_value] = probability / sum_of_probabilities

        return distribution

if len(sys.argv) < 2:
    sys.exit()

data = TextDatasetFileParser().parse(sys.argv[1])
text_classifier = TextClassifier(VotingMethod.maximum)
training_set_end = int(len(data) * 0.9)
classifiers = ["Random Forest", "Regular Expression Classifier"]

shuffle(data)
text_classifier.train(data[0:training_set_end])

test_set = data[training_set_end:]

for i in range(0, len(classifiers)):
    print(classifiers[i] + ":")
    text_classifier.classifiers[i].evaluate(test_set, True)
    print("")

print("Overall:")
text_classifier.evaluate(test_set, True)
