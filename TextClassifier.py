import sys
from enum import Enum
from random import shuffle
from statistics import mean
from statistics import median
from threading import Thread
from RandomForestTextClassifier import RandomForestTextClassifier
from RegexClassifier import RegexClassifier
from TextDatasetFileParser import TextDatasetFileParser

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
    def __init__(self, voting_method=VotingMethod.majority):
        self.classifiers = [RandomForestTextClassifier(), RegexClassifier()]
        self.voting_method = voting_method

    def train(self, data):
        self.classes = set()
        threads = []

        for instance in data:
            self.classes.add(instance.class_value)

        for classifier in self.classifiers:
            threads.append(ClassifierThread(classifier, data))
            threads[len(threads) - 1].start()

        for thread in threads:
            thread.join()

    def classify(self, instance):
        if self.voting_method == VotingMethod.majority:
            return self.__majority(instance)
        elif self.voting_method == VotingMethod.average:
            return self.__average(instance)
        elif self.voting_method == VotingMethod.median:
            return self.__median(instance)
        elif self.voting_method == VotingMethod.product:
            return self.__product(instance)

    def evaluate(self, classifier, test_set, verbose=False):
        correct = 0
        weighted_correct = 0
        weighted_total = 0
        confusion_matrix = {}
        column_width = {}

        for class_value in self.classes:
            confusion_matrix[class_value] = {}

            for prediction in self.classes:
                confusion_matrix[class_value][prediction] = 0

        for instance in test_set:
            inst_class = instance.class_value
            max_class = None
            max_probability = 0
            weighted_total = weighted_total + instance.weight

            for class_value, probability in classifier.classify(instance).items():
                if probability > max_probability:
                    max_class = class_value
                    max_probability = probability

            if max_class == instance.class_value:
                correct = correct + 1
                weighted_correct = weighted_correct + instance.weight

            if instance.class_value not in confusion_matrix:
                confusion_matrix[inst_class] = {}

                for class_value in self.classes:
                    confusion_matrix[inst_class][class_value] = 0

                self.classes.add(inst_class)

                for class_value in self.classes:
                    confusion_matrix[class_value][inst_class] = 0

            confusion_matrix[inst_class][max_class] = confusion_matrix[inst_class][max_class] + 1

            if verbose and instance.class_value not in column_width:
                column_width[inst_class] = len(inst_class)

        accuracy = correct / len(test_set)
        weighted_accuracy = weighted_correct / weighted_total

        if verbose:
            classes = list(self.classes)

            classes.sort()
            print(("Accuracy: %0.2f" % (100 * accuracy)) + "%")
            print(("Weighted Accuracy: %0.2f" % (100 * weighted_accuracy)) + "%\nConfusion Matrix:")

            for class_value, distribution in confusion_matrix.items():
                for prediction, count in distribution.items():
                    if prediction not in column_width:
                        column_width[prediction] = len(str(count))
                    elif prediction in column_width and len(str(count)) > column_width[prediction]:
                        column_width[prediction] = len(str(count))

            for class_value in classes:
                row = ""

                for prediction in classes:
                    for i in range(0, column_width[prediction] - len(str(prediction)) + 1):
                        row = row + " "

                    row = row + prediction

                print(row + " <- Classified As")
                break

            for class_value in classes:
                row = ""

                for prediction in classes:
                    max_column_width = len(str(confusion_matrix[class_value][prediction]))

                    for i in range(0, column_width[prediction] - max_column_width + 1):
                        row = row + " "

                    row = row + str(confusion_matrix[class_value][prediction])

                print(row + " " + class_value)

        return {"accuracy" : accuracy, "weightedaccuracy" : weighted_accuracy,
                "confusionmatrix" : confusion_matrix}

    def __average(self, instance):
        distribution = {}

        for classifier in self.classifiers:
            predictions = self.__check_distribution(classifier.classify(instance))

            for prediction, probability in predictions.items():
                if prediction not in distribution:
                    distribution[prediction] = []

                distribution[prediction].append(probability)

        for prediction, probabilities in distribution.items():
            distribution[prediction] = mean(probabilities)

        return self.__normalize_distribution(distribution)

    def __majority(self, instance):
        distribution = {}

        for classifier in self.classifiers:
            max_class = None
            max_probability = 0
            predictions = self.__check_distribution(classifier.classify(instance))

            for prediction, probability in predictions.items():
                if probability > max_probability:
                    max_class = prediction
                    max_probability = probability
                elif probability == max_probability:
                    max_class = None

            if max_class not in distribution:
                distribution[max_class] = 0

            distribution[max_class] = distribution[max_class] + 1

        return self.__normalize_distribution(distribution)

    def __median(self, instance):
        distribution = {}

        for classifier in self.classifiers:
            predictions = self.__check_distribution(classifier.classify(instance))

            for prediction, probability in predictions.items():
                if prediction not in distribution:
                    distribution[prediction] = []

                distribution[prediction].append(probability)

        for prediction, probabilities in distribution.items():
            distribution[prediction] = median(probabilities)

        return self.__normalize_distribution(distribution)

    def __product(self, instance):
        distribution = {}

        for classifier in self.classifiers:
            predictions = self.__check_distribution(classifier.classify(instance))

            for prediction, probability in predictions.items():
                if prediction not in distribution:
                    distribution[prediction] = 1

                distribution[prediction] = distribution[prediction] * probability

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
text_classifier = TextClassifier(VotingMethod.product)
training_set_ratio = 0.9
test_set = data[int(len(data) * training_set_ratio):]
classifiers = ["Random Forest", "Regular Expression Classifier"]

shuffle(data)
text_classifier.train(data[0:int(len(data) * training_set_ratio)])

for i in range(0, len(classifiers)):
    print(classifiers[i] + ":")
    text_classifier.evaluate(text_classifier.classifiers[i], test_set, True)
    print("")

print("Overall:")
text_classifier.evaluate(text_classifier, test_set, True)
