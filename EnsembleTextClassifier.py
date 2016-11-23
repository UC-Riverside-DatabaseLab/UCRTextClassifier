import sys
from enum import Enum
from random import shuffle
from threading import Thread
from statistics import mean, median
from Word2VecSimilarity import Word2VecSimilarity
from TextDatasetFileParser import TextDatasetFileParser
from AbstractTextClassifier import AbstractTextClassifier
from RandomForestTextClassifier import RandomForestTextClassifier
from RegexClassifier import RegexClassifier


class VotingMethod(Enum):
    """Enumeration of ensemble combination methods. Given an instance to
    classify, the classifier returns a probability distribution as follows:

    average - The predicted probability of each class is the average of
        predicted probabilities for that class among all the classifiers.
    majority - The predicted probability of each class is the ratio of
        classifiers that predicted the instance as that class.
    maximum - The predicted probability of each class is the maximum
    probability for that class among all the classifiers.
    median - The predicted probability of each class is the median of
        predicted probabilities for that class among all the classifiers.
    product - The predicted probability of each class is the product of
        predicted probabilities for that class among all the classifiers.
    """
    average = 1
    majority = 2
    maximum = 3
    median = 4
    product = 5


class ClassifierThread(Thread):
    """Thread class for classifier training.

    Constructor arguments:
    classifier - The classifier that this thread will run
    data - A list of Instance objects (defined in TextDataSetFileParser.py)
    """
    def __init__(self, classifier, data):
        Thread.__init__(self)

        self.classifier = classifier
        self.data = data

    def run(self):
        self.classifier.train(data)


class EnsembleTextClassifier(AbstractTextClassifier):
    """Combines output of multiple text classifiers based on a user-selected
    combination method.

    Constructor arguments:
    voting_method (default majority) - The combination method to use
    use_weights (default True) - If True, apply a weight to each classifier's
    output based on its accuracy
    unlabeled_data (default None) - Path to a file containing unlabeled data
    for classifiers with unsupervised learning. If None, those classifers won't
    be used.
    """
    def __init__(self, voting_method=VotingMethod.majority, use_weights=True,
                 unlabeled_data=None):
        self.classifiers = [RandomForestTextClassifier(), RegexClassifier()]
        self.classifier_weights = []
        self.voting_method = voting_method
        self.use_weights = use_weights

        if unlabeled_data is not None:
            self.classifiers.append(Word2VecSimilarity(unlabeled_data))

        while len(self.classifier_weights) < len(self.classifiers):
            self.classifier_weights.append(1)

    def train(self, data, calculate_training_weights=False):
        """Train the classifier on the given training set.

        Arguments:
        data - A list of Instance objects (defined in TextDataSetFileParser.py)
        calculate_training_weights (default False) - If True, calculate weights
        based on the distribution of classes in the training data
        Returns:
        Nothing
        """
        self.classes = set()
        threads = []

        for instance in data:
            self.classes.add(instance.class_value)

        if calculate_training_weights:
            freq = {}
            max_class = None

            for class_value in self.classes:
                freq[class_value] = 0

            for instance in data:
                freq[class_value] += 1

                if max_class is None or freq[class_value] > freq[max_class]:
                    max_class = class_value

            for instance in data:
                instance.weight = freq[max_class] / freq[instance.class_value]

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
                key = "weightedaccuracy"
                accuracy = self.classifiers[i].evaluate(validation_set)[key]
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
            predictions = self.classifiers[i].classify(instance)
            predictions = self.__check_distribution(predictions)

            for prediction, probability in predictions.items():
                weighted_probability = probability * self.classifier_weights[i]

                if prediction not in distribution:
                    distribution[prediction] = []

                distribution[prediction].append(weighted_probability)

        for prediction, probabilities in distribution.items():
            distribution[prediction] = mean(probabilities)

        return self._normalize_distribution(distribution)

    def __majority(self, instance):
        distribution = {}

        for i in range(0, len(self.classifiers)):
            max_class = None
            max_probability = 0
            predictions = self.classifiers[i].classify(instance)
            predictions = self.__check_distribution(predictions)

            for prediction, probability in predictions.items():
                if probability > max_probability:
                    max_class = prediction
                    max_probability = probability

            if max_class not in distribution:
                distribution[max_class] = 0

            distribution[max_class] += self.classifier_weights[i]

        return self._normalize_distribution(distribution)

    def __maximum(self, instance):
        distribution = {}

        for i in range(0, len(self.classifiers)):
            predictions = self.classifiers[i].classify(instance)
            predictions = self.__check_distribution(predictions)

            for prediction, probability in predictions.items():
                probability = probability * self.classifier_weights[i]

                if prediction in distribution:
                    if probability < distribution[prediction]:
                        continue

                distribution[prediction] = probability

        return self._normalize_distribution(distribution)

    def __median(self, instance):
        distribution = {}

        for i in range(0, len(self.classifiers)):
            predictions = self.classifiers[i].classify(instance)
            predictions = self.__check_distribution(predictions)

            for prediction, probability in predictions.items():
                weighted_probability = probability * self.classifier_weights[i]

                if prediction not in distribution:
                    distribution[prediction] = []

                distribution[prediction].append(weighted_probability)

        for prediction, probabilities in distribution.items():
            distribution[prediction] = median(probabilities)

        return self._normalize_distribution(distribution)

    def __product(self, instance):
        distribution = {}

        for i in range(0, len(self.classifiers)):
            predictions = self.classifiers[i].classify(instance)
            predictions = self.__check_distribution(predictions)

            for prediction, probability in predictions.items():
                if prediction not in distribution:
                    distribution[prediction] = 1

                weighted_probability = probability * self.classifier_weights[i]
                distribution[prediction] *= weighted_probability

        return self._normalize_distribution(distribution)

    def __check_distribution(self, distribution):
        for class_value in self.classes:
            if class_value not in distribution:
                distribution[class_value] = 0

        return distribution

if len(sys.argv) < 3:
    sys.exit()

data = TextDatasetFileParser().parse(sys.argv[1])
text_classifier = EnsembleTextClassifier(VotingMethod.maximum, True,
                                         sys.argv[2])
training_set_end = int(len(data) * 0.9)
classifiers = ["Random Forest", "Regular Expressions", "Word2Vec"]

shuffle(data)
text_classifier.train(data[0:training_set_end])

test_set = data[training_set_end:]

for i in range(0, len(text_classifier.classifiers)):
    print(classifiers[i] + ":")
    text_classifier.classifiers[i].evaluate(test_set, True)
    print("")

print("Overall:")
text_classifier.evaluate(test_set, True)
