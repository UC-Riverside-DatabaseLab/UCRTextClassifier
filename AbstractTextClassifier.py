from abc import ABC, abstractmethod
from random import random


class AbstractTextClassifier(ABC):
    """Abstract class for text classifiers."""
    @abstractmethod
    def train(self, data):
        """Train the classifier on the given training set.

        Arguments:
        data - A list of Instance objects (defined in TextDataSetFileParser.py)

        Returns:
        Nothing
        """
        pass

    @abstractmethod
    def classify(self, instance):
        """Determine the probability distribution of the given instance.

        Arguments:
        instance - An instance object (defined in TextDataSetFileParser.py)

        Returns:
        A dictionary with class strings as keys and probabilities as values
        """
        pass

    def evaluate(self, test_set, verbose=False, output_matrix=None):
        """Evaluate the classifier's performance on the given test set.

        Arguments:
        test_set - A list of Instance objects (defined in
        TextDataSetFileParser.py)
        verbose (default False) - If True, print the results of the evaluation
        output_matrix (default None) - A dictionary that is populated with
        an X marking each incorrectly classified instance (second key) for each
        class (first key) for each classifier in the ensemble.

        Returns:
        A dictionary with the following key-value pairs:
            accuracy - The ratio of correctly classified instances
            weightedaccuracy - The ratio of weights of correctly classified
            instances
            confusionmatrix - A "two-dimensional dictionary" where matrix[A][B]
            yields the number of instances of class A that were classified
            as class B by the classifier
        """
        correct = 0
        weighted_correct = 0
        weighted_total = 0
        confusion_matrix = {}
        column_width = {}
        classes = set()

        for instance in test_set:
            max_class = None
            max_probability = -1
            weighted_total += instance.weight

            for class_value, probability in self.classify(instance).items():
                if (probability > max_probability or
                        probability == max_probability and random() < 0.5):
                    max_class = class_value
                    max_probability = probability

                if class_value not in confusion_matrix:
                    confusion_matrix[class_value] = {}

                    for c_val in classes:
                        confusion_matrix[class_value][c_val] = 0

                    classes.add(class_value)

                    for c_val in classes:
                        confusion_matrix[c_val][class_value] = 0

            if max_class == instance.class_value:
                correct += 1
                weighted_correct += instance.weight

                if output_matrix is not None:
                    if max_class not in output_matrix:
                        output_matrix[max_class] = {}

                    if instance.text not in output_matrix[max_class]:
                        output_matrix[max_class][instance.text] = []

                    output_matrix[max_class][instance.text].append("")
            elif output_matrix is not None:
                if instance.class_value not in output_matrix:
                    output_matrix[instance.class_value] = {}

                if instance.text not in output_matrix[instance.class_value]:
                    output_matrix[instance.class_value][instance.text] = []

                output_matrix[instance.class_value][instance.text].append("X")

            if instance.class_value not in confusion_matrix:
                confusion_matrix[instance.class_value] = {}

                for class_value in classes:
                    confusion_matrix[instance.class_value][class_value] = 0

                classes.add(instance.class_value)

                for class_value in classes:
                    confusion_matrix[class_value][instance.class_value] = 0

            confusion_matrix[instance.class_value][max_class] += 1

            if verbose and instance.class_value not in column_width:
                column_width[instance.class_value] = len(instance.class_value)

        accuracy = correct / len(test_set)
        sum_accuracies = 0.0
        adjustment = 0.0

        for c1 in confusion_matrix:
            TC = confusion_matrix[c1][c1]
            C = 0.0

            for c2 in confusion_matrix[c1]:
                C += confusion_matrix[c1][c2]

            if C > 0.0:
                sum_accuracies += TC / C
            else:
                adjustment += 1

        weighted_acc = sum_accuracies / (len(confusion_matrix) - adjustment)
        # weighted_acc = weighted_correct / weighted_total

        if verbose:
            classes = list(classes)

            classes.sort()
            print(("Accuracy: %0.2f" % (100 * accuracy)) + "%")
            print(("Weighted Accuracy: %0.2f" % (100 * weighted_acc)) + "%")
            print("Confusion Matrix:")

            for class_value, distribution in confusion_matrix.items():
                for prediction, count in distribution.items():
                    if prediction not in column_width:
                        width = max(len(prediction), len(str(count)))
                        column_width[prediction] = width
                    elif prediction in column_width:
                        if len(str(count)) > column_width[prediction]:
                            column_width[prediction] = len(str(count))

            row = ""

            for prediction in classes:
                width = column_width[prediction] - len(str(prediction)) + 1

                for i in range(0, width):
                    row += " "

                row += prediction

            print(row + " <- Classified As")

            for class_value in classes:
                row = ""

                for prediction in classes:
                    str_val = str(confusion_matrix[class_value][prediction])
                    width = column_width[prediction] - len(str_val) + 1

                    for i in range(0, width):
                        row += " "

                    row += str(confusion_matrix[class_value][prediction])

                print(row + " " + class_value)

        return {"accuracy": accuracy, "weightedaccuracy": weighted_acc,
                "confusionmatrix": confusion_matrix}

    def _normalize_distribution(self, distribution):
        sum_of_probabilities = 0

        for class_value, probability in distribution.items():
            sum_of_probabilities += probability

        if sum_of_probabilities > 0:
            for class_value, probability in distribution.items():
                distribution[class_value] = probability / sum_of_probabilities

        return distribution
