from abc import ABC, abstractmethod

class AbstractTextClassifier(ABC):
    @abstractmethod
    def train(self, data):
        pass

    @abstractmethod
    def classify(self, instance):
        pass

    def evaluate(self, test_set, verbose=False):
        correct = 0
        weighted_correct = 0
        weighted_total = 0
        confusion_matrix = {}
        column_width = {}
        classes = set()

        for instance in test_set:
            inst_class = instance.class_value
            max_class = None
            max_probability = 0
            weighted_total = weighted_total + instance.weight

            for class_value, probability in self.classify(instance).items():
                if probability > max_probability:
                    max_class = class_value
                    max_probability = probability

                if class_value not in confusion_matrix:
                    confusion_matrix[class_value] = {}

                    for c_val in classes:
                        confusion_matrix[class_value][c_val] = 0

                    classes.add(class_value)

                    for c_val in classes:
                        confusion_matrix[c_val][class_value] = 0

            if max_class == inst_class:
                correct = correct + 1
                weighted_correct = weighted_correct + instance.weight

            if inst_class not in confusion_matrix:
                confusion_matrix[inst_class] = {}

                for class_value in classes:
                    confusion_matrix[inst_class][class_value] = 0

                self.classes.add(inst_class)

                for class_value in classes:
                    confusion_matrix[class_value][inst_class] = 0

            confusion_matrix[inst_class][max_class] = confusion_matrix[inst_class][max_class] + 1

            if verbose and inst_class not in column_width:
                column_width[inst_class] = len(inst_class)

        accuracy = correct / len(test_set)
        weighted_accuracy = weighted_correct / weighted_total

        if verbose:
            classes = list(classes)

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