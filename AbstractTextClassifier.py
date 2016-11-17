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
            max_class = None
            max_probability = 0
            weighted_total += instance.weight

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

            if max_class == instance.class_value:
                correct += 1
                weighted_correct += instance.weight

            if instance.class_value not in confusion_matrix:
                confusion_matrix[instance.class_value] = {}

                for class_value in classes:
                    confusion_matrix[instance.class_value][class_value] = 0

                self.classes.add(instance.class_value)

                for class_value in classes:
                    confusion_matrix[class_value][instance.class_value] = 0

            confusion_matrix[instance.class_value][max_class] += 1

            if verbose and instance.class_value not in column_width:
                column_width[instance.class_value] = len(instance.class_value)

        accuracy = correct / len(test_set)
        weighted_acc = weighted_correct / weighted_total

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

            for class_value in classes:
                row = ""

                for prediction in classes:
                    width = column_width[prediction] - len(str(prediction)) + 1

                    for i in range(0, width):
                        row = row + " "

                    row = row + prediction

                print(row + " <- Classified As")
                break

            for class_value in classes:
                row = ""

                for prediction in classes:
                    str_val = str(confusion_matrix[class_value][prediction])
                    width = column_width[prediction] - len(str_val) + 1

                    for i in range(0, width):
                        row = row + " "

                    row = row + str(confusion_matrix[class_value][prediction])

                print(row + " " + class_value)

        return {"accuracy": accuracy, "weightedaccuracy": weighted_acc,
                "confusionmatrix": confusion_matrix}
