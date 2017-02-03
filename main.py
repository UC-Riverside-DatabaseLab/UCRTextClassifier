import collections
import sys
from random import shuffle
from TextDatasetFileParser import TextDatasetFileParser
from EnsembleTextClassifier import EnsembleTextClassifier, VotingMethod


def balance_dataset(data):
    balanced_data = []
    classes = []

    for instance in data:
        instance.weight = 1

        classes.append(instance.class_value)

    counter = collections.Counter(classes)
    most_common = counter.most_common(1)[0][0]

    for instance in data:
        if instance.class_value == most_common:
            balanced_data.append(instance)
        else:
            difference = counter[most_common] - counter[instance.class_value]
            num_copies = round(difference / counter[instance.class_value]) + 1

            for i in range(0, num_copies):
                balanced_data.append(instance)

    return balanced_data

if len(sys.argv) < 2:
    print("Missing argument for path to dataset file.")
    sys.exit()

data = TextDatasetFileParser().parse(sys.argv[1])
unlabeled_data_file = sys.argv[2] if len(sys.argv) > 2 else None
text_classifier = EnsembleTextClassifier(voting_method=VotingMethod.maximum,
                                         doc2vec_dir="models/",
                                         unlabeled_data=unlabeled_data_file)
training_set_end = int(len(data) * 0.9)
classifiers = ["CNN", "Random Forest", "RegEx", "Word2Vec", "Doc2Vec"]

shuffle(data)

training_set = balance_dataset(data[0:training_set_end])
test_set = data[training_set_end:]

text_classifier.train(training_set)

for i in range(0, len(text_classifier.classifiers)):
    print(classifiers[i] + ":")
    text_classifier.classifiers[i].evaluate(test_set, True)
    print("")

print("Overall:")
text_classifier.evaluate(test_set, True)
