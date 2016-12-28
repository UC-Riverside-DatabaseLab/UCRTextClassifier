import sys
from random import shuffle
from TextDatasetFileParser import TextDatasetFileParser
from EnsembleTextClassifier import EnsembleTextClassifier, VotingMethod

if len(sys.argv) < 2:
    print("Missing argument for path to dataset file.")
    sys.exit()

data = TextDatasetFileParser().parse(sys.argv[1])
unlabeled_data_file = sys.argv[2] if len(sys.argv) > 2 else None
text_classifier = EnsembleTextClassifier(voting_method=VotingMethod.maximum,
                                         unlabeled_data=unlabeled_data_file)
training_set_end = int(len(data) * 0.9)
classifiers = ["CNN", "Random Forest", "RegEx", "Word2Vec", "Doc2Vec"]

shuffle(data)
text_classifier.train(data[0:training_set_end])

test_set = data[training_set_end:]

for i in range(0, len(text_classifier.classifiers)):
    print(classifiers[i] + ":")
    text_classifier.classifiers[i].evaluate(test_set, True)
    print("")

print("Overall:")
text_classifier.evaluate(test_set, True)
