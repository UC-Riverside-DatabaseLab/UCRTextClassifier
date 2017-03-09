import collections
import sys
from GoogleDatasetBalancer import GoogleDatasetBalancer
from PPDBDatasetBalancer import PPDBDatasetBalancer
from random import shuffle
from RandomForestTextClassifier import RandomForestTextClassifier
from TextDatasetFileParser import TextDatasetFileParser


class TextDatasetBalancer(object):
    def __init__(self, host, database, user, password, input_range, threshold,
                 keys, proxies, engine_id, ppdb=False, google=False,
                 duplicate=True, verbose=False):
        self.__ppdb = None if not ppdb else \
            PPDBDatasetBalancer(host=host, database=database, user=user,
                                password=password, verbose=verbose,
                                threshold=threshold, input_range=input_range)

        self.__google = None if not google else \
            GoogleDatasetBalancer(keys=keys, proxies=proxies, verbose=verbose,
                                  engine_id=engine_id)
        self.__duplicate = duplicate

    def __duplicate_instances(self, data):
        new_instances = []
        classes = []

        for instance in data:
            classes.append(instance.class_value)

        counter = collections.Counter(classes)
        most_common = counter.most_common(1)[0][0]

        for instance in data:
            class_value = instance.class_value

            if instance.class_value != most_common:
                difference = counter[most_common] - counter[class_value]
                num_copies = round(difference / counter[class_value])

                for i in range(0, num_copies):
                    new_instances.append(instance)

        return new_instances

if len(sys.argv) < 2:
    print("Missing argument for path to dataset file.")
    sys.exit()

    def balance(self, data):
        new_instances = []

        if self.__ppdb is not None:
            new_instances += self.__ppdb.balance(data)

        if self.__google is not None:
            new_instances += self.__google.balance(data)

        if self.__duplicate:
            new_instances += self.__duplicate_instances(data)

        return data + new_instances


if len(sys.argv) < 2:
    print("Missing argument for path to dataset file.")
    sys.exit()

data = TextDatasetFileParser().parse(sys.argv[1])
engine_id = "010254973031167365908:s-lpifpdmgs"
keys = []
proxies = []

with open("keys.txt", "r") as file:
    for line in file.readlines():
        split_line = line.replace("\n", "").split(",")

        keys.append(split_line[0])
        proxies.append((split_line[1], 3306))

balancer = TextDatasetBalancer(ppdb=True, google=True, duplicate=False,
                               verbose=False, host="v91.cs.ucr.edu",
                               database="PPDB", user="rriva002",
                               password="passwd", input_range=(2, 3),
                               threshold=4, keys=keys, proxies=proxies,
                               engine_id=engine_id)

shuffle(data)

training_set_end = int(len(data) * 0.9)
training_set = balancer.balance(data[0:training_set_end])
text_classifier = RandomForestTextClassifier(num_jobs=-1, ngram_range=(1, 1))
classes = []

for instance in training_set:
    classes.append(instance.class_value)

counter = collections.Counter(classes)
most_common = counter.most_common(1)[0][0]

for instance in training_set:
    instance.weight = counter[most_common] / counter[instance.class_value]

text_classifier.train(training_set)
text_classifier.evaluate(data[training_set_end:], True)
