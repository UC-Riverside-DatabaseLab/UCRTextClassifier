import json
import math
import socket
import subprocess
from AbstractTextClassifier import AbstractTextClassifier
from collections import Counter
from extractPatterns import PatternExtractor
from RandomForestTextClassifier import RandomForestTextClassifier
from sklearn.feature_extraction.text import CountVectorizer
from TextDatasetFileParser import Instance, TextDatasetFileParser


class HelperCommunicator(object):
    def __init__(self, host="localhost", port=9000):
        # path = "./SemgrexClassifierHelper/bin/SemgrexClassifierHelper"
        # self.__helper = subprocess.Popen(["java", path])
        self.__sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.__sock.connect((host, port))

    def send(self, json):
        self.__sock.send((json + "\n").encode())

    def receive(self):
        json = ""

        while len(json) == 0 or not json[len(json) - 1] == "\n":
            json += self.__sock.recv(1).decode("latin-1")

        return json[0:len(json) - 1].replace("__NEWLINE__", "\n")


class SemgrexClassifier(AbstractTextClassifier):
    def __init__(self, backup_classifier=None, ig_threshold=0,
                 imbalance_threshold=0.75):
        self.__backup_classifier = RandomForestTextClassifier(num_jobs=-1) if \
            backup_classifier is None else backup_classifier
        self.__helper = HelperCommunicator()
        self.__ig_threshold = ig_threshold
        self.__imbalance_threshold = imbalance_threshold

    def classify(self, instance):
        self.__helper.send(json.dumps({"mode": "classify",
                                       "text": instance.text}))
        distribution = json.loads(self.__helper.receive())
        return distribution if len(distribution) > 0 else \
            self.__backup_classifier.classify(instance)

    def __entropy(self, vectors, index):
        value_frequencies = {}
        entropy = 0

        for vector in vectors:
            if vector[index] not in value_frequencies:
                value_frequencies[vector[index]] = 0

            value_frequencies[vector[index]] += 1

        for frequency in value_frequencies.values():
            ratio = frequency / len(vectors)
            entropy -= ratio * math.log(ratio, 2)

        return entropy

    def train(self, data):
        pattern_extractor = PatternExtractor()
        classes = []

        self.__helper.send(json.dumps({"mode": "init"}))
        self.__backup_classifier.train(data)

        for instance in data:
            classes.append(instance.class_value)

        counter = Counter(classes)
        most_common = counter.most_common(1)[0][0]
        classes = set(classes)

        if(counter[most_common] / len(data) > self.__imbalance_threshold):
            classes.remove(most_common)

        for class_value in classes:
            binary_data = []
            ig_words = []
            trees = []

            for instance in data:
                if instance.class_value == class_value:
                    self.__helper.send(json.dumps({"mode": "parse",
                                                   "text": instance.text}))

                    for tree in json.loads(self.__helper.receive()):
                        trees += [tree]

                binary_class = ("" if instance.class_value == class_value else
                                "Not") + str(class_value)

                binary_data.append(Instance(instance.text, binary_class))

            for word in self.__top_information_gain_words(binary_data):
                ig_words += [word]

#            with open(class_value + "_trees.txt", "w") as file:
#                for tree in trees:
#                    file.write(tree + "\n---------\n")

            patterns = pattern_extractor.extract_patterns(ig_words, trees)
            print("Pattern extraction complete.")
            for pattern in patterns:
                self.__helper.send(json.dumps({"mode": "add_pattern",
                                               "pattern": pattern,
                                               "class": str(class_value)}))

    def __top_information_gain_words(self, data):
        vectorizer = CountVectorizer(stop_words="english")
        text = []
        top_words = []
        word_frequencies = {}

        for instance in data:
            text.append(instance.text)

        vector_array = vectorizer.fit_transform(text).toarray()
        vectors = []
        words = vectorizer.get_feature_names()
        class_index = len(words)

        for i in range(0, len(data)):
            vector = []

            for value in vector_array[i]:
                vector.append(value)

            vector.append(data[i].class_value)
            vectors.append(vector)

        entropy = self.__entropy(vectors, class_index)

        for i in range(0, len(words)):
            word_frequencies[i] = {}

        for vector in vectors:
            for i in range(0, len(vector) - 1):
                if vector[i] not in word_frequencies[i]:
                    word_frequencies[i][vector[i]] = 0

                word_frequencies[i][vector[i]] += 1

        for index, value_frequencies in word_frequencies.items():
            subset_entropy = 0

            for value, frequency in value_frequencies.items():
                value_probability = frequency / sum(value_frequencies.values())
                sub = [vector for vector in vectors if vector[index] == value]
                subset_entropy += value_probability * \
                    self.__entropy(sub, class_index)

            if entropy - subset_entropy > self.__ig_threshold:
                top_words.append(words[index])
                print(words[index] + "\t" + str(entropy - subset_entropy))

        return top_words


textDatasetFileParser = TextDatasetFileParser()
training_set = textDatasetFileParser.parse("./Datasets/ShortWaitTime and LongWaitTime Training.arff")
test_set = textDatasetFileParser.parse("./Datasets/ShortWaitTime and LongWaitTime Test.arff")
classifier = SemgrexClassifier(ig_threshold=0.0025)

classifier.train(training_set)
classifier.evaluate(test_set, verbose=True)
