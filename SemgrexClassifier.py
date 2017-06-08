import json
import math
import socket
import subprocess
from AbstractTextClassifier import AbstractTextClassifier
from collections import Counter
from extractPatterns import PatternExtractor
from nltk import word_tokenize
from nltk.stem.snowball import EnglishStemmer
from PPDBDatasetBalancer import PPDBDatasetBalancer
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
    def __init__(self, backup_classifier=None, generate_patterns=True,
                 paraphrase_arguments=None, imbalance_threshold=0.75,
                 ig_threshold=0, split_sentences=False, use_stemming=False):
        self.__backup_classifier = RandomForestTextClassifier(num_jobs=-1) if \
            backup_classifier is None else backup_classifier
        self.__generate_patterns = generate_patterns
        self.__helper = HelperCommunicator() if generate_patterns else None
        self.__imbalance_threshold = imbalance_threshold
        self.__ig_threshold = ig_threshold
        self.__split_sentences = split_sentences
        self.__stemmer = EnglishStemmer() if use_stemming else None

        if paraphrase_arguments is not None and \
                "host" in paraphrase_arguments and \
                "database" in paraphrase_arguments and \
                "user" in paraphrase_arguments and \
                "password" in paraphrase_arguments:
            host = paraphrase_arguments["host"]
            database = paraphrase_arguments["database"]
            user = paraphrase_arguments["user"]
            password = paraphrase_arguments["password"]
            threshold = paraphrase_arguments["threshold"] \
                if "threshold" in paraphrase_arguments else 4
            input_range = paraphrase_arguments["input_range"] \
                if "input_range" in paraphrase_arguments else (2, 3)
            ig = paraphrase_arguments["ig_threshold"] \
                if "ig_threshold" in paraphrase_arguments else 0.01
            self.__ppdb = PPDBDatasetBalancer(host, database, user, password,
                                              threshold=threshold,
                                              input_range=input_range,
                                              ig_threshold=ig)
        else:
            self.__ppdb = None

    def classify(self, instance):
        distribution = {}

        if self.__generate_patterns:
            payload = {"command": "classify",
                       "text": self.__stem_text(instance.text)}

            self.__helper.send(json.dumps(payload))

            distribution = json.loads(self.__helper.receive())
            distribution = self._normalize_distribution(distribution)

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

    def __stem_text(self, text):
        if self.__stemmer is None:
            return text

        stemmed_text = ""

        for word in word_tokenize(text):
            stemmed_text += (" " if len(stemmed_text) > 0 else
                             "") + self.__stemmer.stem(word)

        return stemmed_text

    def train(self, data):
        if self.__ppdb is not None:
            data = data + self.__ppdb.balance(data)

        # self.__backup_classifier.train(TextDatasetFileParser().parse("./Datasets/ShortWaitTime and LongWaitTime Training.arff"))
        self.__backup_classifier.train(data)

        if not self.__generate_patterns:
            return

        pattern_extractor = PatternExtractor()
        classes = []

        self.__helper.send(json.dumps({"command": "set_mode",
                                       "mode": "train"}))
        self.__helper.send(json.dumps({"command": "split_sentences",
                                       "value": "true" if
                                       self.__split_sentences else "false"}))

        for instance in data:
            classes.append(instance.class_value)

        counter = Counter(classes)
        most_common = counter.most_common(1)[0][0]
        classes = set(classes)
        num_classes = len(classes)

        if(counter[most_common] / len(data) > self.__imbalance_threshold):
            classes.remove(most_common)

        if num_classes > 2:
            for class_value in classes:
                binary_data = []
                ig_words = []
                trees = []

                for instance in data:
                    if instance.class_value == class_value:
                        self.__helper.send(json.dumps({"command": "parse",
                                                       "text": instance.text}))

                        for tree in json.loads(self.__helper.receive()):
                            trees += [tree]

                    binary_class = ("" if instance.class_value == class_value
                                    else "Not") + str(class_value)

                    binary_data.append(Instance(instance.text, binary_class))

                for word in self.__top_information_gain_words(binary_data):
                    ig_words += [word]

                for pattern in pattern_extractor.extract_patterns(ig_words,
                                                                  trees):
                    self.__helper.send(json.dumps({"command": "add_pattern",
                                                   "pattern": pattern,
                                                   "class": str(class_value)}))
        else:
            ig_words = []
            trees = []

            for word in self.__top_information_gain_words(data):
                ig_words += [word]

            for class_value in classes:
                for instance in data:
                    if instance.class_value == class_value:
                        self.__helper.send(json.dumps({"command": "parse",
                                                       "text": instance.text}))

                        for tree in json.loads(self.__helper.receive()):
                            trees += [tree]

                for pattern in pattern_extractor.extract_patterns(ig_words,
                                                                  trees):
                    self.__helper.send(json.dumps({"command": "add_pattern",
                                                   "pattern": pattern,
                                                   "class": str(class_value)}))

        self.__helper.send(json.dumps({"command": "set_mode",
                                       "mode": "evaluate"}))

        for instance in data:
            self.__helper.send(json.dumps({"command": "test",
                                           "text": instance.text,
                                           "class": str(class_value)}))

        self.__helper.send(json.dumps({"command": "set_mode",
                                       "mode": "classify"}))

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

        return top_words


training_file = "./Datasets/ShortWaitTime and LongWaitTime Training.arff"
phrases = "./Datasets/ShortWaitTime and LongWaitTime Phrases.csv"
test_file = "./Datasets/ShortWaitTime and LongWaitTime Test.arff"
textDatasetFileParser = TextDatasetFileParser()
training_set = textDatasetFileParser.parse(training_file)
phrases = textDatasetFileParser.parse(phrases)
test_set = textDatasetFileParser.parse(test_file)
paraphrase_arguments = {"host": "localhost", "database": "PPDB",
                        "user": "rriva002", "password": "passwd"}
classifier = SemgrexClassifier(generate_patterns=True, ig_threshold=0.0025,
                               paraphrase_arguments=None,
                               split_sentences=False, use_stemming=False,
                               imbalance_threshold=0.75)

phrase_training_set = []

for instance in phrases:
    for training_instance in training_set:
        if training_instance.text.find(instance.text) >= 0:
            phrase_training_set.append(instance)
            break

# classifier.train(phrase_training_set)
classifier.train(training_set)
classifier.evaluate(test_set, verbose=True)
