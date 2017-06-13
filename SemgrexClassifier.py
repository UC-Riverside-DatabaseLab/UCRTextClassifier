import json
import math
import socket
# import subprocess
from AbstractTextClassifier import AbstractTextClassifier
from collections import Counter
from extractPatterns import PatternExtractor
from nltk import word_tokenize
from nltk.stem.snowball import EnglishStemmer
from PPDBDatasetBalancer import PPDBDatasetBalancer
from RandomForestTextClassifier import RandomForestTextClassifier
from sklearn.feature_extraction.text import CountVectorizer
from TextDatasetFileParser import Instance, TextDatasetFileParser


class NlpService(object):
    def __init__(self, host="localhost", port=9000):
        self.__sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # path = "./SemgrexClassifierHelper/bin/SemgrexClassifierHelper"

        # subprocess.Popen(["java", path])
        self.__sock.connect((host, port))

    def __receive(self):
        data = ""

        while len(data) == 0 or not data[len(data) - 1] == "\n":
            data += self.__sock.recv(1).decode("latin-1")

        return json.loads(data[0:len(data) - 1].replace("__NEWLINE__", "\n"))

    def __send(self, command, data):
        data["command"] = command

        self.__sock.send((json.dumps(data) + "\n").encode())

    def add_pattern(self, pattern, class_value):
        self.__send("add_pattern", {"pattern": pattern, "class": class_value})

    def classify(self, text):
        self.__send("classify", {"text": text})
        return self.__receive()

    def has_pattern(self, pattern, class_value):
        self.__send("has_pattern", {"pattern": pattern, "class": class_value})
        return self.__receive()

    def parse(self, text):
        self.__send("parse", {"text": text})
        return self.__receive()

    def set_mode(self, mode):
        self.__send("set_mode", {"mode": mode})

    def set_split_sentences(self, split):
        self.__send("split_sentences", {"value": split})

    def test(self, text, class_value):
        self.__send("test", {"text": text, "class": class_value})


class SemgrexClassifier(AbstractTextClassifier):
    def __init__(self, backup_classifier=None, generate_patterns=True,
                 nlp_host="localhost", paraphrase_arguments=None,
                 imbalance_threshold=0.75, ig_threshold=0,
                 split_sentences=False, max_iterations=1, use_stemming=False):
        self.__backup_classifier = RandomForestTextClassifier() if \
            backup_classifier is None else backup_classifier
        self.__generate_patterns = generate_patterns
        self.__nlp = NlpService(host=nlp_host) if generate_patterns else None
        self.__imbalance_threshold = imbalance_threshold
        self.__ig_threshold = ig_threshold
        self.__split_sentences = split_sentences
        self.__max_iterations = max(1, max_iterations)
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

    def __top_information_gain_words(self, data, threshold):
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

            if entropy - subset_entropy > threshold:
                top_words.append(words[index])

        return top_words

    def classify(self, instance):
        distribution = {}

        if self.__generate_patterns:
            text = self.__stem_text(instance.text)
            distribution = self.__nlp.classify(text)
            distribution = self._normalize_distribution(distribution)

        return distribution if len(distribution) > 0 else \
            self.__backup_classifier.classify(instance)

    def set_generate_patterns(self, enable):
        self.__generate_patterns = enable

    def train(self, data):
        classes = []

        for instance in data:
            classes.append(instance.class_value)

        counter = Counter(classes)
        most_common = counter.most_common(1)[0][0]
        classes = set(classes)
        num_classes = len(classes)

        if self.__ppdb is not None:
            data = data + self.__ppdb.balance(data)

        self.__backup_classifier.train(data)

        if not self.__generate_patterns:
            return

        pattern_extractor = PatternExtractor(self.__nlp, self.__max_iterations)

        self.__nlp.set_mode("train")
        self.__nlp.set_split_sentences(self.__split_sentences)

        if(counter[most_common] / len(data) > self.__imbalance_threshold):
            classes.remove(most_common)

        if num_classes > 2:
            for class_value in classes:
                binary_data = []
                trees = []

                for instance in data:
                    if instance.class_value == class_value:
                        trees += self.__nlp.parse(instance.text)

                    binary_class = ("" if instance.class_value == class_value
                                    else "Not") + str(class_value)

                    binary_data.append(Instance(instance.text, binary_class))

                threshold = self.__ig_threshold[class_value] if \
                    isinstance(self.__ig_threshold, dict) and class_value in \
                    self.__ig_threshold else self.__ig_threshold
                ig_words = self.__top_information_gain_words(binary_data,
                                                             threshold)

                for tree in trees:
                    pattern_extractor.extract_patterns(ig_words, tree,
                                                       class_value)
        else:
            threshold = self.__ig_threshold[self.__ig_threshold.keys()[0]] if \
                isinstance(self.__ig_threshold, dict) else self.__ig_threshold
            ig_words = self.__top_information_gain_words(data, threshold)

            for class_value in classes:
                trees = []

                for instance in data:
                    if instance.class_value == class_value:
                        trees += self.__nlp.parse(instance.text)

                for tree in trees:
                    pattern_extractor.extract_patterns(ig_words, tree,
                                                       class_value)

        self.__nlp.set_mode("evaluate")

        for instance in data:
            self.__nlp.test(instance.text, class_value)

        self.__nlp.set_mode("classify")

    def train_with_phrases(self, data, phrase_data):
        temp = self.__generate_patterns

        self.train(phrase_data)
        self.set_generate_patterns(False)
        self.train(data)
        self.set_generate_patterns(temp)

training_file = "./Datasets/ShortWaitTime and LongWaitTime Training.arff"
phrases = "./Datasets/ShortWaitTime and LongWaitTime Phrases.csv"
test_file = "./Datasets/ShortWaitTime and LongWaitTime Test.arff"
textDatasetFileParser = TextDatasetFileParser()
training_set = textDatasetFileParser.parse(training_file)
phrases = textDatasetFileParser.parse(phrases)
test_set = textDatasetFileParser.parse(test_file)
paraphrase_arguments = {"host": "localhost", "database": "PPDB",
                        "user": "rriva002", "password": "passwd"}
rf = RandomForestTextClassifier(num_jobs=-1, random_state=10000)
classifier = SemgrexClassifier(backup_classifier=rf, generate_patterns=True,
                               imbalance_threshold=0.75, ig_threshold=0.0025,
                               paraphrase_arguments=None,
                               split_sentences=False, use_stemming=False)
phrase_training_set = []

for instance in phrases:
    for training_instance in training_set:
        if training_instance.text.find(instance.text) >= 0:
            phrase_training_set.append(instance)
            break

# classifier.train_with_phrases(training_set, phrase_training_set)
classifier.train(training_set)
classifier.evaluate(test_set, verbose=True)
