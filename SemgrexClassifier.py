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

    def classify(self, text, class_value=None):
        self.__send("classify", {"text": text, "class": class_value})
        return self.__receive()

    def end(self):
        self.__send("end", {})

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


class DefaultRuleClassifier(AbstractTextClassifier):
    def __init__(self):
        self.__default_distribution = {}

    def classify(self, data):
        return self.__default_distribution

    def train(self, data):
        classes = []

        for instance in data:
            classes.append(instance.class_value)

        most_common = Counter(classes).most_common(1)[0][0]

        for class_label in set(classes):
            self.__default_distribution[class_label] = 1 if \
                class_label == most_common else 0


class SemgrexClassifier(AbstractTextClassifier):
    def __init__(self, backup_classifier=None, generate_patterns=True,
                 nlp_host="localhost", paraphrase_arguments=None,
                 imbalance_threshold=0.75, ig_threshold=0,
                 split_sentences=False, max_words=4, use_stemming=False):
        self.__backup_classifier = DefaultRuleClassifier() if \
            backup_classifier is None else backup_classifier
        self.__generate_patterns = generate_patterns
        self.__nlp = NlpService(host=nlp_host) if generate_patterns else None
        self.__imbalance_threshold = imbalance_threshold
        self.__ig_threshold = ig_threshold
        self.__split_sentences = split_sentences
        self.__max_words = max(1, max_words)
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
                if "ig_threshold" in paraphrase_arguments else 0
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
            distribution = self.__nlp.classify(self.__stem_text(instance.text),
                                               instance.class_value)
            distribution = self._normalize_distribution(distribution)

        return distribution if len(distribution) > 0 else \
            self.__backup_classifier.classify(instance)

    def disconnect(self):
        if self.__nlp is not None:
            self.__nlp.end()

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

        for class_value in classes:
            print(class_value + ": " + str(counter[class_value]))

        if self.__ppdb is not None:
            data = data + self.__ppdb.balance(data)

        self.__backup_classifier.train(data)

        if not self.__generate_patterns:
            return

        pattern_extractor = PatternExtractor(self.__nlp, self.__max_words)

        self.__nlp.set_mode("train")
        self.__nlp.set_split_sentences(self.__split_sentences)

        if(counter[most_common] / len(data) > self.__imbalance_threshold):
            classes.remove(most_common)

        if num_classes > 2:
            for class_value in classes:
                binary_data = []
                trees = []

                for instance in data:
                    text = self.__stem_text(instance.text)

                    if instance.class_value == class_value:
                        trees += self.__nlp.parse(text)

                        binary_data.append(Instance(text, class_value))
                    else:
                        binary_data.append(Instance(text, "Not" + class_value))

                threshold = self.__ig_threshold[class_value] if \
                    isinstance(self.__ig_threshold, dict) and class_value in \
                    self.__ig_threshold else self.__ig_threshold
                ig_words = self.__top_information_gain_words(binary_data,
                                                             threshold)

                for tree in trees:
                    pattern_extractor.extract_patterns(ig_words, tree,
                                                       class_value)
        else:
            least_common = counter.most_common(num_classes)[num_classes - 1][0]
            threshold = self.__ig_threshold[least_common] if \
                isinstance(self.__ig_threshold, dict) else self.__ig_threshold
            ig_words = self.__top_information_gain_words(data, threshold)

            for class_value in classes:
                trees = []

                for instance in data:
                    text = self.__stem_text(instance.text)

                    if instance.class_value == class_value:
                        trees += self.__nlp.parse(text)

                for tree in trees:
                    pattern_extractor.extract_patterns(ig_words, tree,
                                                       class_value)

        self.__nlp.set_mode("evaluate")

        for instance in data:
            self.__nlp.test(instance.text, class_value)

        self.__nlp.set_mode("classify")


training_file = "./Datasets/ShortWaitTime and LongWaitTime Training.arff"
test_file = "./Datasets/ShortWaitTime and LongWaitTime Test.arff"
textDatasetFileParser = TextDatasetFileParser()
training_set = textDatasetFileParser.parse(training_file)
test_set = textDatasetFileParser.parse(test_file)
test_set += textDatasetFileParser.parse("./Datasets/NewShortWaitTime.arff")
paraphrase_arguments = {"host": "localhost", "database": "PPDB",
                        "user": "rriva002", "password": "passwd",
                        "ig_threshold": 0.08, "threshold": 3}
rf = RandomForestTextClassifier(num_jobs=-1, random_state=10000)
ig = {"EasyToMakeAppointment":     0.0007, "HardToMakeAppointment": 0.001,
      "GoodBedsideManner":         0.008,  "BadBedsideManner":      0.004,
      "GoodMedicalSkills":         0.001,  "BadMedicalSkills":      0.002,
      "GoodStaff":                 0.003,  "BadStaff":              0.004,
      "LongVisitTime":             0.0003, "ShortVisitTime":        0.0002,
      "LowCost":                   0.0004, "HighCost":              0.004,
      "PromoteInformationSharing": 0.005,  "NoInformationSharing":  0.0005,
      "ShortWaitTime":             0.003,  "LongWaitTime":          0.005}
classifier = SemgrexClassifier(backup_classifier=rf, generate_patterns=True,
                               imbalance_threshold=0.4, ig_threshold=ig,
                               paraphrase_arguments=None,
                               split_sentences=False, max_words=4,
                               use_stemming=False)

classifier.train(training_set)
classifier.evaluate(test_set, verbose=True)
classifier.disconnect()
