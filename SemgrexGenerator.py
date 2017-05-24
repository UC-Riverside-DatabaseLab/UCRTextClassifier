import json
import math
import subprocess
from AbstractTextClassifier import AbstractTextClassifier
from collections import Counter
from extractPatterns import PatternExtractor
from os.path import isfile
from sklearn.feature_extraction.text import CountVectorizer
from TextDatasetFileParser import Instance


class SemgrexClassifier(AbstractTextClassifier):
    def __init__(self, ig_threshold=0, imbalance_threshold=0.75):
        java_directory = "SemgrexClassifierHelper/"
        helper_filename = "SemgrexClassifierHelper"

        if not isfile(java_directory + "bin/" + helper_filename + ".class"):
            classpath = "./" + java_directory
            libraries = ["json-simple-1.1.1.jar",
                         "stanford-corelnlp-3.7.0.jar",
                         "stanford-corelnlp-3.7.0-models.jar"]

            for library in libraries:
                classpath += ";./" + java_directory + "/" + library

            subprocess.run(["javac", "-cp", classpath,
                            "src/" + helper_filename + ".java"])

        self.__helper = subprocess.Popen(["java", java_directory + "bin/" +
                                          helper_filename],
                                         stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE,
                                         universal_newlines=True)
        self.__ig_threshold = ig_threshold
        self.__imbalance_threshold = imbalance_threshold

    def classify(self, instance):
        payload = {"mode": "classify", "text": instance.text}
        (distribution, _) = self.__helper.communicate(input=payload)
        return json.loads(distribution)

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
        trees = []

        self.__helper.communicate(input=json.dumps({"mode": "init"}))

        for instance in data:
            payload = {"mode": "parse", "text": instance.text}
            (l, _) = self.__helper.communicate(input=json.dumps(payload))

            for tree in json.loads(l):
                trees.append(tree)

            classes.append(instance.class_value)

        counter = Counter(classes)
        most_common = counter.most_common(1)[0][0]
        classes = set(classes)

        if(counter[most_common] / len(data) > self.__imbalance_threshold):
            classes.remove(most_common)

        for class_value in classes:
            binary_data = []

            for instance in data:
                binary_class = ("" if instance.class_value == class_value else
                                "Not") + str(class_value)

                binary_data.append(Instance(instance.text, binary_class))

                ig_words = self.__top_information_gain_words(binary_data)

                for pattern in pattern_extractor.extract_patterns(ig_words,
                                                                  trees):
                    payload = {"mode": "add_pattern", "pattern": pattern,
                               "class": str(class_value)}

                    self.__helper.communicate(input=payload)

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
