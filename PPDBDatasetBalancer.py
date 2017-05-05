import collections
import math
from mysql import connector
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from TextDatasetFileParser import Instance


class PPDBDatasetBalancer(object):
    def __init__(self, host, database, user, password, input_range=(1, 1),
                 scoring_feature="PPDB2Score", threshold=0, ig_threshold=0,
                 verbose=False):
        self.__scoring_feature = scoring_feature
        self.__threshold = threshold
        self.__ig_threshold = ig_threshold
        self.__input_range = (max(1, input_range[0]), max(1, input_range[1]))
        self.__connection = connector.connect(host=host, database=database,
                                              user=user, password=password)
        self.__verbose = verbose

        if self.__input_range[0] > self.__input_range[1]:
            self.__input_range = self.__input_range[::-1]

    def balance(self, data):
        if not self.__connection.is_connected():
            print("Not connected to database.")
            return

        classes = []
        top_words = self.__top_information_gain_words(data) \
            if self.__ig_threshold > 0 else None

        for instance in data:
            classes.append(instance.class_value)

            instance.weight = 1

        counter = collections.Counter(classes)
        most_common = counter.most_common(1)[0][0]
        new_instances = []

        for instance in data:
            if instance.class_value == most_common:
                continue

            if self.__verbose:
                print(instance.text)

            class_value = instance.class_value
            words = word_tokenize(instance.text)
            num_words = len(words)
            paraphrases = {}

            for i in range(0, num_words):
                min_cutoff = min(num_words, i + self.__input_range[0])
                max_cutoff = min(num_words, i + self.__input_range[1])
                phrase = ""
                length = 0

                for j in range(i, min_cutoff):
                    phrase += (" " if length > 0 else "") + words[j]
                    length += 1

                if length < self.__input_range[0]:
                    break

                self.__find_paraphrases(phrase, paraphrases, top_words)

                for j in range(i + length, max_cutoff):
                    phrase += (" " if length > 0 else "") + words[j]
                    length += 1

                    self.__find_paraphrases(phrase, paraphrases, top_words)

            for phrase in paraphrases.keys():
                index = instance.text.find(phrase)
                text = instance.text.replace(phrase, "")

                for paraphrase in paraphrases[phrase].keys():
                    paraphrased = text[0:index] + paraphrase + text[index:]

                    if self.__verbose:
                        print("    " + paraphrased)

                    new_instances.append(Instance(paraphrased, class_value))

        if self.__verbose:
            print("Added " + str(len(new_instances)) + " new instances.")
        return new_instances

    def close_connection(self):
        self.__connection.close()

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

    def __find_paraphrases(self, phrase, paraphrases, top_words):
        if phrase in paraphrases.keys():
            return

        if top_words is not None:
            valid = False

            for word in top_words:
                if phrase.find(word) > -1:
                    valid = True
                    break

            if not valid:
                return

        sql = "SELECT paraphrase, " + self.__scoring_feature + " FROM ppdb " \
            + "WHERE phrase = '" \
            + phrase.replace("\\", "\\\\").replace("'", "\\'") + "' AND " \
            + self.__scoring_feature + " > " + str(self.__threshold)
        cursor = self.__connection.cursor()

        cursor.execute(sql)

        for row in cursor.fetchall():
            if phrase not in paraphrases:
                paraphrases[phrase] = {}

            paraphrases[phrase][row[0]] = row[1]

        cursor.close()

    def __top_information_gain_words(self, data):
        vectorizer = CountVectorizer(stop_words="english")
        text = []
        labels = []
        top_words = []
        word_frequencies = {}

        for instance in data:
            text.append(instance.text)
            labels.append([instance.class_value])

        vectors = vectorizer.fit_transform(text)
        words = vectorizer.get_feature_names()
        entropy = self.__entropy(labels, 0)

        for i in range(0, len(words)):
            word_frequencies[i] = {}

        for vector in vectors:
            for i in range(0, len(vector)):
                if vector[i] not in word_frequencies[i]:
                    word_frequencies[i][vector[i]] = 0

                word_frequencies[i][vector[i]] += 1

        for index, value_frequencies in word_frequencies.items():
            subset_entropy = 0

            for value, frequency in value_frequencies.items():
                value_probability = frequency / sum(value_frequencies.values())
                sub = [vector for vector in vectors if vector[index] == value]
                subset_entropy += value_probability * entropy(sub, index)

            if entropy - subset_entropy > self.__ig_threshold:
                top_words.append(words[index])

        return top_words
