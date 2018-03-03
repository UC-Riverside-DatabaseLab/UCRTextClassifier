import collections
import csv
from abc import ABC, abstractmethod
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from httplib2 import Http, ProxyInfo, socks
from mysql import connector
from nltk.corpus import stopwords
from nltk.data import load
from nltk.tokenize import RegexpTokenizer, word_tokenize
from numpy import dot
from RegexClassifier import RegexClassifier
from utils import top_information_gain_words


class TextDatasetBalancer(ABC):
    @abstractmethod
    def balance(self, data):
        pass


class GoogleDatasetBalancer(TextDatasetBalancer):
    def __init__(self, query_type="sentence", snippet_parsing="sentence",
                 similarity="jaccard", doc2vec=None, threshold=1/3,
                 quotes=False, site=None, keys=None, proxies=None,
                 engine_id=None, pages_per_query=1, outfile=None,
                 use_phrase_file=False, verbose=False):
        self.__query_type = query_type
        self.__snippet_parsing = snippet_parsing
        self.__similarity = similarity
        self.__doc2vec = doc2vec
        self.__threshold = threshold
        self.__quotes = quotes
        self.__site = site
        self.__keys = keys
        self.proxies = proxies
        self.__key_index = 0 if keys and len(keys) > 0 else -1
        self.__engine_id = engine_id
        self.__pages_per_query = pages_per_query
        self.__outfile = outfile
        self.__verbose = verbose
        self.__sentence_tokenizer = None
        self.__stopwords = None

    def balance(self, data, labels):
        counter = collections.Counter(labels)
        most_common = counter.most_common(1)[0][0]
        search = True
        X = []
        y = []

        if self.__query_type == "sentence":
            queries, query_labels = data, labels
        elif self.__query_type == "regex":
            queries, query_labels = self.__regex_queries(data, counter,
                                                         most_common)
        elif self.__query_type == "phrase":
            queries, query_labels = data, labels
            most_common = "__ignore__"
            counter[most_common] = float("inf")
        else:
            self.__print("Query type not recognized.")
            return

        for class_value in set(labels):
            if counter[class_value] >= counter[most_common]:
                continue

            for i in range(len(queries)):
                if query_labels[i] == class_value:
                    for j in range(self.__pages_per_query):
                        sentences = self.__search_for_sentences(queries[i], j)

                        if sentences is None:
                            search = False
                            break
                        elif len(sentences) == 0:
                            break

                        for sentence in sentences:
                            counter[class_value] += 1

                            X.append(sentence)
                            y.append(class_value)

                            if counter[class_value] >= counter[most_common]:
                                break

                if counter[class_value] >= counter[most_common] or not search:
                    break

            if not search:
                break

        if self.__outfile and len(X) > 0:
            self.__write_dataset_file(data + X, labels + y)

        self.__print("Added " + str(len(X)) + " new instances.")
        return X, y

    def __append(self, similar_sentences, sentence):
        self.__print("    " + sentence)
        similar_sentences.append(sentence)

    def __google_search(self, sentence, **kwargs):
        http = None

        if self.__key_index < 0:
            return None
        elif self.__key_index > 0:
            proxy = self.proxies[self.__key_index - 1]
            http = Http(proxy_info=ProxyInfo(socks.PROXY_TYPE_HTTP,
                                             proxy_host=proxy[0],
                                             proxy_port=proxy[1]))

        g = build(serviceName="customsearch", version="v1", http=http,
                  developerKey=self.__keys[self.__key_index])
        q = ('"' + sentence + '"' if self.__quotes else sentence)
        q += (" site:" + self.__site if self.__site and len(self.__site) > 0
              else "")

        try:
            response = g.cse().list(q=q, cx=self.__engine_id,
                                    **kwargs).execute()
        except HttpError:
            self.__key_index += 1

            if self.__key_index >= len(self.__keys):
                self.__key_index = -1

            response = self.__google_search(sentence, **kwargs)

        return response

    def __parse_snippet(self, sentence, snippet, similar_sentences):
        snippet = snippet.replace("\n", "")

        if self.__snippet_parsing == "sentence":
            self.__parse_sentences(sentence, snippet, similar_sentences)
        elif self.__snippet_parsing == "ellipsis":
            self.__parse_ellipses(sentence, snippet, similar_sentences)
        elif self.__snippet_parsing == "snippet" and \
                self.__similar(sentence, self.__tokenize(snippet)):
            self.__append(similar_sentences, snippet)

    def __parse_ellipses(self, sentence, snippet, similar_sentences):
        for result_sentence in snippet.split(" ... "):
            result_sentence = self.__remove_ellipses(result_sentence)

            if self.__similar(sentence, self.__tokenize(result_sentence)):
                self.__append(similar_sentences, result_sentence)

    def __parse_sentences(self, sentence, snippet, similar_sentences):
        if not self.__sentence_tokenizer:
            self.__sentence_tokenizer = load('tokenizers/punkt/english.pickle')

        for result_sentence in self.__sentence_tokenizer.tokenize(snippet):
            result_sentence = self.__remove_ellipses(result_sentence)

            if self.__similar(sentence, self.__tokenize(result_sentence)):
                self.__append(similar_sentences, result_sentence)

    def __print(self, s):
        if self.__verbose:
            print(s)

    def __regex_queries(self, data, labels, counter, most_common):
        regex_classifier = RegexClassifier(jump_length=0)
        sample_weights = [counter[most_common] / counter[lb] for lb in labels]

        regex_classifier.fit(data, labels, sample_weights)

        regex_rules = regex_classifier.regex_rules
        query_dictionary = {}
        X = []
        y = []

        for regex_rule in regex_rules[0:len(regex_rules) - 1]:
            if regex_rule.class_value not in query_dictionary.keys():
                query_dictionary[regex_rule.class_value] = set()

            query_dictionary[regex_rule.class_value].add(regex_rule.phrase)

        for class_value in query_dictionary.keys():
            for query in query_dictionary[class_value]:
                X.append(query)
                y.append(class_value)

        return X, y

    def __remove_ellipses(self, sentence):
        leading_ellipse = "... "
        trailing_ellipse = " ..."

        if sentence.startswith(leading_ellipse):
            sentence = sentence.replace(leading_ellipse, "", 1)

        if sentence.endswith(trailing_ellipse):
            sentence = sentence[0:sentence.rfind(trailing_ellipse)]

        return sentence.strip()

    def __search_for_sentences(self, sentence, page=0):
        if self.__key_index < 0 or not self.__engine_id or len(sentence) == 0 \
                or (self.__similarity == "cosine" and not self.__doc2vec):
            self.__print("No valid keys.")
            return None

        self.__print(sentence)

        similar_sentences = []
        num = 10
        start = page * num + 1
        response = self.__google_search(sentence, num=num, start=start)

        if not response:
            self.__print("Query limit reached.")
            return None
        elif "items" not in response.keys():
            return similar_sentences

        sentence = self.__tokenize(sentence)

        if self.__similarity == "cosine" and self.__doc2vec:
            sentence = self.__doc2vec.infer_vector(sentence)

        for search_result in response["items"]:
            self.__parse_snippet(sentence, search_result["snippet"],
                                 similar_sentences)

        self.__print("")
        return similar_sentences

    def __similar(self, a, b):
        if self.__similarity == "jaccard":
            if not self.__stopwords:
                self.__stopwords = set(stopwords.words("english"))

            a = set(a) - self.__stopwords
            b = set(b) - self.__stopwords
            intersection_count = float(len(a.intersection(b)))
            union_count = float(len(a.union(b)))

            return intersection_count / union_count >= self.__threshold if \
                union_count > 0.0 else 0.0
        elif self.__similarity == "cosine":
            return dot(a, self.__doc2vec.infer_vector(b)) >= self.__threshold

        return False

    def __tokenize(self, sentence):
        return RegexpTokenizer(r"\w+").tokenize(sentence.lower())

    def __write_dataset_file(self, data, labels):
        with open(self.__outfile, 'w', newline="", errors="ignore") as file:
            writer = csv.writer(file, lineterminator="\n")

            for i in range(len(data)):
                writer.writerow([data[i], labels[i]])


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

    def balance(self, data, labels):
        if not self.__connection.is_connected():
            print("Not connected to database.")
            return

        if self.__ig_threshold > 0:
            top_words = top_information_gain_words(data, labels,
                                                   min_ig=self.__ig_threshold)
        else:
            top_words = None

        counter = collections.Counter(labels)
        most_common = counter.most_common(1)[0][0]
        X = []
        y = []

        for i in range(len(data)):
            if labels[i] == most_common:
                continue

            if self.__verbose:
                print(data[i])

            words = word_tokenize(data[i])
            num_words = len(words)
            paraphrases = {}

            for j in range(num_words):
                min_cutoff = min(num_words, j + self.__input_range[0])
                max_cutoff = min(num_words, j + self.__input_range[1])
                phrase = ""
                length = 0

                for k in range(j, min_cutoff):
                    phrase += (" " if length > 0 else "") + words[k]
                    length += 1

                if length < self.__input_range[0]:
                    break

                self.__find_paraphrases(phrase, paraphrases, top_words)

                for k in range(j + length, max_cutoff):
                    phrase += (" " if length > 0 else "") + words[k]
                    length += 1

                    self.__find_paraphrases(phrase, paraphrases, top_words)

            for phrase in paraphrases.keys():
                index = data[i].find(phrase)
                text = data[i].replace(phrase, "")

                for paraphrase in paraphrases[phrase].keys():
                    paraphrased = text[:index] + paraphrase + text[index:]

                    if self.__verbose:
                        print("    " + paraphrased)

                    X.append(paraphrased)
                    y.append(labels[i])

        if self.__verbose:
            print("Added " + str(len(X)) + " new instances.")

        return X, y

    def close_connection(self):
        self.__connection.close()

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
