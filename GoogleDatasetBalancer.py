import collections
import csv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from httplib2 import Http, ProxyInfo, socks
from nltk.corpus import stopwords
from nltk.data import load
from nltk.tokenize import RegexpTokenizer
from numpy import dot
from RegexClassifier import RegexClassifier
from TextDatasetFileParser import Instance


class GoogleDatasetBalancer(object):
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

    def balance(self, data):
        classes = []
        new_instances = []

        for instance in data:
            classes.append(instance.class_value)

            instance.weight = 1

        counter = collections.Counter(classes)
        most_common = counter.most_common(1)[0][0]
        search = True

        if self.__query_type == "sentence":
            queries = data
        elif self.__query_type == "regex":
            queries = self.__regex_queries(data, counter, most_common)
        elif self.__query_type == "phrase":
            queries = data
            most_common = "__ignore__"
            counter[most_common] = float("inf")
        else:
            self.__print("Query type not recognized.")
            return

        for class_value in set(classes):
            if counter[class_value] >= counter[most_common]:
                continue

            for query in queries:
                if query.class_value == class_value:
                    for i in range(0, self.__pages_per_query):
                        sentences = self.__search_for_sentences(query.text, i)

                        if sentences is None:
                            search = False
                            break
                        elif len(sentences) == 0:
                            break

                        for sentence in sentences:
                            counter[class_value] += 1
                            instance = Instance(sentence, class_value)

                            new_instances.append(instance)

                            if counter[class_value] >= counter[most_common]:
                                break

                if counter[class_value] >= counter[most_common] or not search:
                    break

            if not search:
                break

        if self.__outfile and len(new_instances) > 0:
            self.__write_dataset_file(data + new_instances)

        self.__print("Added " + str(len(new_instances)) + " new instances.")
        return new_instances

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

    def __regex_queries(self, data, counter, most_common):
        regex_classifier = RegexClassifier(jump_length=0)
        training_data = []

        for instance in data:
            text = instance.text
            weight = counter[most_common] / counter[instance.class_value]

            training_data.append(Instance(text, instance.class_value, weight))

        regex_classifier.train(training_data)

        regex_rules = regex_classifier.regex_rules
        query_dictionary = {}
        queries = []

        for regex_rule in regex_rules[0:len(regex_rules) - 1]:
            if regex_rule.class_value not in query_dictionary.keys():
                query_dictionary[regex_rule.class_value] = set()

            query_dictionary[regex_rule.class_value].add(regex_rule.phrase)

        for class_value in query_dictionary.keys():
            for query in query_dictionary[class_value]:
                queries.append(Instance(query, class_value))

        return queries

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

    def __write_dataset_file(self, data):
        with open(self.__outfile, 'w', newline="", errors="ignore") as file:
            writer = csv.writer(file)

            for instance in data:
                writer.writerow([instance.text, instance.class_value])
