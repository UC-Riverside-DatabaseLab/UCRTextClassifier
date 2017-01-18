import collections
import sys
from googleapiclient.discovery import build
from nltk import word_tokenize
from nltk.data import load
from numpy import dot
from random import shuffle
from RandomForestTextClassifier import RandomForestTextClassifier
from RegexClassifier import RegexClassifier
from TextDatasetFileParser import Instance, TextDatasetFileParser


class GoogleDatasetBalancer(object):
    def __init__(self, use_regex=False, snippet_parsing="sentence",
                 similarity="jaccard", doc2vec=None, threshold=0.5,
                 quotes=False, site=None, keys=None, engine_id=None,
                 verbose=False):
        self.use_regex = use_regex
        self.snippet_parsing = snippet_parsing
        self.similarity = similarity
        self.doc2vec = doc2vec
        self.threshold = threshold
        self.quotes = quotes
        self.site = site
        self.keys = keys
        self.key_index = 0 if keys and len(keys) > 0 else -1
        self.queries_remaining = 87 if self.key_index >= 0 else 0
        self.engine_id = engine_id
        self.verbose = verbose
        self.sentence_tokenizer = load('tokenizers/punkt/english.pickle')

    def balance(self, data):
        classes = []
        new_instances = []

        for instance in data:
            classes.append(instance.class_value)

        counter = collections.Counter(classes)
        most_common = counter.most_common(1)[0][0]
        evaluated_classes = set()

        evaluated_classes.add(most_common)

        if self.use_regex:
            regex_classifier = RegexClassifier()

            regex_classifier.train(data)

            regex_rules = regex_classifier.regex_rules
            queries = regex_rules[0:len(regex_rules) - 1]
        else:
            queries = data

        for class_value in sorted(counter, key=counter.get, reverse=False):
            if class_value in evaluated_classes or \
                    counter[class_value] >= counter[most_common]:
                continue

            evaluated_classes.add(class_value)

            for query in queries:
                if query.class_value == class_value:
                    sentences = self.__search_for_sentences(query.text)

                    if not sentences:
                        return new_instances

                    for sentence in sentences:
                        new_instances.append(Instance(sentence, class_value))

                        counter[class_value] += 1

                        if counter[class_value] >= counter[most_common]:
                            break

                if counter[class_value] >= counter[most_common]:
                    break

        return new_instances

    def __append(self, similar_sentences, sentence):
        if self.verbose:
            print("    " + sentence)

        similar_sentences.append(sentence)

    def __google_search(self, sentence, **kwargs):
        g = build("customsearch", "v1", developerKey=self.keys[self.key_index])
        q = ('"' + sentence + '"' if self.quotes else sentence)
        q += (" site:" + self.site if self.site and len(self.site) > 0 else "")
        return g.cse().list(q=q, cx=self.engine_id, **kwargs).execute()

    def __search_for_sentences(self, sentence, page=0):
        if self.key_index < 0 or not self.engine_id or len(sentence) == 0 \
                or (self.similarity == "cosine" and not self.doc2vec):
            return []

        if self.verbose:
            print(sentence)

        similar_sentences = []
        num = 10
        start = page * num + 1
        response = self.__google_search(sentence, num=num, start=start)

        if not response or not response["items"]:
            return None

        sentence = word_tokenize(sentence)

        if self.similarity == "cosine" and self.doc2vec:
            sentence = self.doc2vec.infer_vector(sentence)

        for search_result in response["items"]:
            if self.snippet_parsing == "sentence":
                self.__parse_sentences(sentence, search_result["snippet"],
                                       similar_sentences)
            elif self.snippet_parsing == "ellipsis":
                self.__parse_ellipses(sentence, search_result["snippet"],
                                      similar_sentences)
            elif self.snippet_parsing == "snippet" and \
                    self.__similar(sentence, search_result["snippet"]):
                self.__append(similar_sentences, search_result["snippet"])

        self.queries_remaining -= 1

        if self.queries_remaining == 0:
            self.key_index += 1
            self.queries_remaining = 100

            if self.key_index >= len(self.keys):
                self.key_index = -1
                return similar_sentences

        if self.verbose:
            print("")

        return similar_sentences

    def __parse_ellipses(self, sentence, snippet, similar_sentences):
        for result_sentence in snippet.split(" ... "):
            while result_sentence.endswith(" ..."):
                result_sentence = result_sentence.replace(" ...", "")

                if self.__similar(sentence, word_tokenize(result_sentence)):
                    self.__append(similar_sentences, result_sentence)

    def __parse_sentences(self, sentence, snippet, similar_sentences):
        for result_sentence in self.sentence_tokenizer.tokenize(snippet):
            if self.__similar(sentence, word_tokenize(result_sentence)):
                self.__append(similar_sentences, result_sentence)

    def __similar(self, a, b):
        if self.similarity == "jaccard":
            a = set(a)
            b = set(b)
            i = a.intersection(b)
            u = a.union(b)
            return float(len(i)) / float(len(u)) >= self.threshold
        elif self.similarity == "cosine":
            return dot(a, self.doc2vec.infer_vector(b)) >= self.threshold

        return False


if len(sys.argv) < 2:
    print("Missing argument for path to dataset file.")
    sys.exit()

data = TextDatasetFileParser().parse(sys.argv[1])
keys = ["AIzaSyC0MlmDjTS_XLBJWAdIyGniDR3iMhkIT3k"]
engine_id = "010254973031167365908:s-lpifpdmgs"
balancer = GoogleDatasetBalancer(keys=keys, engine_id=engine_id, verbose=True)
text_classifier = RandomForestTextClassifier(num_jobs=-1)
training_set_end = int(len(data) * 0.9)

for instance in data:
    instance.weight = 1

data += balancer.balance(data)

shuffle(data)
text_classifier.train(data[0:training_set_end])
text_classifier.evaluate(data[training_set_end:], True)
