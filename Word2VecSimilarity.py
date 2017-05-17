import nltk.data
from AbstractTextClassifier import AbstractTextClassifier
from gensim.matutils import unitvec
from gensim.models.word2vec import Word2Vec, LineSentence
from pathlib import Path
from nltk import word_tokenize
from numpy import array, dot


class Word2VecSimilarity(AbstractTextClassifier):
    """Uses Word2Vec to determine the similarity of two text instances.

    Constructor arguments:
    unlableled_data - Path to a file containing unlabeled data
    """
    def __init__(self, unlabeled_data):
        self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.training_weights = {}
        self.training_data = {}
        path = "./models/w2v_" + unlabeled_data[unlabeled_data.rfind("/") +
                                                1] + ".model"

        if Path(path).is_file():
            self.word2vec = Word2Vec.load(path)
        else:
            self.word2vec = Word2Vec(LineSentence(unlabeled_data))

            self.word2vec.save(path)

    def train(self, data):
        self.training_data.clear()

        for instance in data:
            words = set()

            for word in word_tokenize(instance.text):
                if word in self.word2vec.vocab:
                    words.add(word)

            if instance.class_value not in self.training_data:
                self.training_data[instance.class_value] = []

            if len(words) > 0:
                v = [self.word2vec[word] for word in words]
                v = unitvec(array(v).mean(axis=0))

                self.training_data[instance.class_value].append(v)

    def classify(self, instance):
        words = set()
        distribution = {}

        for word in word_tokenize(instance.text):
            if word in self.word2vec.vocab:
                words.add(word)

        if len(words) > 0:
            v = unitvec(array([self.word2vec[w] for w in words]).mean(axis=0))

        for class_value in self.training_data.keys():
            if len(words) == 0:
                distribution[class_value] = 0
                continue

            best_score = 0

            for training_instance in self.training_data[class_value]:
                score = dot(v, training_instance)

                if score > best_score:
                    best_score = score

            distribution[class_value] = max(0, best_score)

        return self._normalize_distribution(distribution)
