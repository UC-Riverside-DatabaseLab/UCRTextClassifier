from numpy import array, dot
from gensim.matutils import unitvec
from gensim.models.word2vec import LineSentence
from gensim.models.word2vec import Word2Vec
from AbstractTextClassifier import AbstractTextClassifier


class Word2VecSimilarity(AbstractTextClassifier):
    """Uses Word2Vec to determine the similarity of two text instances.

    Constructor arguments:
    unlableled_data - Path to a file containing unlabeled data
    """
    def __init__(self, unlabeled_data):
        self.unlabeled_data = unlabeled_data
        self.word2vec = Word2Vec()
        self.training_data = {}

    def train(self, data):
        self.word2vec.build_vocab(LineSentence(self.unlabeled_data))
        self.word2vec.train(LineSentence(self.unlabeled_data))
        self.training_data.clear()

        for instance in data:
            words = set()

            for word in instance.text.split(" "):
                if word in self.word2vec.vocab:
                    words.add(word)

            if instance.class_value not in self.training_data:
                self.training_data[instance.class_value] = []

            if len(words) > 0:
                # self.training_data[instance.class_value].append(words)
                v = [self.word2vec[word] for word in words]
                v = unitvec(array(v).mean(axis=0))

                self.training_data[instance.class_value].append(v)

    def classify(self, instance):
        words = set()
        distribution = {}

        for word in instance.text.split(" "):
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
#        text = set(instance.text.split(" "))
#        distribution = {}
#
#        for class_value in self.training_data.keys():
#            best_overall_score = 0
#
#            for training_instance in self.training_data[class_value]:
#                score_a = 0
#                score_b = 0
#
#                for word in text:
#                    if word not in self.word2vec.vocab:
#                        continue
#
#                    best_score = 0
#
#                    for training_word in training_instance:
#                        score = self.word2vec.similarity(word, training_word)
#
#                        if score > best_score:
#                            best_score = score
#
#                        if best_score >= 1:
#                            break
#
#                    score_a += best_score
#
#                for training_word in training_instance:
#                    best_score = 0
#
#                    for word in text:
#                        if word not in self.word2vec.vocab:
#                            continue
#
#                        score = self.word2vec.similarity(word, training_word)
#
#                        if score > best_score:
#                            best_score = score
#
#                        if best_score >= 1:
#                            break
#
#                    score_b += best_score
#
#                score_a /= float(len(text))
#                score_b /= float(len(training_instance))
#                overall_score = score_a + score_b
#
#                if overall_score > best_overall_score:
#                    best_overall_score = overall_score
#
#                if best_overall_score >= 2:
#                    break
#
#            distribution[class_value] = max(0, best_overall_score)

        return self._normalize_distribution(distribution)
