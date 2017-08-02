from numpy import array
from nltk.stem.snowball import EnglishStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from AbstractTextClassifier import AbstractTextClassifier


class SVMTextClassifier(AbstractTextClassifier):
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0,
                 shrinking=True, tol=0.001, cache_size=200, class_weight=None,
                 verbose=False, max_iter=-1, decision_function_shape=None,
                 random_state=None, ngram_range=(1, 3), min_df=0.03,
                 max_word_features=1000, tf_idf=True):
        stemmer = EnglishStemmer()

        if tf_idf:
            analyzer = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df,
                                       max_features=max_word_features,
                                       stop_words="english").build_analyzer()
        else:
            analyzer = CountVectorizer(ngram_range=ngram_range, min_df=min_df,
                                       max_features=max_word_features,
                                       stop_words="english").build_analyzer()

        def stemmed_words(text):
            return (stemmer.stem(word) for word in analyzer(text))

        if tf_idf:
            self.__vectorizer = TfidfVectorizer(ngram_range=ngram_range,
                                                min_df=min_df,
                                                max_features=max_word_features,
                                                stop_words="english",
                                                analyzer=stemmed_words)
        else:
            self.__vectorizer = CountVectorizer(ngram_range=ngram_range,
                                                min_df=min_df,
                                                max_features=max_word_features,
                                                stop_words="english",
                                                analyzer=stemmed_words)

        self.__svc = SVC(C, kernel, degree, gamma, coef0, shrinking,
                         True, tol, cache_size, class_weight, verbose,
                         max_iter, decision_function_shape, random_state)

    def train(self, data):
        training_data = []
        training_labels = []
        training_weights = []

        for instance in data:
            training_data.append(instance.text)
            training_labels.append(instance.class_value)
            training_weights.append(instance.weight)

        self.__svc.fit(self.__vectorizer.fit_transform(training_data),
                       array(training_labels), array(training_weights))

    def classify(self, instance):
        distribution = {}
        test_data = self.__vectorizer.transform([instance.text])
        ordered_distribution = self.__svc.predict_proba(test_data)

        for i in range(0, len(ordered_distribution[0])):
            if ordered_distribution[0, i] > 0:
                class_value = self.__svc.classes_[i]
                distribution[class_value] = ordered_distribution[0, i]

        return distribution
