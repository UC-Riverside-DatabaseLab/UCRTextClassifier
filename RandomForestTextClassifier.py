import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from AbstractTextClassifier import AbstractTextClassifier


class RandomForestTextClassifier(AbstractTextClassifier):
    """Random forest classifier that converts text to the bag-of-words model
    before training/classification. For further information, including
    constructor arguments, see scikit-learn's random forest documentation.
    """
    def __init__(self, num_trees=10, criterion="gini", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0, max_features="auto",
                 max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True,
                 oob_score=False, num_jobs=1, random_state=None, verbose=0,
                 warm_start=False, class_weight=None):
        self.vectorizer = TfidfVectorizer()
        self.random_forest = RandomForestClassifier(num_trees, criterion,
                                                    max_depth,
                                                    min_samples_split,
                                                    min_samples_leaf,
                                                    min_weight_fraction_leaf,
                                                    max_features,
                                                    max_leaf_nodes,
                                                    min_impurity_split,
                                                    bootstrap, oob_score,
                                                    num_jobs, random_state,
                                                    verbose, warm_start,
                                                    class_weight)

    def train(self, data):
        training_data = []
        training_labels = []
        training_weights = []

        for instance in data:
            training_data.append(instance.text)
            training_labels.append(instance.class_value)
            training_weights.append(instance.weight)

        self.random_forest.fit(self.vectorizer.fit_transform(training_data),
                               np.array(training_labels),
                               np.array(training_weights))

    def classify(self, instance):
        distribution = {}
        test_data = self.vectorizer.transform([instance.text])
        ordered_distribution = self.random_forest.predict_proba(test_data)

        for i in range(0, len(ordered_distribution[0])):
            if ordered_distribution[0, i] > 0:
                class_value = self.random_forest.classes_[i]
                distribution[class_value] = ordered_distribution[0, i]

        return distribution
