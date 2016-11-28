import os
import math
import logging
import argparse
from timeit import default_timer 
import multiprocessing 
from collections import namedtuple

import numpy as np
from gensim import utils
from gensim.models import Doc2Vec
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
from sklearn.ensemble import RandomForestClassifier

from AbstractTextClassifier import AbstractTextClassifier


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter(
    '%(asctime)s-%(name)s-%(levelname)s - %(message)s'))
logger.addHandler(ch)

LineDocument = namedtuple("LineDocument", "words tags labeled label weight")


class DocVecClassifier(AbstractTextClassifier):
    """Classifier using Document Vector model

    Use Random Forest to go from document's vector to classes
    Attributes:
        saved_model: path to load the pre-trained model, default: None
        path_of_new_model: where to save the new model, default:None
        unlabeled_data: list of LineDocument objects that don't have label,
            support the vector training, default: None
        model_vec_size: size of document's vector in an individual model,
            default: 200
        num_epoch: number of epoch of training document's vectors
        infer_num_passes: number of passes for inferring vector of a new
            document, default: 1000
    """

    def __init__(self, saved_model=None, path_of_new_model="models/",
                 unlabeled_data=None, model_vec_size=200, num_epoch=20,
                 infer_num_passes=1000):

        self.model = self.load_model(saved_model) if saved_model \
                                                  else None
        self.unlabeled_docs = self.import_data(unlabeled_data, labeled=False) \
                if unlabeled_data else []
        self.model_vec_size = model_vec_size
        self.path_of_new_model = path_of_new_model
        self.num_epoch = num_epoch
        self.infer_num_passes = infer_num_passes

        # For training new doc2vec model if necessary
        cores = multiprocessing.cpu_count()
        self.name_to_models = {
                "dm_concat": Doc2Vec(dm=1, dm_concat=1,
                    size=model_vec_size, window=10,
                    sample=1e-4, hs=0, negative=5, min_count=2, workers=cores),
                "dbow": Doc2Vec(dm=0, size=model_vec_size, window=10,
                    sample=1e-4, hs=0, negative=5, min_count=2, workers=cores)
                }
        # For classifying class using trained document's vector
        self.random_forest = RandomForestClassifier()

    def train(self, data):

        self.labeled_docs = self.import_data(data)
        all_docs = self.labeled_docs + self.unlabeled_docs
            
        # model contains the vectors of all documents
        if not self.model:
            self.train_doc_vec(all_docs, self.path_of_new_model,
                    num_epoch=self.num_epoch)
            self.model = self.load_model(self.path_of_new_model)

        # Train the classifier using document vectors and their labels
        labeled_vecs = []
        labels = []
        weights = []
        for labeled_doc in self.labeled_docs:
            labeled_vecs.append(self.model.docvecs[labeled_doc.tags[0]])
            labels.append(labeled_doc.label)
            weights.append(labeled_doc.weight)
        logger.debug("# labeled_vecs={}, # labels={}, # weights={}" \
                .format(len(labeled_vecs), len(labels), len(weights)))

        self.random_forest.fit(labeled_vecs,
                               np.array(labels),
                               np.array(weights))

    def classify(self, instance):
        """Classify a text instance
        
        Returns:
            distribution: dict {class: possibility}
        """

        distribution = {}

        words = self.normalize_text(instance.text).split()
        test_vec = self.model.infer_vector(words, steps=self.infer_num_passes)
        ordered_distribution = self.random_forest.predict_proba(test_vec)

        for i in range(0, len(ordered_distribution[0])):
            if ordered_distribution[0, i] > 0:
                class_value = self.random_forest.classes_[i]
                distribution[class_value] = ordered_distribution[0, i]

        #logger.debug("classify \"{}\": {}".format(instance.text, distribution))
        return distribution

    def train_doc_vec(self, line_documents, model_dir, num_epoch=10):
        """Train the vectors of line_documents

        Args:
            line_documents: list of LineDocument objects that need vector
                representation.
            model_dir: where to save the documents' vectors 
            num_epoch: # epoch for training doc2vec models
        """ 
        # Setup doc2vec models 
        start_time = default_timer()
        cores = multiprocessing.cpu_count() 
        for model in self.name_to_models.values():
            model.build_vocab(line_documents)
        logger.debug("Time to setup model's dataset: {} s" \
                .format(default_timer() - start_time))
        
        # Train doc2vec models
        start_time = default_timer()
        previous_epoch_time = default_timer()
        for name, model in self.name_to_models.items():
            logger.debug("Training doc2vec using model {} . . .".format(name))
            for epoch in range(num_epoch):
                np.random.shuffle(line_documents)

                model.train(line_documents)
                current_epoch_time = default_timer()
                logger.debug("Model {} finished epoch {} in {} s".format(name,
                        epoch, (current_epoch_time - previous_epoch_time)))
                previous_epoch_time = current_epoch_time

            logger.debug("most similar to \"great\": {}".format(
                    model.most_similar(positive=["great"])))

            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            model.save(model_dir + name)

            logger.debug("{} time: {} s, saved model to {}".format(name,
                    default_timer() - start_time, model_dir + name))

    def load_model(self, model_dir):
        """Load models 
        """ 
        for name, model in self.name_to_models.items():
            model = Doc2Vec.load(model_dir + name) 
        loaded_model = ConcatenatedDoc2Vec(list(self.name_to_models.values()))
        return loaded_model


    def import_data(self, data, labeled=True):
        """Import dataset into LineDocument list

        Args:
            data: list of TextDatasetFileParser.Instance if labeled=True,
                otherwise, list of String
            labeled: is dataset labeled, default: True
        Returns:
            line_documents: list of LineDocument objects
                "tags" field of a LineDocument is "labeled" + doc_id
                or "unlabled" + doc_id, depending on labeled parameter
        """ 
        line_documents = []
        for doc_id, instance in enumerate(data):
            if labeled:
                words = self.normalize_text(instance.text).split()
                tags = [self.doc_tag(doc_id, labeled)]
                label = instance.class_value
                weight = instance.weight
                line_documents.append(LineDocument(words, tags,
                                                   labeled, label, weight))
            else:
                words = self.normalize_text(instance).split()
                tags = [self.doc_tag(doc_id, labeled)]
                line_documents.append(LineDocument(words, tags,
                                                   labeled, None, None))


        return line_documents

    def doc_tag(self, doc_id, labeled=True):
        """Create a tag for a document"""

        tag = "labeled" + str(doc_id) if labeled \
                                      else "unlabeled" + str(doc_id)
        return tag 

    def normalize_text(self, text):
        """Normalize raw input
    
        Convert text to lower-case and strip punctuation/symbols from words
        """
    
        #norm_text = unicode(text.lower(), "utf-8", "ignore")
        norm_text = text
    
        # Replace breaks with spaces
        norm_text = norm_text.replace('<br />', ' ')
    
        # Pad punctuation with spaces on both sides
        for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
            norm_text = norm_text.replace(char, ' ' + char + ' ')
    
        return utils.to_unicode(norm_text) 


def main(args): 

    from TextDatasetFileParser import TextDatasetFileParser
    data = TextDatasetFileParser().parse(args.training_data) 
    unlabeled_data = TextDatasetFileParser().parse_unlabeled(
            args.unlabeled_data) if args.unlabeled_data else None 

    docvec_classifier = DocVecClassifier(unlabeled_data=unlabeled_data)
    #docvec_classifier = DocVecClassifier()

    np.random.shuffle(data)
    training_set_end = int(len(data) * 0.9) 
    docvec_classifier.train(data[0:training_set_end])

    test_set = data[training_set_end:]
    docvec_classifier.evaluate(test_set, True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Document Vector based Text Classifier")
    parser.add_argument(
            "training_data",
            help="path of training data file") 
    parser.add_argument(
            "--output",
            help="output path")
    parser.add_argument(
            "--unlabeled-data",
            help="path of unlabeled data file") 
    args = parser.parse_args()
    logger.debug("args: {}".format(args))
    main(args)
