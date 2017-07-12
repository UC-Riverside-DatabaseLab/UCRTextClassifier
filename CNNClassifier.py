import os
import sys
sys.path.insert(0, os.path.abspath("./cnn"))
import data_helpers as dh
import numpy as np
import tensorflow as tf
from AbstractTextClassifier import AbstractTextClassifier
from gensim.models.word2vec import Word2Vec, LineSentence
from pathlib import Path
from text_cnn import TextCNN
from tensorflow.contrib.learn import preprocessing


class CNNClassifier(AbstractTextClassifier):
    def __init__(self, embedding_dim=128, filter_sizes="2,3,4,5",
                 num_filters=128, dropout_keep_prob=0.5, l2_reg_lambda=0.0,
                 batch_size=64, num_epochs=20, evaluate_every=100,
                 checkpoint_every=100, allow_soft_placement=True,
                 log_device_placement=False, unlabeled_data=None):
        self.sess = None
        self.cnn = None
        # Parameters
        # ==================================================
        # Model Hyperparameters
        tf.flags._global_parser = tf.flags._argparse.ArgumentParser()
        tf.flags.DEFINE_integer("embedding_dim", embedding_dim,
                                "Dimensionality of character embedding "
                                "(default: 128)")
        tf.flags.DEFINE_string("filter_sizes", filter_sizes,
                               "Comma-separated filter sizes "
                               "(default: '3,4,5')")
        tf.flags.DEFINE_integer("num_filters", num_filters,
                                "Number of filters per filter size "
                                "(default: 128)")
        tf.flags.DEFINE_float("dropout_keep_prob", dropout_keep_prob,
                              "Dropout keep probability (default: 0.5)")
        tf.flags.DEFINE_float("l2_reg_lambda", l2_reg_lambda,
                              "L2 regularizaion lambda (default: 0.0)")

        # Training parameters
        tf.flags.DEFINE_integer("batch_size", batch_size,
                                "Batch Size (default: 64)")
        # tf.flags.DEFINE_integer("num_epochs", 20000,
        #                         "Number of training epochs (default: 200)")
        tf.flags.DEFINE_integer("num_epochs", num_epochs,
                                "Number of training epochs (default: 200)")
        tf.flags.DEFINE_integer("evaluate_every", evaluate_every,
                                "Evaluate model on dev set after this many "
                                "steps (default: 100)")
        tf.flags.DEFINE_integer("checkpoint_every", checkpoint_every,
                                "Save model after this many steps "
                                "(default: 100)")
        # Misc Parameters
        tf.flags.DEFINE_boolean("allow_soft_placement", allow_soft_placement,
                                "Allow device soft device placement")
        tf.flags.DEFINE_boolean("log_device_placement", log_device_placement,
                                "Log placement of ops on devices")
        self.class_names = []

        if unlabeled_data is None:
            self.w2v = None
        else:
            path = "./models/w2v_" + unlabeled_data[unlabeled_data.rfind("/") +
                                                    1] + ".model"

            if Path(path).is_file():
                word2vec = Word2Vec.load(path)
            else:
                word2vec = Word2Vec(LineSentence(unlabeled_data))

                word2vec.save(path)

            self.w2v = {}

            for word in word2vec.vocab:
                self.w2v[word] = word2vec[word]

    def train(self, data):
        FLAGS = tf.flags.FLAGS
        FLAGS._parse_flags()
        # Data Preparatopn
        # ==================================================
        # Load Data
        labels = set()
        for i in data:
            labels.add(i.class_value)
        self.class_names = list(labels)
        if self.w2v is None:
            x_t, y = dh.load_from_instances(data, self.class_names)
        else:
            x_t, y = dh.load_from_instances_withWord2vec(data,
                                                         self.class_names,
                                                         self.w2v)

        # Build vocabulary
        max_doc_len = max([len(x.split(" ")) for x in x_t])
        self.vocab_processor = preprocessing.VocabularyProcessor(max_doc_len)
        x = np.array(list(self.vocab_processor.fit_transform(x_t)))

        # Randomly shuffle data
        np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_shuffled = x[shuffle_indices]
        y_shuffled = y[shuffle_indices]
        x_train = x_shuffled
        y_train = y_shuffled

        # Training
        # ==================================================
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=FLAGS.allow_soft_placement,
                log_device_placement=FLAGS.log_device_placement)
            self.sess = tf.Session(config=session_conf)
            with self.sess.as_default():
                self.cnn = TextCNN(
                    sequence_length=x_train.shape[1],
                    num_classes=y_train.shape[1],
                    vocab_size=len(self.vocab_processor.vocabulary_),
                    embedding_size=FLAGS.embedding_dim,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=FLAGS.num_filters,
                    l2_reg_lambda=FLAGS.l2_reg_lambda,
                    word2vec=self.w2v)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(self.cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars,
                                                 global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    h = "{}/grad/hist".format(v.name.replace(":", "_"))
                    s1 = "{}/grad/sparsity".format(v.name.replace(":", "_"))
                    s2 = tf.nn.zero_fraction(g)
                    grad_hist_summary = tf.summary.histogram(h, g)
                    sparsity_summary = tf.summary.scalar(s1, s2)
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            tf.summary.merge(grad_summaries)

            # Initialize all variables
            self.sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  self.cnn.input_x: x_batch,
                  self.cnn.input_y: y_batch,
                  self.cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, loss, accuracy = self.sess.run(
                    [train_op, global_step, self.cnn.loss, self.cnn.accuracy],
                    feed_dict)
            # Generate batches
            batches = dh.batch_iter(list(zip(x_train, y_train)),
                                    FLAGS.batch_size,
                                    FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                tf.train.global_step(self.sess, global_step)

    def classify(self, instance):
        """
        Evaluates model on a dev set
        """
        if self.w2v is None:
            x_t, y = dh.load_from_instances([instance], self.class_names)
        else:
            x_t, y = dh.load_from_instances_withWord2vec([instance],
                                                         self.class_names,
                                                         self.w2v)
        x = np.array(list(self.vocab_processor.fit_transform(x_t)))
        # Build vocabulary
        feed_dict = {
          self.cnn.input_x: x,
          self.cnn.input_y: y,
          self.cnn.dropout_keep_prob: 1.0
        }
        scores, distributions = self.sess.run([self.cnn.scores,
                                               self.cnn.distributions],
                                              feed_dict)
        # print(scores,distributions)
        distribution = {}
        for i in range(len(distributions[0])):
            distribution[self.class_names[i]] = distributions[0][i]
        return distribution
