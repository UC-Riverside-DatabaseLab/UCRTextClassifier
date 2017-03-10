################################################################
################################################################
############# Author: Moloud Shahbazi
################################################################
################################################################

import tensorflow as tf
import numpy as np

# Add wordEmbeddings to initialize the embedding tensor, W
# Add Relu before fully connected
# Add another Relu before fully connected
# Add control over model type


class TextCNN(object):
    """
    A CNN for text classification. Uses an embedding layer, followed by a
    convolutional, max-pooling and softmax layer.
    """
    def __init__(self, sequence_length, num_classes, vocab_size,
                 embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0,
                 word2vec=None, relu1=False, relu2=False):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length],
                                      name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes],
                                      name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32,
                                                name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            if word2vec is not None:
                values = []
                for value in word2vec.values():
                    values.append(value)
                W = tf.Variable(tf.pack(np.array([len(values[0]) * [0]] +
                                                 values, dtype=np.float32)),
                                name="W")
                embedding_size = len(values[0])
                vocab_size = len(word2vec.keys()) + 1
            else:
                W = tf.Variable(tf.random_uniform([vocab_size, embedding_size],
                                                  -1.0, 1.0),
                                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars,
                                                          -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1),
                                name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]),
                                name="b")
                conv = tf.nn.conv2d(self.embedded_chars_expanded, W,
                                    strides=[1, 1, 1, 1], padding="VALID",
                                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(h, ksize=[1, sequence_length -
                                        filter_size + 1, 1, 1], name="pool",
                                        strides=[1, 1, 1, 1], padding='VALID')
                pooled_outputs.append(pooled)
        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat,
                                        self.dropout_keep_prob)

        if relu1:
            # Add a hidden layer
            n_hidden1 = 128
            with tf.name_scope("relu1"):
                wh1 = tf.Variable(tf.random_normal([num_filters_total,
                                                    n_hidden1]), name="Wh")
                bh1 = tf.Variable(tf.random_normal([n_hidden1]), name="Bh")
                relulayer_1 = tf.nn.relu(tf.nn.xw_plus_b(self.h_drop, wh1,
                                                         bh1))

            if relu2:
                # Add anothe hidden layer
                n_hidden2 = 128
                with tf.name_scope("relu2"):
                    wh2 = tf.Variable(tf.random_normal([n_hidden1, n_hidden2]),
                                      name="Wh")
                    bh2 = tf.Variable(tf.random_normal([n_hidden2]), name="Bh")
                    relulayer_2 = tf.nn.relu(tf.nn.xw_plus_b(relulayer_1, wh2,
                                                             bh2))
            else:
                relulayer_2 = relulayer_1
        else:
            relulayer_2 = self.h_drop

        print(relulayer_2.get_shape())
        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                # shape=[num_filters_total, num_classes],
                # shape=[n_hidden2, num_classes],
                shape=[relulayer_2.get_shape()[1], num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            # self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.scores = tf.nn.xw_plus_b(relulayer_2, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            self.distributions = tf.nn.softmax(self.scores)

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores,
                                                             self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions,
                                           tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions,
                                                   "float"), name="accuracy")
