#####################################################
#
#
#   github - http://github.com/phraniiac
#   project - tensorflow boilerplate
#
#
#####################################################

import numpy as np
import tensorflow as tf

from utils.utils import *
from model import Config
from model import Nueral_Net_Model as nnm


class P2PE_DNNS(nnm):
    """ Point to Point Network Encryption technique
    using Deep Nueral Networks.
    """

#    def __init__(self, config):
#        self.c = config

    def add_placeholders(self):
        self.input_message_placeholder = tf.placeholder(dtype=tf.float32, \
					shape=[self.c.batch_size, self.c.plain_text_length])


    def add_training_vars(self):
        # peer 1 weights.
        ## peer 1 fc layer.
        with tf.name_scope('peer_1_fc_layer'):
            self.peer_1_fc_weight = self.lw.get_weights('peer_1_fc_weight', [self.c.plain_text_length] * 2, 'xavier')
            self.peer_1_fc_bias = self.lw.get_weights('peer_1_fc_bias', [self.c.plain_text_length],\
                                        initializer='constant', constant=0.1)
            self.add_to_summaries([self.peer_1_fc_weight, self.peer_1_fc_bias])
        ## peer 1 conv1 layer.
        with tf.name_scope('peer_1_conv1_layer'):
            self.peer_1_conv1_weight = self.lw.get_weights('peer_1_conv1_weight', [4, 1, 2], 'xavier')
            self.peer_1_conv1_bias = self.lw.get_weights('peer_1_conv1_bias', [self.c.plain_text_length, 2], 'constant', 0.1)
            self.add_to_summaries([self.peer_1_conv1_weight, self.peer_1_conv1_bias])
        ## peer 1 conv2 layer.
        with tf.name_scope('peer_1_conv2_layer'):
            self.peer_1_conv2_weight = self.lw.get_weights('peer_1_conv2_weight', [2, 2, 4], 'xavier')
            self.peer_1_conv2_bias = self.lw.get_weights('peer_1_conv2_bias', [self.c.plain_text_length/2, 4], 'constant', 0.1)
            self.add_to_summaries([self.peer_1_conv2_weight, self.peer_1_conv2_bias])
        ## peer 1 conv3 layer.
        with tf.name_scope('peer_1_conv3_layer'):
            self.peer_1_conv3_weight = self.lw.get_weights('peer_1_conv3_weight', [1, 4, 4], 'xavier')
            self.peer_1_conv3_bias = self.lw.get_weights('peer_1_conv3_bias', [self.c.plain_text_length/2, 4], 'constant', 0.1)
            self.add_to_summaries([self.peer_1_conv3_weight, self.peer_1_conv3_bias])
        ## peer 1 conv4 layer.
        with tf.name_scope('peer_1_conv4_layer'):
            self.peer_1_conv4_weight = self.lw.get_weights('peer_1_conv4_weight', [1, 4, 1], 'xavier')
            self.peer_1_conv4_bias = self.lw.get_weights('peer_1_conv4_bias', [self.c.plain_text_length/2, 1], 'constant', 0.1)
            self.add_to_summaries([self.peer_1_conv4_weight, self.peer_1_conv4_bias])
        ## peer 1 private key. FC Layer.
        with tf.name_scope('peer_1_private_key_layer'):
            self.peer_1_private_weight = self.lw.get_weights('peer_1_private_key_weight', [self.c.plain_text_length, self.c.plain_text_length], 'xavier')
            self.peer_1_private_bias = self.lw.get_weights('peer_1_private_key_bias', [self.c.plain_text_length], 'constant', 0.1)
            self.add_to_summaries([self.peer_1_private_weight, self.peer_1_private_bias])

        # peer 2 weights.
        with tf.name_scope('peer_2_fc_layer'):
            self.peer_2_fc_weight = self.lw.get_weights('peer_2_fc_weight',[self.c.plain_text_length,\
									 self.c.plain_text_length*2], 'xavier')
            self.peer_2_fc_bias = self.lw.get_weights('peer_2_fc_bias', [self.c.plain_text_length*2],\
									initializer='constant', constant=0.1)
            self.add_to_summaries([self.peer_2_fc_weight, self.peer_2_fc_bias])
        ## peer 2 conv1 layer.
        with tf.name_scope('peer_1_conv1_layer'):
            self.peer_2_conv1_weight = self.lw.get_weights('peer_2_conv1_weight', [4, 1, 2], 'xavier')
            self.peer_2_conv1_bias = self.lw.get_weights('peer_2_conv1_bias', [self.c.plain_text_length*2, 2], 'constant', 0.1)
            self.add_to_summaries([self.peer_2_conv1_weight, self.peer_2_conv1_bias])
        ## peer 2 conv2 layer.
        with tf.name_scope('peer_2_conv2_layer'):
            self.peer_2_conv2_weight = self.lw.get_weights('peer_2_conv2_weight', [2, 2, 4], 'xavier')
            self.peer_2_conv2_bias = self.lw.get_weights('peer_2_conv2_bias', [self.c.plain_text_length, 4], 'constant', 0.1)
            self.add_to_summaries([self.peer_2_conv2_weight, self.peer_2_conv2_bias])
        ## peer 2 conv3 layer.
        with tf.name_scope('peer_2_conv3_layer'):
            self.peer_2_conv3_weight = self.lw.get_weights('peer_2_conv3_weight', [1, 4, 4], 'xavier')
            self.peer_2_conv3_bias = self.lw.get_weights('peer_2_conv3_bias', [self.c.plain_text_length, 4], 'constant', 0.1)
            self.add_to_summaries([self.peer_2_conv3_weight, self.peer_2_conv3_bias])
        ## peer 2 conv4 layer.
        with tf.name_scope('peer_2_conv4_layer'):
            self.peer_2_conv4_weight = self.lw.get_weights('peer_2_conv4_weight', [1, 4, 1], 'xavier')
            self.peer_2_conv4_bias = self.lw.get_weights('peer_2_conv4_bias', [self.c.plain_text_length, 1], 'constant', 0.1)
            self.add_to_summaries([self.peer_2_conv4_weight, self.peer_2_conv4_bias])

        # peer 2 private key. FC Layer.
        with tf.name_scope('peer_2_private_key_layer'):
            self.peer_2_private_weight = self.lw.get_weights('peer_2_private_key_weight', [self.c.plain_text_length,\
                                                            self.c.plain_text_length], 'xavier')
            self.peer_2_private_bias = self.lw.get_weights('peer_2_private_key_bias', [self.c.plain_text_length])
            self.add_to_summaries([self.peer_2_private_weight, self.peer_2_private_bias])


    def add_training_op(self):
        # take input to first 
        with tf.name_scope('input_training'):
            with tf.name_scope('peer_1'):
                ## fc layer for key part 1.
                with tf.name_scope('peer_1_key_op'):
                    self.peer_1_fc_key_out = self.lw.FC_layer(self.input_message_placeholder, self.peer_1_private_weight,\
                                                        self.peer_1_private_bias)
                    self.peer_1_fc_key_out = tf.nn.sigmoid(self.peer_1_fc_key_out)
                    self.add_to_summaries([self.peer_1_fc_key_out])
                with tf.name_scope('peer_1_fc_layer'):
                    self.peer_1_fc_out = self.lw.FC_layer(self.peer_1_fc_key_out,\
                                                self.peer_1_fc_weight, self.peer_1_fc_bias)
                    self.peer_1_fc_out = tf.nn.sigmoid(self.peer_1_fc_out)
                    self.add_to_summaries([self.peer_1_fc_out])
                    self.peer_1_fc_out = tf.expand_dims(self.peer_1_fc_out, axis=2)
                with tf.name_scope('peer_1_conv1_layer'):
                    self.peer_1_conv1_out = self.lw.conv1d_layer(self.peer_1_fc_out, self.peer_1_conv1_weight, \
                                                            self.peer_1_conv1_bias, strides=1)
                    self.peer_1_conv1_out = tf.nn.sigmoid(self.peer_1_conv1_out)
                    self.add_to_summaries([self.peer_1_conv1_out])
                with tf.name_scope('peer_1_conv2_layer'):
                    self.peer_1_conv2_out = self.lw.conv1d_layer(self.peer_1_conv1_out, self.peer_1_conv2_weight,\
                                                self.peer_1_conv2_bias, strides=2)
                    self.peer_1_conv2_out = tf.nn.sigmoid(self.peer_1_conv2_out)
                    self.add_to_summaries([self.peer_1_conv2_out])
                with tf.name_scope('peer_1_conv3_layer'):
                    self.peer_1_conv3_out = self.lw.conv1d_layer(self.peer_1_conv2_out, self.peer_1_conv3_weight,\
                                                self.peer_1_conv3_bias, strides=1)
                    self.peer_1_conv3_out = tf.nn.sigmoid(self.peer_1_conv3_out)
                    self.add_to_summaries([self.peer_1_conv3_out])
                with tf.name_scope('peer_1_conv4_layer'):
                    self.peer_1_conv4_out = self.lw.conv1d_layer(self.peer_1_conv3_out, self.peer_1_conv4_weight,\
                                                self.peer_1_conv4_bias, strides=1)
                    self.peer_1_conv4_out = tf.nn.tanh(self.peer_1_conv4_out)
                    self.peer_1_conv4_out = tf.squeeze(self.peer_1_conv4_out)
                    self.add_to_summaries([self.peer_1_conv4_out])
            #### ---------- peer 1 output = self.peer_1_out.

            with tf.name_scope('peer_2'):
                ## append the last 8 bits of message(which is also a key).
                with tf.name_scope('key_preprocess'):
                    split0, split1 = tf.split(self.input_message_placeholder, [8, 8], 1)
                    self.peer_2_input = tf.concat([self.peer_1_conv4_out, split1], axis=1)
                ## peer 2 key op.
                with tf.name_scope('peer_2_key_op'):
                    self.peer_2_fc_key_out = self.lw.FC_layer(self.peer_2_input, self.peer_2_private_weight,\
                                                        self.peer_2_private_bias)
                    self.peer_2_out = tf.nn.sigmoid(self.peer_2_fc_key_out)
                    self.add_to_summaries([self.peer_2_out])
                with tf.name_scope('peer_2_fc_layer'):
                    self.peer_2_fc_out = self.lw.FC_layer(self.peer_2_out, self.peer_2_fc_weight, \
                                                        self.peer_2_fc_bias)
                    self.peer_2_fc_out = tf.nn.sigmoid(self.peer_2_fc_out)
                    self.add_to_summaries([self.peer_2_fc_out])
                    self.peer_2_fc_out = tf.expand_dims(self.peer_2_fc_out, axis=2)
                with tf.name_scope('peer_2_conv1_layer'):
                    self.peer_2_conv1_out = self.lw.conv1d_layer(self.peer_2_fc_out, self.peer_2_conv1_weight,\
                                                        self.peer_2_conv1_bias, strides=1)
                    self.peer_2_conv1_out = tf.nn.sigmoid(self.peer_2_conv1_out)
                    self.add_to_summaries([self.peer_2_conv1_out])
                with tf.name_scope('peer_2_conv2_layer'):
                    self.peer_2_conv2_out = self.lw.conv1d_layer(self.peer_2_conv1_out, self.peer_2_conv2_weight,\
                                                        self.peer_2_conv2_bias, strides=2)
                    self.peer_2_conv2_out = tf.nn.sigmoid(self.peer_2_conv2_out)
                    self.add_to_summaries([self.peer_2_conv2_out])
                with tf.name_scope('peer_2_conv3_layer'):
                    self.peer_2_conv3_out = self.lw.conv1d_layer(self.peer_2_conv2_out, self.peer_2_conv3_weight,\
                                                        self.peer_2_conv3_bias, strides=1)
                    self.peer_2_conv3_out = tf.nn.sigmoid(self.peer_2_conv3_out)
                    self.add_to_summaries([self.peer_2_conv3_out])
                with tf.name_scope('peer_2_conv4_layer'):
                    self.peer_2_conv4_out = self.lw.conv1d_layer(self.peer_2_conv3_out, self.peer_2_conv4_weight,\
                                                        self.peer_2_conv4_bias, strides=1)
                    self.peer_2_conv4_out = tf.nn.sigmoid(self.peer_2_conv4_out)
                    self.peer_2_conv4_out = tf.squeeze(self.peer_2_conv4_out)
                    self.add_to_summaries([self.peer_2_conv4_out])
                ## final result peer 2 = self.peer_2_conv4_out.


    def add_loss_op(self):
        abss = tf.abs(tf.subtract(self.peer_2_conv4_out, self.input_message_placeholder))
        with tf.name_scope('loss_train_keys'):
            self.loss_train = tf.reduce_mean(abss)
            self.variable_summaries(self.loss_train)
        with tf.name_scope('bits_wrong'):
            self.bits_wrong_train = tf.reduce_mean(tf.reduce_sum(abss, axis=1))
            self.variable_summaries(self.bits_wrong_train)
        return


    def add_optimizers(self):
        self.training_vars = tf.trainable_variables()
        self.actual_trainers = [var for var in self.training_vars \
                                    if not 'key' in var.name]
        self.actual_trainers = [var for var in self.actual_trainers \
                                    if 'peer_2' in var.name]
        self.optimizer_1 = tf.train.AdamOptimizer(self.c.lr).minimize(self.loss_train, \
                                    var_list=self.actual_trainers)

    def print_bits_wrong(self):
        tf.Print(self.bits_wrong_train, [self.bits_wrong_train])

    def run_batch(self, session, batch_input_message, is_summary=False):
        feed_dict = {
            self.input_message_placeholder: batch_input_message
        }

        if not is_summary:
            loss_total, bits_wrong_1, _op1 = \
                                session.run([self.loss_train, self.bits_wrong_train, \
                                self.optimizer_1], feed_dict=feed_dict)
        else:
            merge = tf.summary.merge_all()
            return session.run(merge, feed_dict=feed_dict)


def main(debug=True):

    config = Config()
    with tf.Graph().as_default():
        lw = layers_and_weights()
        customs = {
            'lw': lw
        }

        if debug:
            config.num_batches = 51
            config.num_epochs = 1

        p2p_model = P2PE_DNNS(config, **customs)
        init = tf.global_variables_initializer()
        writer = tf.summary.FileWriter('./tf_summary/crypto/1')

        with tf.Session() as sess:
            sess.run(init)
            writer.add_graph(sess.graph)

            for epoch in range(config.num_epochs):
                for batches in range(config.num_batches):
                    batch_message = DataClass.get_batch_sized_data(config.batch_size, \
                        config.plain_text_length)
                    if batches % 50 == 0:
                        # write summaries.
                        s = p2p_model.run_batch(sess, batch_message, True)
                        print("Summarizing - " + str((config.num_batches * epoch) + batches))
                        writer.add_summary(s, (config.num_batches * epoch) + batches)
                    else:
                        res = p2p_model.run_batch(sess, batch_message)
    if not debug:
        print(res)
    return 0


if __name__ == '__main__':
    debug = False
    main(debug)