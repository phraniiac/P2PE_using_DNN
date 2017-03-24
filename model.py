#####################################################
#
#
#   github - http://github.com/phraniiac
#   project - tensorflow boilerplate
#
#
#####################################################

import abc
import numpy as np
import tensorflow as tf


class Config:
    """ Config class to hold config vars.
        An instance to be passed to the actual model.
    """
    def __init__(self, **kw):
        self.dropout = 0.4
        self.lr = 0.01
        self.batch_size = 4096
        self.plain_text_length = 16
        self.cipher_text_length = 128
        self.key_length = self.plain_text_length
        self.num_epochs = 10
        self.num_batches = 1000

        ##
        self.update_vars(**kw)

    def update_vars(self, **kw):
        # Update manual configs.
        self.__dict__.update(kw)


class Nueral_Net_Model:
    """ Model class to be extended to train a Tensorflow
        model. The initializer needs a config object.
    """
    __metaclass__  = abc.ABCMeta

    # Taken from - https://www.tensorflow.org/get_started/summaries_and_tensorboard
    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for \
            TensorBoard visualization).
        """
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


    def __init__(self, config, **kw):
        self.c = config
        # add custom helper classes and functionality.
        self.add_customs(**kw)
        self.ready_model()
    
    def add_to_summaries(self, variable_list):
        for variable in variable_list:
            self.variable_summaries(variable)
    
    def add_customs(self, **kw):
        self.__dict__.update(kw)
    
    def ready_model(self):
        self.add_placeholders()
        self.add_training_vars()
        self.add_training_op()
        self.add_loss_op()
        self.add_optimizers()

    @abc.abstractmethod
    def add_placeholders(self):
        # Add placeholders in your tensorflow graph.
        raise NotImplementedError

    @abc.abstractmethod
    def add_training_vars(self):
        # Add training variables in your tensorflow graph.
        raise NotImplementedError

    @abc.abstractmethod
    def add_training_op(self):
        # Add training ops in your tensorflow graph.
        # All the matmul's and others should be here.
        raise NotImplementedError

    @abc.abstractmethod
    def add_loss_op(self):
        # Add training ops in your tensorflow graph.
        # All the loss vars are to be declared here.
        raise NotImplementedError

    @abc.abstractmethod
    def add_optimizers(self):
        # Add optimizers in your tensorflow graph.
        # All the optimizers should be added with proper
        # training vars.
        raise NotImplementedError

    @abc.abstractmethod
    def run_epoch(self, session, is_summary=False):
        # Run a batch of data, take the session, and
        # create feed_dict from the data and periodically
        # write summaries in the summary writer.
        raise NotImplementedError