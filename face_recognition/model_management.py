from __future__ import absolute_import, division, print_function

import argparse
import copy
import math
import os
import pickle
import re
import sys
import time
from os.path import join as pjoin

import tensorflow as tf
from scipy import misc
from sklearn.externals import joblib
from sklearn.svm import SVC
from tensorflow.python.platform import gfile

import detect_face


class Model():
    @staticmethod
    def get_model_filenames(model_dir):
        files = os.listdir(model_dir)
        meta_files = [s for s in files if s.endswith('.meta')]
        if len(meta_files) == 0:
            raise ValueError(
                'No meta file found in the model directory (%s)' % model_dir)
        elif len(meta_files) > 1:
            raise ValueError(
                'There should not be more than one meta file in the model directory (%s)' % model_dir)
        meta_file = meta_files[0]
        meta_files = [s for s in files if '.ckpt' in s]
        max_step = -1
        for f in files:
            step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
            if step_str is not None and len(step_str.groups()) >= 2:
                step = int(step_str.groups()[1])
                if step > max_step:
                    max_step = step
                    ckpt_file = step_str.groups()[0]
        return meta_file, ckpt_file

    @staticmethod
    def load_model(model):
        # Check if the model is a model directory (containing a metagraph and a checkpoint file)
        #  or if it is a protobuf file with a frozen graph
        model_exp = os.path.expanduser(model)
        if (os.path.isfile(model_exp)):
            print('Model filename: %s' % model_exp)
            with gfile.FastGFile(model_exp, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
        else:
            print('Model directory: %s' % model_exp)
            meta_file, ckpt_file = get_model_filenames(model_exp)

            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)

            saver = tf.train.import_meta_graph(
                os.path.join(model_exp, meta_file))
            saver.restore(tf.get_default_session(),
                          os.path.join(model_exp, ckpt_file))
