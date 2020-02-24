#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
###################################################################
# File Name: labeling_faces.py
# Author: Abhik Sarkar
# mail: abhiksark@gmail.com
# Created Time: Thu Oct  5 02:27:35 2017 IST
###################################################################
"""


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

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy import misc
from sklearn.externals import joblib
from sklearn.svm import SVC
from tensorflow.python.platform import gfile

import face_recognition.detect_face as detect_face
import face_recognition.facenet as facenet


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

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        saver.restore(tf.get_default_session(),
                      os.path.join(model_exp, ckpt_file))


print('Creating networks and loading parameters')
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(
        gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, './data')

        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        margin = 44
        frame_interval = 3
        batch_size = 10
        image_size = 182

        input_image_size = 160
        minsize = 100  # minimum size of face
        threshold = [0.7, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        image_size = 182

        HumanNames = os.listdir("./labelled_faces")  # train human name
        HumanNames.sort()
        print('Loading feature extraction model')
        modeldir = './20170511-185253/20170511-185253.pb'
        load_model(modeldir)

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        classifier_filename = './cls/my_classifier.pkl'
        classifier_filename_exp = os.path.expanduser(classifier_filename)
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile)
            print('load classifier file-> %s' % classifier_filename_exp)
        c = 0

        ret = True
        frame = cv2.imread('./test.jpg')

        if frame.ndim == 2:
            frame = facenet.to_rgb(frame)
        frame = frame[:, :, 0:3]
        bounding_boxes, _ = detect_face.detect_face(
            frame, minsize, pnet, rnet, onet, threshold, factor)
        nrof_faces = bounding_boxes.shape[0]
        print('Detected_FaceNum: %d' % nrof_faces)
        det = bounding_boxes[:, 0:4]
        det_morethanone = det
        if nrof_faces > 0:
            for i in range(nrof_faces):
                det = det_morethanone[int(i), :]
                img_size = np.asarray(frame.shape)[0:2]
                cropped = []
                scaled = []
                scaled_reshape = []
                bb = np.zeros((nrof_faces, 4), dtype=np.int32)
                emb_array = np.zeros((1, embedding_size))

                bb[i][0] = det[0]
                bb[i][1] = det[1]
                bb[i][2] = det[2]
                bb[i][3] = det[3]

                # inner exception
                if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                    print('face is inner of range!')
                    continue

                cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                cropped[0] = facenet.flip(cropped[0], False)
                scaled.append(misc.imresize(
                    cropped[0], (image_size, image_size), interp='bilinear'))
                scaled[0] = cv2.resize(scaled[0], (input_image_size, input_image_size),
                                       interpolation=cv2.INTER_CUBIC)
                scaled[0] = facenet.prewhiten(scaled[0])
                scaled_reshape.append(
                    scaled[0].reshape(-1, input_image_size, input_image_size, 3))
                feed_dict = {
                    images_placeholder: scaled_reshape[0], phase_train_placeholder: False}
                emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                predictions = model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(
                    len(best_class_indices)), best_class_indices]
                # boxing face
                cv2.rectangle(frame, (bb[i][0], bb[i][1]),
                              (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                # plot result idx under box
                text_x = bb[i][0]
                text_y = bb[i][3] + 20
                print('result: ', best_class_probabilities)
                for H_i in HumanNames:
                    if HumanNames[best_class_indices[0]] == H_i:
                        result_names = HumanNames[best_class_indices[0]]
                        cv2.putText(frame, result_names, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1, (255, 255, 0), thickness=2, lineType=8)

            cv2.imwrite('./'+str("prediction")+".jpg", frame)

        else:
            print('Unable to align')
