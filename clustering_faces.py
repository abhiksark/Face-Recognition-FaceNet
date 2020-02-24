#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
###################################################################
# File Name: classifying_faces.py
# Author: Abhik Sarkar
# mail: abhiksark@gmail.com
# Created Time: Thu Oct  5 02:27:35 2017 IST
###################################################################
"""


from __future__ import absolute_import, division, print_function

import argparse
import math
import os
import pickle
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.svm import SVC

import face_recognition.detect_face as detect_face
import face_recognition.facenet as facenet

DATA_DIR = './faces'

with tf.Graph().as_default():
    with tf.Session() as sess:
        dataset = facenet.get_dataset(DATA_DIR)
        paths, labels = facenet.get_image_paths_and_labels(dataset)

        print('Number of classes: %d' % len(dataset))
        print('Number of images: %d' % len(paths))
        print('Loading feature extraction model')

        modeldir = './20170511-185253/20170511-185253.pb'
        facenet.load_model(modeldir)

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        # Run forward pass to calculate embeddings
        print('Calculating features for images')
        batch_size = 25
        image_size = 160

        nrof_images = len(paths)
        nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))
        emb_array = np.zeros((nrof_images, embedding_size))
        for i in range(nrof_batches_per_epoch):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, nrof_images)
            paths_batch = paths[start_index:end_index]
            images = facenet.load_data(paths_batch, False, False, image_size)
            feed_dict = {images_placeholder: images,
                         phase_train_placeholder: False}
            emb_array[start_index:end_index, :] = sess.run(
                embeddings, feed_dict=feed_dict)

names = []
for i in range(1, 129):
    names.append("col"+str(i))

value = emb_array
key = paths
d = dict(zip(key, value))
df = pd.DataFrame.from_dict(d, orient='index', columns=names)

X = df.iloc[:, :].values
y = df.index.values

kmeans = KMeans(n_clusters=10, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

for i in range(len(paths)):
    print(str(y_kmeans[i]) + "/" + str(y[i].split("/")[-1]))
    try:
        os.rename(y[i], "./faces/group_photos/" +
                  str(y_kmeans[i]) + "/" + str(y[i].split("/")[-1]))
    except:
        os.mkdir("./faces/group_photos/" + str(y_kmeans[i]))
        print()
        os.rename(y[i], "./faces/group_photos/" +
                  str(y_kmeans[i]) + "/" + str(y[i].split("/")[-1]))
