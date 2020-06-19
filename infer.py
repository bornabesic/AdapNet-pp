''' AdapNet++:  Self-Supervised Model Adaptation for Multimodal Semantic Segmentation

 Copyright (C) 2018  Abhinav Valada, Rohit Mohan and Wolfram Burgard

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.'''

import argparse
import datetime
import importlib
import os
import numpy as np
import tensorflow as tf
import yaml
import cv2
from pathlib import Path
from dataset.helper import *
from utils import semseg_image_to_carla_palette

PARSER = argparse.ArgumentParser()
PARSER.add_argument("modality", type=str)
PARSER.add_argument("label", type=str)
PARSER.add_argument('-c', '--config', default='config/cityscapes_test.config')

def test_func(config, path_modality, path_label):
    os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu_id']
    module = importlib.import_module('models.' + config['model'])
    model_func = getattr(module, config['model'])
    data_list, iterator = get_test_data(config)
    resnet_name = 'resnet_v2_50'

    with tf.variable_scope(resnet_name):
        model = model_func(num_classes=config['num_classes'], training=False)
        images_pl = tf.placeholder(tf.float32, [None, config['height'], config['width'], 3])
        model.build_graph(images_pl)

    config1 = tf.ConfigProto()
    config1.gpu_options.allow_growth = True
    sess = tf.Session(config=config1)
    sess.run(tf.global_variables_initializer())
    import_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    print('total_variables_loaded:', len(import_variables))
    saver = tf.train.Saver(import_variables)
    # saver.restore(sess, config['checkpoint'])
    saver.restore(sess, tf.train.latest_checkpoint(config['checkpoint']))
    sess.run(iterator.initializer)
    step = 0
    total_num = 0
    output_matrix = np.zeros([config['num_classes'], 3])
    img = cv2.imread(path_modality)[np.newaxis, ...].astype(np.float32)
    label = cv2.imread(path_label, cv2.IMREAD_ANYCOLOR)[np.newaxis, ...].astype(np.int64)
    # img - (B, H, W, C)
    feed_dict = {images_pl : img}
    probabilities = sess.run([model.softmax], feed_dict=feed_dict)
    prediction = np.argmax(probabilities[0], 3)
    gt = label
    # prediction - (B, H, W)
    # gt - (B, H, W)
    prediction[gt == 0] = 0
    prediction_cmap = semseg_image_to_carla_palette(prediction[0])
    cv2.imshow("RGB", img.astype(np.uint8)[0])
    cv2.imshow("SS", prediction_cmap[:, :, ::-1]); cv2.waitKey(0)

def main():
    args = PARSER.parse_args()
    assert Path(args.modality).exists()
    assert Path(args.label).exists()
    if args.config:
        file_address = open(args.config)
        config = yaml.load(file_address)
    else:
        print('--config config_file_address missing')
    test_func(config, args.modality, args.label)

if __name__ == '__main__':
    main()
