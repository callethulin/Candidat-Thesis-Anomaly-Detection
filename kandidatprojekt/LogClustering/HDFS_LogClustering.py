#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append('/Users/callethulin/Desktop/T6/TDDE53/Projektet/loglizer') # Path to loglizer
from loglizer import dataloader
from loglizer.models import LogClustering
from loglizer import preprocessing

struct_log = '/Users/callethulin/Desktop/T6/TDDE53/Projektet/Dataset/HDFS_v1/HDFS.log_structured.csv' # The structured log file
label_file = '/Users/callethulin/Desktop/T6/TDDE53/Projektet/Dataset/HDFS_v1/anomaly_label.csv' # The anomaly label file
max_dist = 0.3 # the threshold to stop the clustering process
anomaly_threshold = 0.3 # the threshold for anomaly detection

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = dataloader.load_HDFS(struct_log,
                                                                label_file=label_file,
                                                                window='session', 
                                                                train_ratio=1,
                                                                split_type='sequential')
    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf')
    

    model = LogClustering(max_dist=max_dist, anomaly_threshold=anomaly_threshold, mode='online')
    model.fit(x_train[y_train == 0, :]) # Use only normal samples for training

    print('Test validation:')
    precision, recall, f1 = model.evaluate(x_train, y_train)
    