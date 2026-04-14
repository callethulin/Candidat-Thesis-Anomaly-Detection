#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append('/Users/callethulin/Desktop/T6/TDDE53/Projektet/loglizer') # Path to loglizer

from loglizer import dataloader
from loglizer.models import IsolationForest
from loglizer import preprocessing

struct_log = '/Users/callethulin/Desktop/T6/TDDE53/Projektet/Dataset/HDFS_v1/HDFS.log_structured.csv' # The structured log file
label_file = '/Users/callethulin/Desktop/T6/TDDE53/Projektet/Dataset/HDFS_v1/anomaly_label.csv' # The anomaly label file
anomaly_ratio = 0.03 # Estimate the ratio of anomaly samples in the data

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = dataloader.load_HDFS(struct_log,
                                                                label_file=label_file,
                                                                window='session', 
                                                                train_ratio=1,
                                                                split_type='sequential',
                                                                save_csv=False
                                                                )
    
    
    feature_extractor = preprocessing.FeatureExtractor()

    x_train = feature_extractor.fit_transform(x_train)
    
    model = IsolationForest(contamination=anomaly_ratio)
    model.fit(x_train)

    print('Test validation:')
    precision, recall, f1 = model.evaluate(x_train, y_train)
    