#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('/Users/carl/Desktop/Kandidat/kandidatprojekt/LogPai/loglizer/')
from loglizer.models import PCA
from loglizer import dataloader, preprocessing

struct_log = '/Users/carl/Desktop/Kandidat/kandidatprojekt/data/HDFS_v1/HDFS.log_structured.csv' # The structured log file
label_file = '/Users/carl/Desktop/Kandidat/kandidatprojekt/data/HDFS_v1/preprocessed/anomaly_label.csv' # The anomaly label file

pkl_path = "../../proceeded_data/BGL"
threshold = 0.9
n_components = 0.91
n_components2 = 0.93
threshold2 = 0.9

#threshold 0.9, ncomponents 0.93: precis: 0.947, recall: 0.716 F1: 0.815

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = dataloader.load_HDFS(struct_log,
                                                                label_file=label_file,
                                                                window='session', 
                                                                train_ratio=1,
                                                                split_type='sequential',
                                                                )
    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf', 
                                              normalization='zero-mean')
    x_test = feature_extractor.transform(x_test)

    model = PCA(threshold=threshold, n_components=n_components2)
    model.fit(x_train)

    print('Train validation 1:')
    precision, recall, f1 = model.evaluate(x_train, y_train)
    



   
    