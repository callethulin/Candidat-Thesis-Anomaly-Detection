import csv
import numpy as np
import sys
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
from sklearn.svm import LinearSVC

sys.path.append('/Users/callethulin/Desktop/T6/TDDE53/Projektet/loglizer') # Path to loglizer

from loglizer import dataloader
from loglizer.models import SVM

para = {
    'save_path': '/Users/callethulin/Desktop/T6/TDDE53/Projektet/Dataset/BGL/',
    'window_size': 8,
    'step_size': 3
}

raw_data = []
event_mapping_data = []

# Open the file and process it
with open('/Users/callethulin/Desktop/T6/TDDE53/Projektet/Dataset/BGl/BGL_full.log_structured_v2.csv', 'r') as structuredFile:
    reader = csv.DictReader(structuredFile)  # Reads CSV as a dictionary
    for i, row in enumerate(reader):
        if row['Label'] == '-':
            Label = 0
        else:
            Label = 1
        Time = int(row['Timestamp'])
        raw_data.append((Label, Time))

        EventId = int(row['EventId'].replace("E", ""))
        event_mapping_data.append([EventId])
        
        
       
raw_data = np.array(raw_data)
event_count_matrix, labels = dataloader.bgl_preprocess_data(para, raw_data, event_mapping_data)
labels = np.array(labels)

#Filip och Albins kod
num_chunks = 10
test_fraction_per_chunk = 0.2

total_len = len(labels)
chunk_size = total_len // num_chunks
test_indices = []

for i in range(num_chunks):
    start = i * chunk_size
    end = (i + 1) * chunk_size if i < num_chunks - 1 else total_len

    X_chunk = event_count_matrix[start:end]
    y_chunk = labels[start:end]
    counts = Counter(y_chunk)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction_per_chunk, random_state=42)
    for _, chunk_test_idx in sss.split(X_chunk, y_chunk):
        test_indices.extend(start + idx for idx in chunk_test_idx)


# Create corresponding train indices
all_indices = set(range(total_len))
train_indices = list(all_indices - set(test_indices))


# Final train/test split
X_train = event_count_matrix[train_indices]
y_train = labels[train_indices]
X_test = event_count_matrix[test_indices]
y_test = labels[test_indices]


# Show anomaly balance in test set
anomalies_in_test = np.sum(y_test)
total_anomalies = np.sum(labels)
print(f"Anomaly rate in dataset: {anomalies_in_test/total_anomalies:.2%}")



model = SVM()
model.classifier = LinearSVC(
    penalty=model.classifier.penalty,
    tol=model.classifier.tol,
    C=model.classifier.C,
    dual=model.classifier.dual,
    class_weight=model.classifier.class_weight,
    max_iter=model.classifier.max_iter,
    random_state=42  
)
model.fit(X_train, y_train) 
    
print('Test validation:')
precision, recall, f1 = model.evaluate(X_test, y_test)