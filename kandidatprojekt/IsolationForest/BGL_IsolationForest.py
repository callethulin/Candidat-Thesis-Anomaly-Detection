import csv
import numpy as np
import sys
sys.path.append('/Users/callethulin/Desktop/T6/TDDE53/Projektet/loglizer') # Path to loglizer
from loglizer import dataloader
from loglizer.models import IsolationForest



para = {
    'save_path': '/Users/callethulin/Desktop/T6/TDDE53/Projektet/Dataset/BGL/',
    'window_size': 6,
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





model = IsolationForest(contamination=0.30034808147668395, n_estimators=2000, max_samples=1000, random_state=42, max_features=5)
model.fit(event_count_matrix)

    
print('Test validation:')
precision, recall, f1 = model.evaluate(event_count_matrix, labels)
