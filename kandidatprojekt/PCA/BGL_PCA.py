import csv
import numpy as np
import sys
sys.path.append('/Users/callethulin/Desktop/T6/TDDE53/Projektet/loglizer') # Path to loglizer

from loglizer import dataloader
from loglizer.models import PCA



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





model = PCA(threshold=100000, n_components=0.99)




model.fit(event_count_matrix)


    
print('Test validation:')
precision, recall, f1 = model.evaluate(event_count_matrix, labels)
