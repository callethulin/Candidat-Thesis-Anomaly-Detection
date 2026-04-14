from river import metrics, preprocessing, anomaly, feature_extraction
import csv
import time
from collections import defaultdict

# Path to the structured cleaned file
path_to_structured_cleaned_file = '/Users/carl/Desktop/Kandidat/kandidatprojekt/data/HDFS_v1/labeled_structuredHDFS.csv'

# Lägg till en räknare för antal vektorer
vector_count = 0

# Initialize model and metrics
model = preprocessing.StandardScaler()  | anomaly.QuantileFilter(anomaly.OneClassSVM(nu=0.03), q=0.995)
precision = metrics.Precision()
recall = metrics.Recall()
f1 = metrics.F1()

# Predefined eventIDs
event_ids = [
    'bbb51b95', '3d91fa85', 'd38aa58d', '46003790', '5d5de21c', '44614d71',
    '9c784e29', '75627efd', '54e5f6b4', '4dec0816', '728076ac', '40651754',
    '32777b38', 'ace40671', 'e817fa45', 'bcc910df', 'dba996ef', '0567184d',
    '69bca6e5', 'd013b7a3', 'c294d20f', 'a5478aff', 'f7c33085', 'f266840a',
    '15b484af', '9621139e', 'd62e5638', '1061e5d9', '2ecc047e', '4610d0f1',
    '1a718242', 'd63ef163', '2e68ccc3', '8f2bc724', 'a333d363', 'e024fa48',
    '25382c88', '20317105', '06d16156', '78915d3a', '13eb7010', 'f79898ae',
    '124068c6', '625a2a34', 'b65fc512', 'fcd37a6d', '559305d8', 'fcf2c482'
]

# Dictionary to store occurrences per blkID
blk_event_occurrences = defaultdict(lambda: {eid: 0 for eid in event_ids})
blkID_label = {}
# Define prediction threshold
THRESHOLD = 17.82

tp = 0
tn = 0
fp = 0
fn= 0
anomaly_total = 0
anomaly_count = 0
normal_total = 0
normal_count = 0

# Determine amount of training data
def get_train_test_split(file_path):
    with open(file_path, "r") as f:
        line_count = sum(1 for _ in f)
    return line_count*0.8  # 80% for training  
train_test_split = get_train_test_split(path_to_structured_cleaned_file)

# Start measuring time
start_time = time.time()

# Open the file and process it
with open(path_to_structured_cleaned_file, 'r') as structuredFile:
    reader = csv.DictReader(structuredFile)  # Reads CSV as a dictionary
    for i, row in enumerate(reader):
        
        # Extract features
        blkID = row['blkID']
        eventID = row['EventId']

        # Inkrementera räknaren för vektorer
        vector_count += 1

        # Append true label
        label = (row['Label'])
        if label == "Normal":
            true_label = 0
        else:
            true_label = 1
       

        # Ensure the eventID is one of the predefined ones
        if eventID in event_ids:
            # Increment the count of the eventID for that blkID
            blk_event_occurrences[blkID][eventID] += 1
        
        
        # Training phase
        if i < train_test_split:
            if blkID not in blkID_label:
                blkID_label[blkID] = label
        elif i == train_test_split:
            if blkID not in blkID_label:
                blkID_label[blkID] = label
            for blkID in blk_event_occurrences:
                data_point = data_point_dict = {eid: blk_event_occurrences[blkID][eid] for eid in event_ids}
                model.learn_one(data_point)
                transformed_data_point = model.transform_one(data_point)
                if blkID_label[blkID] == 0:
                    model.learn_one(transformed_data_point)
            print('Training Done')
  

            
        # Test phase
        else:
           # if true_label == 0:
              #  model.learn_one(transformed_data_point)#Alternativet är att träna på den data modellen anser normal
            
            data_point = data_point_dict = {eid: blk_event_occurrences[blkID][eid] for eid in event_ids}
            model.learn_one(data_point)
            transformed_data_point = model.transform_one(data_point)
            predicted_value = model.score_one(transformed_data_point)
            predicted = predicted_value > THRESHOLD
            #if not predicted:
            
           

        
            
            precision.update(true_label, predicted)
            recall.update(true_label, predicted)
            f1.update(true_label, predicted)

            # Update confusion matrix
            if true_label == 1 and predicted:  # True positive
                tp += 1
            elif true_label == 1 and not predicted:  # False negative
                fn += 1
            elif true_label == 0 and not predicted:  # True negative
                tn += 1
            elif true_label == 0 and predicted:  # False positive
                fp += 1

            # Track anomaly and normal scores for reporting
            if true_label == 1:  # Anomaly
                anomaly_total += predicted_value
                anomaly_count += 1
            else:  # Normal
                normal_total += predicted_value
                normal_count += 1

# Efter att alla vektorer har behandlats, skriv ut antalet vektorer
print(f"Totalt antal vektorer skapade: {vector_count}")

# End measuring time
end_time = time.time()

# Print the execution time
execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.4f} seconds")

# Print the results
print(precision)
print(recall)
print(f1)
print('TP:', tp)
print('TN:', tn)
print('FP:', fp)
print('FN:', fn)
print(f"Anomaly Mean Score: {anomaly_total/anomaly_count}")
print(f"Normal Mean Score: {normal_total/normal_count}")
