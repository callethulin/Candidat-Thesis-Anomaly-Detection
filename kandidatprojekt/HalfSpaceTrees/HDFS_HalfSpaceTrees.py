from river import metrics, preprocessing, anomaly
import csv
import time
from collections import defaultdict

# Path to the structured cleaned file
path_to_structured_cleaned_file = '/Users/callethulin/Desktop/T6/TDDE53/Projektet/Dataset/HDFS_v1/HDFS.log_structured_cleaned_blk_eventID.csv'

# Initialize model and metrics
halfSpaceTrees = anomaly.HalfSpaceTrees(seed=42)

scaler = preprocessing.MinMaxScaler()

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
THRESHOLD = 0.4

tp = 0
tn = 0
fp = 0
fn = 0
anomaly_total = 0
anomaly_count = 0
normal_total = 0
normal_count = 0

# Determine amount of training data based on percentage
def get_train_test_split(file_path, train_percentage):
    with open(file_path, "r") as f:
        line_count = sum(1 for _ in f)
    
    # Calculate the number of training samples based on the percentage
    train_size = int(line_count * (train_percentage / 100))
    return train_size
train_size = get_train_test_split(path_to_structured_cleaned_file, 80)

# Start measuring time
start_time = time.time()

# Open the file and process it
with open(path_to_structured_cleaned_file, 'r') as structuredFile:
    reader = csv.DictReader(structuredFile)  # Reads CSV as a dictionary
    for i, row in enumerate(reader):

        # Extract features
        blkID = row['blkID']
        eventID = row['EventID']

        # Append true label
        label = int(row['Label'])
        true_label = label == 1  # True if label is 1, False otherwise

        blk_event_occurrences[blkID][eventID] += 1
            
        # Training phase
        if i < train_size:
            if blkID not in blkID_label:
                blkID_label[blkID] = label
        elif i == train_size:
            if blkID not in blkID_label:
                blkID_label[blkID] = label
            for blkID in blk_event_occurrences:
                data_point = data_point_dict = {eid: blk_event_occurrences[blkID][eid] for eid in event_ids}
                scaler.learn_one(data_point)
                transformed_data_point = scaler.transform_one(data_point)
                if blkID_label[blkID] == 0:
                    halfSpaceTrees.learn_one(transformed_data_point)
            print('Training Done')
        # Test phase
        else:
            # Create the data point with only the occurrence counts
            data_point = data_point_dict = {eid: blk_event_occurrences[blkID][eid] for eid in event_ids}
            scaler.learn_one(data_point)
            transformed_data_point = scaler.transform_one(data_point)

            predicted_value = halfSpaceTrees.score_one(transformed_data_point)
            predicted = predicted_value > THRESHOLD

            precision.update(true_label, predicted)
            recall.update(true_label, predicted)
            f1.update(true_label, predicted)
            if true_label and predicted:
                tp += 1
            elif true_label and not predicted:
                fn += 1
            elif not true_label and not predicted:
                tn += 1
            elif not true_label and predicted:
                fp += 1

            if true_label:
                anomaly_total += predicted_value
                anomaly_count += 1
            else:
                normal_total += predicted_value
                normal_count += 1
        

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