from river import metrics, preprocessing, anomaly, feature_extraction
import csv
import time
from collections import defaultdict
from collections import deque

# Path to the structured cleaned file
path_to_structured_cleaned_file = '/Users/carl/Desktop/Kandidat/kandidatprojekt/data/BGL/BGL_full.log_structured_v2.csv'

# Initialize model and metrics
model = preprocessing.StandardScaler()  | anomaly.QuantileFilter(anomaly.OneClassSVM(nu=0.053), q=0.995)
precision = metrics.Precision()
recall = metrics.Recall()
f1 = metrics.F1()

#nu = 0.05, q=0.995, threshold = 30, prec 16.93, recall 57.99 F1 26.21

# Lägg till en räknare för antal vektorer
vector_count = 0
sliding_window_size = 100

sliding_window = deque(maxlen=sliding_window_size)
event_counts = defaultdict(int)
# Initialize event counts
event_counts = [0] * 320  # Assuming 320 unique EventIds

# Define prediction threshold
THRESHOLD = 30

tp = 0
tn = 0
fp = 0
fn = 0
anomaly_total = 0
anomaly_count = 0
normal_total = 0
normal_count = 0
highestvalue = 0

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
        label = row['Label']
        eventID = int(row['EventId'].replace("E",""))
       
      
        # Append true label
        true_label = label != '-'  # False if label is -, True otherwise
    
       # if len(sliding_window) == sliding_window.maxlen:
        #    oldest = sliding_window.popleft()
       #     event_counts[oldest-1] -= 1

       # sliding_window.append(eventID)
       # event_counts[eventID-1] += 1
       # sliding_dict = {i: count for i, count in enumerate(event_counts)}
        
        #model.learn_one(sliding_dict)
            
        # Training phase
        if i < train_size:
            if not true_label:
                if len(sliding_window) == sliding_window.maxlen:
                    oldest = sliding_window.popleft()
                    event_counts[oldest-1] -= 1

                sliding_window.append(eventID)
                event_counts[eventID-1] += 1
                sliding_dict = {i: count for i, count in enumerate(event_counts)}
        
                model.learn_one(sliding_dict)
    
                transformed_data_point = model.transform_one(sliding_dict)
                model.learn_one(transformed_data_point)

        # Test phase
        else:
            if i == train_size:
                print("Training complete. Starting testing phase.")
            if len(sliding_window) == sliding_window.maxlen:
                    oldest = sliding_window.popleft()
                    event_counts[oldest-1] -= 1

            sliding_window.append(eventID)
            event_counts[eventID-1] += 1
            sliding_dict = {i: count for i, count in enumerate(event_counts)}
        
            model.learn_one(sliding_dict)

            transformed_data_point = model.transform_one(sliding_dict)

            predicted_value = model.score_one(transformed_data_point)
            predicted = predicted_value > THRESHOLD

            if not true_label:
                if predicted_value > highestvalue:
                    highestvalue = predicted_value

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
print("Highest Value normal data:", highestvalue)
print(precision)
print(recall)
print(f1)
print('TP:', tp)
print('TN:', tn)
print('FP:', fp)
print('FN:', fn)
print(f"Anomaly Mean Score: {anomaly_total/anomaly_count}")
print(f"Normal Mean Score: {normal_total/normal_count}")
print("Amount of Normals:", normal_count)
print("Amount of Anomalies:", anomaly_count)