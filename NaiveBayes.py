import pandas as pd
from collections import defaultdict
import sys
import csv
import math


def load_data(file_path):
    data = pd.read_csv(file_path, header=0, usecols=lambda x: x != 0)
    return data

def train_naive_bayes(train_data):
    from collections import defaultdict
    
    count = defaultdict(int)
    
    # Initialize counts with Laplace smoothing
    count_feature = defaultdict(lambda: defaultdict(lambda: 1))  # Starting counts at 1 instead of 0 for each feature value
    total_feature = defaultdict(lambda: defaultdict(int))  # For each feature class pair initialize total count
    
    # count the number of each class
    for index, row in train_data.iterrows():
        y = row['class']
        count[y] += 1
    
    # Initialize total counts for Laplace smoothing
    features = train_data.drop(columns=['class', 'Unnamed: 0']).columns
    for y in count:
        for feature in features:
            total_count = 0
            feature_values = set(train_data[feature])  # Determine the set of all possible feature values
            for value in feature_values:
                total_count += count_feature[feature][(value, y)]
            total_feature[feature][y] = total_count
    
    # calculate the probability of each class and the probability of each feature in each class with Laplace
    prob = {}
    total_classes = len(count)
    for y in count:
        prob[y] = (count[y] + 1) / (len(train_data) + total_classes) 
        for feature in features:
            unique_feature_vals = len(set(train_data[feature])) 
            feature_class_total = total_feature[feature][y] + unique_feature_vals  
            for xi in set(train_data[feature]):
                prob[(feature, xi, y)] = (count_feature[feature][(xi, y)] + 1) / feature_class_total
    return prob

def calculate_score(instance, class_label, prob):
    score = prob[class_label]
    for feature in instance.index[1:]:
        score *= prob.get((feature, instance[feature], class_label), 1)
    return score

def predict(test_data, prob):
    predictions = []
    for index, instance in test_data.iterrows():
        max_log_score = float('-inf')
        pred_class = None
        for class_label in prob:
            if isinstance(class_label, str):  
                log_score = math.log(prob[class_label]) 
                for feature in instance.index[1:]:      
                    if feature not in ['class', 'Unnamed: 0']:  
                        log_score += math.log(prob.get((feature, instance[feature], class_label), 1e-10))
                if log_score > max_log_score:
                    max_log_score = log_score
                    pred_class = class_label
        predictions.append(pred_class)
    return predictions

def calculate_accuracy(true_labels, pred_labels):
    correct_predictions = sum([1 for true, pred in zip(true_labels, pred_labels) if true == pred])
    accuracy = correct_predictions / len(true_labels)
    return accuracy

def main(train_file, test_file):
    train_data = load_data(train_file)
    test_data = load_data(test_file)

    prob = train_naive_bayes(train_data)

    # Make predictions
    predictions = predict(test_data, prob)

    # Add these lines to calculate and print accuracy
    true_labels = test_data['class'].tolist() 
    accuracy = calculate_accuracy(true_labels, predictions)
    print(f'The accuracy of the model is: {accuracy*100:.2f}%.')

    # Write predictions to a CSV file
    with open('sampleoutput.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
            
        # Write header
        header = list(test_data.columns) + ['Predicted Class']
        writer.writerow(header)
            
        # Write predictions
        for index, prediction in enumerate(predictions):
            instance = test_data.iloc[index].tolist()
            instance.append(prediction)
            writer.writerow(instance)        

    # Read CSV file and print formatted table
    with open('sampleoutput.csv', 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)

        # Find maximum length of each column
        max_lengths = [max(len(str(cell)) for cell in column) for column in zip(*rows)]

        # Print formatted table
        for row in rows:
            formatted_row = [cell.ljust(length + 2) for cell, length in zip(row, max_lengths)]
            print("".join(formatted_row))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python NaiveBayes.py <train_file> <test_file>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
