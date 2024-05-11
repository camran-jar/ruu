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

                #print(f"Tuple number: {len(prob)}, Feature: {feature} ({xi}, {y}), Probability: {prob[(feature, xi, y)]}")
   
    return prob

def calculate_score(instance, class_label, prob):
    score = prob[class_label]
    for feature in instance.index[1:]:
        score *= prob.get((feature, instance[feature], class_label), 1)
    return score

def predict(test_data, prob):
    predictions = []
    for index, instance in test_data.iterrows():
        max_prob_score = float('-inf')
        pred_class = None
        for class_label in prob:
            if isinstance(class_label, str):  
                prob_score = prob[class_label]
                for feature in instance.index[1:]:      
                    if feature not in ['class', 'Unnamed: 0']:  
                        prob_feature_y_given_class = prob.get((feature, instance[feature], class_label), 1e-10)
                        print(f"Feature: {feature} ({instance[feature]}, {class_label}), Probability: {prob_feature_y_given_class}")  # print feature, value, class, and associated probability
                        prob_score *= prob_feature_y_given_class
                if prob_score > max_prob_score:
                    max_prob_score = prob_score
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

    # Print class probabilities
    class_probs = {class_label: prob[class_label] for class_label in prob if isinstance(class_label, str)}
    print("Class probs:", class_probs)

    # Print feature probabilities
    features = train_data.drop(columns=['class', 'Unnamed: 0']).columns
    for feature in features:
        feature_probs = defaultdict(float)
        for y in class_probs:
            for value in set(train_data[feature]):
                feature_probs[(feature, value, y)] = prob.get((feature, value, y), 0)
        # Normalize feature probabilities to sum up to 1 for each class
        for y in class_probs:
            total_prob = sum(feature_probs[(feature, value, y)] for value in set(train_data[feature]))
            if total_prob != 0:  # Avoid division by zero
                for value in set(train_data[feature]):
                    feature_probs[(feature, value, y)] /= total_prob
        print(f"Feature: {feature}")
        for key, value in feature_probs.items():
            print(f"  {key}, sum = {value:.2f}")

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
