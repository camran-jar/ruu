import pandas as pd
from collections import defaultdict
import sys
import csv
import math

# load the data from the csv, taking the header line and skipping the tuple ID row (0)
def load_data(file_path):
    data = pd.read_csv(file_path, header=0, usecols=lambda x: x != 0)
    return data

# Function to trian the Naive Bayes model
def train(train_data, alpha=1):
    # Initialise defaultdict to count occurrences of class and feature
    count = defaultdict(int)
    count_feature = defaultdict(lambda: defaultdict(int)) 
    
    # count the number of each class and each feature
    for index, row in train_data.iterrows():
        y = row['class']
        count[y] += 1
        for feature in train_data.drop(columns=['class', 'Unnamed: 0']):
            count_feature[feature][(row[feature], y)] += 1
    
    # calculate the probabilities with Laplace smoothing
    prob = {}
    total_classes = len(count)
    total_instances = len(train_data)
    for y in count:
        prob[y] = (count[y] + 1) / (total_instances + 1 * total_classes)
        for feature in train_data.drop(columns=['class', 'Unnamed: 0']):
            feature_values = set(train_data[feature])
            for xi in feature_values:
                feature_class_total = sum(count_feature[feature][(xi, y)] for y in count)
                prob[(feature, xi, y)] = (count_feature[feature][(xi, y)] + 1) / (feature_class_total + 1 * len(feature_values))

    return prob


def predict(test_data, prob):
    predictions = []
    prob_scores = []

    for index, instance in test_data.iterrows():
        max_prob_score = float('-inf')
        pred_class = None
        class_p_score = {}

        for class_label in prob:
            if isinstance(class_label, str):
                prob_score = prob[class_label]
                for feature in instance.index[1:]:
                    if feature not in ['class', 'Unnamed: 0']:
                        prob_score *= prob.get((feature, instance[feature], class_label), 0)
                class_p_score[class_label] = prob_score

                if prob_score > max_prob_score:
                    max_prob_score = prob_score
                    pred_class = class_label
        
        # Debugging: Print instance and class probabilities
        #print(f"Instance: {instance}")
        #for class_label, prob_score in class_p_score.items():
        #    print(f" Training / Class: {class_label}, Prob Score: {prob_score}")

        predictions.append(pred_class)
        prob_scores.append(class_p_score)

    return predictions, prob_scores

# function to calculate the score for a certain instance and class label 
def calculate_score(instance, class_label, prob):
    score = prob[class_label]
    for feature in instance.index[1:]:
        score *= prob.get((feature, instance[feature], class_label), 1)
    return score

# Function to make predictions on the test data 
def predict(test_data, prob):
    predictions = []
    prob_scores = []

    for index, instance in test_data.iterrows():
        max_prob_score = float('-inf')
        pred_class = None
        class_p_score = {}

        for class_label in prob:
            if isinstance(class_label, str):
                prob_score = prob[class_label]
                for feature in instance.index[1:]:
                    if feature not in ['class', 'Unnamed: 0']:
                        prob_score *= prob.get((feature, instance[feature], class_label), 0)
                class_p_score[class_label] = prob_score

                if class_p_score[class_label] > max_prob_score:
                    max_prob_score = class_p_score[class_label]
                    pred_class = class_label
                    
        # Debugging: Print instance and class probabilities
        print(f"Instance: {instance}")
        for class_label, prob_score in class_p_score.items():
            print(f"Class: {class_label}, Prob Score: {prob_score}")

        predictions.append(pred_class)
        prob_scores.append(class_p_score)

    return predictions, prob_scores


def calculate_accuracy(true_labels, pred_labels):
    correct_predictions = sum([1 for true, pred in zip(true_labels, pred_labels) if true == pred])
    accuracy = correct_predictions / len(true_labels)
    return accuracy

def write_csv(predictions, prob_scores, test_data, output_file = 'sampleoutput.csv'):
    # Write predictions to a CSV file
    with open('sampleoutput.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
            
        # Write header
        header = list(test_data.columns) + ['Predicted Class', 'Score (no-recurrence-events)', 'Score (recurrence-events)']
        writer.writerow(header)
            
        # Write predictions and scores
        for index, (prediction, prob_score) in enumerate(zip(predictions, prob_scores)):
            instance = test_data.iloc[index].tolist()
            instance.append(prediction)
            instance.append(prob_score['no-recurrence-events'])
            instance.append(prob_score['recurrence-events'])
            writer.writerow(instance)        

def print_table(file_path):
    with open('sampleoutput.csv', 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)

        # Find maximum length of each column
        max_lengths = [max(len(str(cell)) for cell in column) for column in zip(*rows)]

        # Print formatted table
        for row in rows:
            formatted_row = [cell.ljust(length + 2) for cell, length in zip(row, max_lengths)]
            print("".join(formatted_row))

def main(train_file, test_file):
    train_data = load_data(train_file)
    test_data = load_data(test_file)

    prob = train(train_data)

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
    predictions, prob_scores = predict(test_data, prob)

    # print accuracy
    true_labels = test_data['class'].tolist() 
    accuracy = calculate_accuracy(true_labels, predictions)
    print(f'The accuracy of the model is: {accuracy*100:.2f}%.')

    # write test predictions to csv
    write_csv(predictions, prob_scores, test_data)
    # print the csv file for checks
    print_table('sampleoutput.csv')


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python NaiveBayes.py <train_file> <test_file>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
