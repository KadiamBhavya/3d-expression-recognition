import sys
from classifier_trainer import ClassifierTrainer
from data_processor import DataProcessor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier



if __name__ == '__main__':
    # Checking if correct number of command-line arguments is provided
    if len(sys.argv) != 4:
        print("Usage: python main.py <algorithm> <data_type> <dataset_path>")# Print usage instruction if incorrect arguments are provided
        sys.exit(1)
    a=ClassifierTrainer()
    algo, data_type, dataset_path = sys.argv[1:] # Extracting algorithm, data_type, and dataset_path from command-line arguments
    
    # Reading data from dataset
    df = DataProcessor.read_data(dataset_path)# Reading data from the provided dataset path using DataProcessor class
    x_data, y_data = a.get_processed_data(data_type, df)# Processing data using a class

    # Dictionary mapping algorithm names to corresponding classifier classes
    classifiers = {'RF': RandomForestClassifier,
                   'SVM': SVC,
                   'DT': DecisionTreeClassifier}

    # Checking if the specified algorithm is supported
    if algo in classifiers:
        a.train_classifier(classifiers[algo], x_data, y_data.values.ravel())# Training classifier using a class
    else:
        print("Invalid algorithm. Please choose RF, SVM, or DT.")
 
    a.visualization(df)
    

