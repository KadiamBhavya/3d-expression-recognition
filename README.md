# Project Title: 3D Facial Expression Recognition

## Tech Stack
Programming Language: Python 
Machine Learning Models: Random Forest, Support Vector Machine (SVM), Decision Tree
Visualization: Matplotlib (3D scatter plots)
Libraries: scikit-learn, pandas, numpy

## Problem Statement
Recognize human emotions by analyzing 3D facial landmarks.
Evaluate how different transformations (translation, rotation) affect classification accuracy.
Compare performance of traditional classifiers under 10-fold subject-independent cross-validation.

## Solution
Preprocess 3D facial landmarks by flattening, translating to origin, and rotating across X/Y/Z axes.
Train models using Random Forest, SVM, and Decision Tree.
Visualize results and compute performance metrics (accuracy, precision, recall, confusion matrix).

# How to Setup
## Backend (Python Environment):
Navigate to the root project directory and create a Python virtual environment:
python -m venv env

Activate the environment:
On Windows:
.\env\Scripts\activate
On macOS/Linux:
source env/bin/activate

Install dependencies:
pip install -r requirements.txt

# Running the Project:
Use the following command to start the evaluation pipeline:
python main.py <algorithm> <data_type> <dataset_path>

## Arguments:
<algorithm>: RF (Random Forest), SVM, or DT (Decision Tree)
<data_type>:
o: Original data
t: Translated to origin
x, y, z: Rotated around X, Y, Z axes
<dataset_path>: Path to root folder of the BU-4DFE dataset (e.g., ./BU4DFE_BND_V1.1)

# Example:
python main.py RF x ./BU4DFE_BND_V1.1

# How Does It Work?
## Step 1: Data Loading
Parses .bnd files containing 83 3D landmarks per expression.
Converts each face into a 249-dimension vector (flattened x, y, z).

## Step 2: Preprocessing Options
Original: Uses raw landmarks
Translated: Centers landmarks to origin using mean subtraction
Rotated: Rotates points 180° across X, Y, or Z axis

## Step 3: Model Training
10-fold cross-validation is applied.
Models are evaluated on unseen subjects.

## Step 4: Metrics & Visualization
Outputs:
   Average confusion matrix
   Accuracy, Precision, Recall
Generates 3D scatter plots of a random face in:
   Original
   Translated
   Rotated (X, Y, Z)
  
# Glossary
3D Landmark: A point in 3D space representing a facial keypoint.
Cross-Validation: A method to ensure model generalization by splitting the data into training and testing sets multiple times.
Rotation Matrix: A transformation matrix used to rotate vectors in 3D space.
Flattening: Converting 3D landmark arrays to a 1D vector for ML input.

# Use-Cases
P0: Emotion Detection in Human-Computer Interaction
Identify user emotions in real-time using facial geometry.
P1: Surveillance & Security
Detect suspicious behavior through subtle facial expression cues.
P2: Clinical Research
Analyze facial expressions in patients for mental health diagnostics.

# Solution Architecture
Input: .bnd files of 3D facial landmarks

Process:
Preprocessing (translate, rotate)
Train classifier (RF/SVM/DT)
Evaluate using 10-fold CV

Output:
Metrics: Accuracy, Precision, Recall, Confusion Matrix
3D plots of landmarks










