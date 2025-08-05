import os
import glob
import pandas as pd
import numpy as np

class DataProcessor:
    @staticmethod
    def read_landmarks_file(file_path):
        # Static method to read landmarks file and extract landmark coordinates
        with open(file_path, 'r') as file:
            landmarks = [[float(value) for value in line.split()[1:4]] for line in file if line.strip()]
            # Extracting landmark coordinates from each line of the file
        return np.array(landmarks)# Returning landmark coordinates as a numpy array

    @staticmethod
    def read_data(data_directory):
        # Static method to read data from a directory containing landmark files
        data, labels, genders = [], [], [] # Initializing lists to store data, labels, and genders
        expression_to_int = {}

        # Add key-value pairs one by one
        expression_to_int['Angry'] = 0
        expression_to_int['Disgust'] = 1
        expression_to_int['Fear'] = 2
        expression_to_int['Happy'] = 3
        expression_to_int['Sad'] = 4
        expression_to_int['Surprise'] = 5
        # expression_to_int = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Sad': 4, 'Surprise': 5}
        # Mapping expressions to integer labels
        h=['F', 'M']
        for gender in h:
            n=list(range(1,59))
            for subject_id in n:
                subject_dir = os.path.join(data_directory, f"{gender}{subject_id:03d}")
                # Generating subject directory path

                if not os.path.exists(subject_dir):
                    continue # Skipping if subject directory doesn't exist

                for expression, label in expression_to_int.items():
                    expr_dir = os.path.join(subject_dir, expression)
                    # Generating expression directory path

                    if not os.path.exists(expr_dir):
                        continue # Skipping if expression directory doesn't exist

                    for bnd_file_name in os.listdir(expr_dir):
                        if not bnd_file_name.endswith(".bnd"):
                            continue # Skipping if file is not a .bnd file

                        bnd_file_path = os.path.join(expr_dir, bnd_file_name)
                        landmarks_data = DataProcessor.read_landmarks_file(bnd_file_path)
                        # Reading landmarks data from .bnd file

                        data.append(landmarks_data.flatten())# Flattening landmark data and adding to data list
                        labels.append(label)# Adding label to labels list
                        genders.append(gender)# Adding gender to genders list

        column_names = [f'feature_{i}' for i in range(len(data[0]))]
        df = pd.DataFrame(data, columns=column_names)
        # Creating DataFrame from data list with column names as 'feature_i'
        df['label'] = labels # Adding 'label' column to DataFrame
        df['gender'] = genders # Adding 'gender' column to DataFrame

        return df # Returning DataFrame containing processed data

    @staticmethod
    def rotate_points_3d(points, axis, degrees=180):
         # Static method to rotate 3D points around specified axis by given degrees
        radians = np.radians(degrees) # Converting degrees to radians
        cos_val = np.cos(radians) # Calculating cosine value
        sin_val = np.sin(radians) # Calculating sine value

        if axis == 'x':
            rotation_matrix= np.array([[1, 0, 0],
                             [0, cos_val, -sin_val],
                             [0, sin_val, cos_val]])
            # Rotation matrix for x-axis rotation
        elif axis == 'y':
            rotation_matrix= np.array([[cos_val, 0, sin_val],
                             [0, 1, 0],
                             [-sin_val, 0, cos_val]])
            # Rotation matrix for y-axis rotation
        elif axis == 'z':
            rotation_matrix= np.array([[cos_val, -sin_val, 0],
                             [sin_val, cos_val, 0],
                             [0, 0, 1]])
            # Rotation matrix for z-axis rotation
        else:
            raise ValueError("Invalid axis. Axis must be 'x', 'y', or 'z'.")
        # Raise ValueError if axis is not 'x', 'y', or 'z'

        return np.dot(points, rotation_matrix.T)
    # Return rotated points by multiplying with the transpose of rotation matrix
        

    @staticmethod
    def translate_to_origin(points):
        # Static method to translate points to origin by subtracting mean
        return points - np.mean(points, axis=0)
    # Return translated points by subtracting mean along each axis

