# File: scripts/data_loading.py

import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    data = pd.read_csv(file_path)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    return train_data, test_data

def preprocess_data(train_data, test_data):
    # Assume your data has features in a column named 'features'
    # and labels in a column named 'labels'
    train_images = train_data['features'].values
    train_labels = train_data['labels'].values
    test_images = test_data['features'].values
    test_labels = test_data['labels'].values

    return train_images, train_labels, test_images, test_labels

def data_augmentation():
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip(),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
    ])
    return data_augmentation

if __name__ == "__main__":
    train_data, test_data = load_data('data/train_data.csv')
    train_images, train_labels, test_images, test_labels = preprocess_data(train_data, test_data)
    augment = data_augmentation()
