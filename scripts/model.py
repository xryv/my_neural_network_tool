# File: scripts/model.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

def get_model():
    base_model = tf.keras.applications.EfficientNetB7(weights='imagenet', include_top=False)
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    return model
