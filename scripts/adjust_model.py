# File: scripts/adjust_model.py

import tensorflow as tf
from scripts.model import get_model
from scripts.data_loading import load_data, preprocess_data
from scripts.training import train_model

def adjust_layers(model):
    # Example of adjusting layers
    model.layers[1].units = 512  # Adjusting units of the second layer
    return model

def adjust_optimizer(learning_rate):
    # Adjusting optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    return optimizer

def adjust_training_config(epochs, batch_size):
    # Adjusting training configurations
    return epochs, batch_size

if __name__ == "__main__":
    # Load and preprocess data
    train_data, test_data = load_data('data/train_data.csv')
    train_images, train_labels, test_images, test_labels = preprocess_data(train_data, test_data)

    # Get existing model or a new model
    try:
        model = tf.keras.models.load_model('models/my_model/saved_model.pb')
    except:
        model = get_model()

    # Adjust model, optimizer, and training configurations
    model = adjust_layers(model)
    optimizer = adjust_optimizer(learning_rate=0.0005)
    epochs, batch_size = adjust_training_config(epochs=20, batch_size=32)

    # Compile the model
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    train_model(model, train_images, train_labels, epochs, batch_size)

    # Save the adjusted model
    model.save('models/my_model/saved_model.pb')
