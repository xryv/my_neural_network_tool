# File: scripts/retrain.py

import tensorflow as tf
from scripts.model import get_model
from scripts.data_loading import load_data, preprocess_data, data_augmentation

def retrain_model(model, train_images, train_labels, epochs, learning_rate):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=epochs)

def fine_tune_model(model, train_images, train_labels, epochs, learning_rate):
    model.trainable = True  # Ensure the model is trainable
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=epochs)

if __name__ == "__main__":
    # Load and preprocess data
    train_data, _ = load_data('data/train_data.csv')
    train_images, train_labels, _, _ = preprocess_data(train_data, train_data)  # No test data for retraining

    # Get existing model or a new model
    try:
        model = tf.keras.models.load_model('models/my_model/saved_model.pb')
    except:
        model = get_model()

    # Retrain or fine-tune the model
    choice = input("Enter 'retrain' to retrain the model or 'fine-tune' to fine-tune the model: ").lower()
    epochs = int(input("Enter the number of epochs: "))
    learning_rate = float(input("Enter the learning rate: "))

    if choice == 'retrain':
        retrain_model(model, train_images, train_labels, epochs, learning_rate)
    elif choice == 'fine-tune':
        fine_tune_model(model, train_images, train_labels, epochs, learning_rate)
    else:
        print("Invalid choice. Exiting.")

    # Save the retrained/fine-tuned model
    model.save('models/my_model/saved_model.pb')
