# File: scripts/test.py

import tensorflow as tf
from scripts.data_loading import load_data, preprocess_data
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, test_images, test_labels):
    loss, accuracy = model.evaluate(test_images, test_labels, verbose=0)
    print(f"Loss: {loss}, Accuracy: {accuracy}")

    # Predictions
    predictions = model.predict(test_images)
    predicted_labels = tf.argmax(predictions, axis=1)

    # Classification Report
    print("Classification Report:\n", classification_report(test_labels, predicted_labels))

    # Confusion Matrix
    conf_matrix = confusion_matrix(test_labels, predicted_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d")
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()

if __name__ == "__main__":
    # Load and preprocess data
    _, test_data = load_data('data/test_data.csv')
    _, _, test_images, test_labels = preprocess_data(test_data, test_data)  # No training data for testing

    # Load the model
    model = tf.keras.models.load_model('models/my_model/saved_model.pb')

    # Evaluate the model
    evaluate_model(model, test_images, test_labels)
