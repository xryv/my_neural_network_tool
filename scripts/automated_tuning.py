# File: scripts/automated_tuning.py

import tensorflow as tf
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from scripts.model import get_model
from scripts.data_loading import load_data, preprocess_data

def objective(config):
    # Load and preprocess data
    train_data, test_data = load_data('data/train_data.csv')
    train_images, train_labels, test_images, test_labels = preprocess_data(train_data, test_data)

    # Get the model
    model = get_model()

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config["lr"],
        beta_1=config["beta_1"],
        beta_2=config["beta_2"]
    )
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=10)

    # Evaluate the model
    loss, accuracy = model.evaluate(test_images, test_labels)
    tune.report(loss=loss, accuracy=accuracy)

if __name__ == "__main__":
    search_space = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "beta_1": tune.uniform(0.7, 0.99),
        "beta_2": tune.uniform(0.8, 0.9999)
    }

    analysis = tune.run(
        objective,
        config=search_space,
        scheduler=ASHAScheduler(metric="loss", mode="min"),
        num_samples=10,
        resources_per_trial={"cpu": 2, "gpu": 1}
    )

    print("Best config: ", analysis.get_best_config(metric="loss", mode="min"))
