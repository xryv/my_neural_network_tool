import google.auth
from google.cloud import aiplatform

# Define your project and location
project_id = "531679258927"  # Use your actual project ID
location = "europe"  # Use the appropriate multi-region location in the EU, e.g., "europe-west4"

# Initialize Vertex AI
aiplatform.init(project=project_id, location=location)

# Define and train your neural network model
def create_and_train_model():
    # Define your model architecture and train it as before

# Create and train your model
trained_model = create_and_train_model()

# Define the display name for your Vertex AI Model
model_display_name = "my-vertex-ai-model"

# Specify the serving container image for Vertex AI
serving_container_image_uri = "gcr.io/cloud-aiplatform/predictor/cpu.2-0:latest"  # Use your actual URI

# Upload the model to Vertex AI
model = aiplatform.Model.upload(
    display_name=model_display_name,
    artifact_uri="gs://private_neural_network/models/my_model",  # Use your actual GCS URI
    serving_container_image_uri=serving_container_image_uri,
)

print(f'Model uploaded to Vertex AI: {model.display_name}')
