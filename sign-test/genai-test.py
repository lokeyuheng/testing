import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

google_cloud_project = os.getenv("GOOGLE_CLOUD_PROJECT")
google_cloud_location = os.getenv("GOOGLE_CLOUD_LOCATION")

client = genai.Client(
    vertexai=True,
    project=google_cloud_project,
    location=google_cloud_location,
)
model_name = "gemini-2.0-flash-exp"

# Generate content
response = client.models.generate_content(
    model=model_name,
    contents="How does AI work?",
)
print(response.text)