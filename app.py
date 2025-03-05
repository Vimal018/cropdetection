import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image  # type: ignore
from io import BytesIO
from PIL import Image
import gdown # type: ignore

# Disable oneDNN warning before importing TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Google Drive Model Download Configuration
MODEL_ID = "1_qSNaUeNmle2DpCBqrEIWdsF3RBmAt6k"  # Extracted from your Google Drive link
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"
MODEL_PATH = "crop_disease_model.h5"

# Download the model if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)

# Load the trained TensorFlow model with error handling
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Handle model loading failure

# Define class labels
CLASS_LABELS = {
    0: "American Bollworm on Cotton", 1: "Anthracnose on Cotton", 2: "Army worm",
    3: "Bacterial Blight in Rice", 4: "Brownspot", 5: "Common_Rust", 6: "Cotton Aphid",
    7: "Flag Smut", 8: "Gray_Leaf_Spot", 9: "Healthy Maize", 10: "Healthy Wheat",
    11: "Healthy Cotton", 12: "Leaf Curl", 13: "Leaf Smut", 14: "Mosaic Sugarcane",
    15: "RedRot Sugarcane", 16: "RedRust Sugarcane", 17: "Rice Blast", 18: "Sugarcane Healthy",
    19: "Tungro", 20: "Wheat Brown Leaf Rust", 21: "Wheat Stem Fly", 22: "Wheat Aphid",
    23: "Wheat Black Rust", 24: "Wheat Leaf Blight", 25: "Wheat Mite", 26: "Wheat Powdery Mildew",
    27: "Wheat Scab", 28: "Wheat Yellow Rust", 29: "Wilt", 30: "Yellow Rust Sugarcane",
    31: "Bacterial Blight in Cotton", 32: "Boll Rot on Cotton", 33: "Bollworm on Cotton",
    34: "Cotton Mealy Bug", 35: "Cotton Whitefly", 36: "Maize Ear Rot", 37: "Maize Fall Armyworm",
    38: "Maize Stem Borer", 39: "Pink Bollworm in Cotton", 40: "Red Cotton Bug", 41: "Thrips on Cotton"
}

@app.get("/")
def home():
    return {"message": "FastAPI Server is Running!"}

def preprocess_image(image_file):
    """Preprocess the uploaded image for model prediction."""
    try:
        img = Image.open(BytesIO(image_file)).convert("RGB")  # Ensure RGB format
        img = img.resize((224, 224))  # Resize to model input size
        img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing error: {e}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Handle image uploads and return crop disease predictions."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    MAX_FILE_SIZE_MB = 5  # Set max file size limit (MB)
    contents = await file.read()

    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File size exceeds 5MB limit")

    try:
        img_array = preprocess_image(contents)  # Preprocess image
        prediction = model.predict(img_array)  # Get model prediction

        if prediction is None or len(prediction) == 0:
            raise HTTPException(status_code=500, detail="Model did not return a valid prediction")

        predicted_index = int(np.argmax(prediction))  # Get highest probability index
        predicted_label = CLASS_LABELS.get(predicted_index, "Unknown")  # Handle missing label
        confidence = float(np.max(prediction))  # Get highest probability

        return {
            "status": "success",
            "data": {
                "prediction": predicted_label,
                "confidence": confidence,
                "all_confidences": prediction.tolist()
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
