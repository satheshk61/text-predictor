from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import joblib

app = Flask(__name__)

# -------------------
# Paths
# -------------------
MODEL_PATH = "model/resnet50_text_predictor.pth"
ENCODER_PATH = "model/label_encoder.pkl"

# -------------------
# 1️⃣ Load Label Encoder
# -------------------
try:
    le = joblib.load(ENCODER_PATH)
except Exception as e:
    print("⚠️ Label encoder load error:", e)
    le = None

# -------------------
# 2️⃣ Define Model Architecture (match training fc layer)
# -------------------
def get_model(num_classes):
    model = models.resnet50(weights=None)  # weights=None replaces deprecated pretrained=False
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)
    )
    return model

# -------------------
# 3️⃣ Load Model Weights
# -------------------
try:
    if le is not None:
        model = get_model(num_classes=len(le.classes_))
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        model.eval()
    else:
        model = None
except Exception as e:
    print("⚠️ Model load error:", e)
    model = None

# -------------------
# 4️⃣ Image Transform
# -------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------
# 5️⃣ Routes
# -------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        img = Image.open(file).convert("RGB")
        tensor = transform(img).unsqueeze(0)  # Shape: (1, 3, 224, 224)

        if model and le:
            with torch.no_grad():
                output = model(tensor)
                pred_idx = output.argmax(dim=1).item()
                predicted_label = le.inverse_transform([pred_idx])[0]
        else:
            predicted_label = "demo_prediction"

        return jsonify({'prediction': predicted_label})
    
    except Exception as e:
        print("❌ Prediction error:", e)
        return jsonify({'error': str(e)}), 500

# -------------------
# 6️⃣ Run App
# -------------------
if __name__ == '__main__':
    app.run(debug=True)
