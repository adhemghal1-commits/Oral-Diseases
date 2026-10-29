"""
Flask Deployment App for Oral Disease Classification
=====================================================
Provides a modern web interface & REST API to upload oral images
and classify them into 6 oral disease categories using the best-performing
deep learning model.
"""

import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image

# Suppress TF oneDNN and verbose logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import keras

# ── Paths & Configuration ─────────────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent
MODEL_DIR = BASE_DIR / "models"
UPLOAD_FOLDER = APP_DIR / "static" / "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp", "gif"}
IMG_SIZE = (160, 160)

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# ── Load Model and Metadata ───────────────────────────────────────────────────
MODEL_PATH = MODEL_DIR / "best_model.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

print(f"Loading best deployment model from: {MODEL_PATH}")
model = None
if MODEL_PATH.exists():
    try:
        model = keras.models.load_model(MODEL_PATH)
        print("✓ Best model loaded successfully into memory.")
    except Exception as e:
        print(f"⚠ Warning: Could not load model: {e}")
else:
    print(f"⚠ Model file not found at {MODEL_PATH}. App will load in demo mode until training finishes.")

if CLASS_NAMES_PATH.exists():
    with open(CLASS_NAMES_PATH, "r") as f:
        CLASS_NAMES = json.load(f)
else:
    CLASS_NAMES = ["Calculus", "Caries", "Gingivitis", "Hypodontia", "Mouth_Ulcer", "Tooth_Discoloration"]

print(f"Active classes: {CLASS_NAMES}")

# ── Clinical Knowledge Base ───────────────────────────────────────────────────
DISEASE_INFO = {
    "Calculus": {
        "title": "Dental Calculus (Tartar)",
        "description": "Hardened dental plaque caused by precipitation of salivary minerals onto existing biofilm. Often forms along the gingival margin.",
        "severity": "Moderate",
        "badge_color": "#f59e0b",
        "icon": "fa-teeth",
        "symptoms": ["Rough, yellowish or brown deposits on teeth", "Persistent bad breath (halitosis)", "Gum irritation"],
        "recommendation": "Requires professional dental scaling and polishing. Daily toothbrushing and interdental flossing prevent recurrence.",
    },
    "Caries": {
        "title": "Dental Caries (Cavities / Decay)",
        "description": "Demineralization and structural breakdown of tooth enamel and dentin caused by acids produced from bacterial fermentation of dietary carbohydrates.",
        "severity": "High",
        "badge_color": "#ef4444",
        "icon": "fa-tooth",
        "symptoms": ["Tooth sensitivity to sweet, hot, or cold foods", "Visible dark pits or holes", "Mild to sharp localized tooth pain"],
        "recommendation": "Immediate clinical evaluation is required. Treatments include composite fillings, crowns, or root canal therapy depending on lesion depth.",
    },
    "Gingivitis": {
        "title": "Gingivitis (Early Gum Disease)",
        "description": "Reversible, localized inflammation of the marginal gingiva without loss of underlying connective tissue attachment or alveolar bone.",
        "severity": "Moderate",
        "badge_color": "#f97316",
        "icon": "fa-teeth-open",
        "symptoms": ["Red, swollen, or puffy gums", "Bleeding upon brushing or flossing", "Tender gums"],
        "recommendation": "Enhanced oral hygiene with soft-bristled brush, chlorhexidine or antibacterial rinse, and professional prophylaxis.",
    },
    "Hypodontia": {
        "title": "Hypodontia (Congenital Tooth Absence)",
        "description": "A developmental dental anomaly characterized by the congenital absence of one to five permanent teeth (excluding third molars).",
        "severity": "Specialized",
        "badge_color": "#8b5cf6",
        "icon": "fa-ban",
        "symptoms": ["Persistent gaps where permanent teeth should erupt", "Retained primary (baby) teeth", "Malocclusion"],
        "recommendation": "Interdisciplinary orthodontic and prosthodontic consultation. Options include space closure, resin-bonded bridges, or dental implants.",
    },
    "Mouth_Ulcer": {
        "title": "Mouth Ulcer (Aphthous Stomatitis)",
        "description": "Painful round or ovoid mucosal lesions with a grayish-white base and erythematous inflamed borders.",
        "severity": "Low to Moderate",
        "badge_color": "#ec4899",
        "icon": "fa-circle-dot",
        "symptoms": ["Burning or tingling before lesion appearance", "Painful shallow ulceration", "Discomfort when eating acidic/salty foods"],
        "recommendation": "Topical anesthetics or corticosteroid gels. Avoid abrasive or acidic foods. Seek biopsy/consultation if lesions persist beyond 14 days.",
    },
    "Tooth_Discoloration": {
        "title": "Tooth Discoloration / Staining",
        "description": "Abnormal staining or alteration of tooth coloration, classified as extrinsic (chromogenic foods, tobacco) or intrinsic (fluorosis, trauma).",
        "severity": "Low",
        "badge_color": "#64748b",
        "icon": "fa-palette",
        "symptoms": ["Yellow, brown, gray, or mottled enamel spots", "Loss of natural translucency"],
        "recommendation": "Professional in-office whitening, microabrasion, porcelain veneers, or reducing chromogenic foods and beverages.",
    },
}

# ── Flask Application ─────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(filepath):
    """Loads and preprocesses an image using PIL for maximum compatibility."""
    img = Image.open(filepath).convert("RGB")
    img = img.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return arr


def run_inference(filepath):
    """Performs model inference and returns formatted probability distribution."""
    global model
    if model is None and MODEL_PATH.exists():
        model = keras.models.load_model(MODEL_PATH)

    if model is None:
        # Fallback dummy distribution for development mode
        predicted_idx = 0
        conf = 95.0
        all_probs = {c: (95.0 if i == 0 else 1.0) for i, c in enumerate(CLASS_NAMES)}
        return CLASS_NAMES[0], conf, all_probs

    arr = preprocess_image(filepath)
    preds = model.predict(arr, verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(preds[pred_idx]) * 100.0

    all_probs = {
        CLASS_NAMES[i]: round(float(preds[i]) * 100.0, 2)
        for i in range(len(CLASS_NAMES))
    }
    sorted_probs = dict(sorted(all_probs.items(), key=lambda item: item[1], reverse=True))

    return pred_class, confidence, sorted_probs


# ── Web Routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("index.html", error="No file part in upload request.")

    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", error="No file selected for classification.")

    if file and allowed_file(file.filename):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = secure_filename(file.filename)
        saved_filename = f"{timestamp}_{safe_name}"
        saved_path = UPLOAD_FOLDER / saved_filename
        file.save(saved_path)

        predicted_class, confidence, all_predictions = run_inference(saved_path)
        disease_info = DISEASE_INFO.get(predicted_class, {})

        return render_template(
            "result.html",
            predicted_class=predicted_class.replace("_", " "),
            raw_class=predicted_class,
            confidence=round(confidence, 2),
            all_predictions=all_predictions,
            disease_info=disease_info,
            image_path=f"uploads/{saved_filename}",
        )

    return render_template(
        "index.html",
        error="Unsupported file format. Please upload JPG, PNG, WEBP, or BMP images.",
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """REST API endpoint for programmatic integration."""
    if "file" not in request.files:
        return jsonify({"error": "No file part provided."}), 400

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file or extension."}), 400

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = secure_filename(file.filename)
    saved_filename = f"{timestamp}_{safe_name}"
    saved_path = UPLOAD_FOLDER / saved_filename
    file.save(saved_path)

    predicted_class, confidence, all_predictions = run_inference(saved_path)
    info = DISEASE_INFO.get(predicted_class, {})

    return jsonify({
        "status": "success",
        "predicted_class": predicted_class,
        "confidence_percentage": round(confidence, 2),
        "class_probabilities": all_predictions,
        "clinical_details": {
            "title": info.get("title"),
            "description": info.get("description"),
            "severity": info.get("severity"),
            "recommendation": info.get("recommendation"),
            "symptoms": info.get("symptoms", []),
        },
    })


@app.route("/health")
def health():
    return jsonify({
        "status": "online",
        "model_loaded": model is not None,
        "num_classes": len(CLASS_NAMES),
        "classes": CLASS_NAMES,
    })


if __name__ == "__main__":
    print("\n🚀 Starting Oral Disease Flask Classifier Web Service on http://127.0.0.1:5000 ...")
    app.run(host="0.0.0.0", port=5000, debug=False)
