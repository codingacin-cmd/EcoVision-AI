import logging
import uuid
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image, UnidentifiedImageError
from flask import Flask, render_template, request, jsonify, url_for
from sklearn.ensemble import RandomForestClassifier
from torchvision import models, transforms

import database


# ==================================================
# 1. FLASK CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
MODEL_CACHE = BASE_DIR / "model_cache"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
MODEL_CACHE.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB

logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
DEVICE = torch.device("cpu")

PLACES_MODEL_URL = (
    "https://places2.csail.mit.edu/models_places365/"
    "resnet18_places365.pth.tar"
)
PLACES_LABELS_URL = (
    "https://raw.githubusercontent.com/CSAILVision/places365/"
    "master/categories_places365.txt"
)

MODEL_PATH = MODEL_CACHE / "resnet18_places365.pth.tar"
LABELS_PATH = MODEL_CACHE / "categories_places365.txt"

scene_model = None
scene_labels = None


# ==================================================
# 2. LAND RISK PREDICTION MODEL
# ==================================================
# Educational demonstration model trained on synthetic examples.
# It is not a scientifically validated environmental assessment.

X_TRAIN = np.array([
    [90, 85, 90, 10],
    [80, 75, 85, 20],
    [75, 80, 75, 15],
    [65, 65, 70, 30],
    [60, 55, 60, 45],
    [55, 60, 55, 50],
    [45, 50, 50, 60],
    [40, 35, 45, 70],
    [30, 40, 35, 80],
    [20, 25, 30, 90],
    [25, 20, 20, 85],
    [15, 30, 25, 95],
], dtype=float)

Y_TRAIN = np.array([
    "Low", "Low", "Low", "Low",
    "Medium", "Medium", "Medium", "Medium",
    "High", "High", "High", "High",
])

risk_model = RandomForestClassifier(n_estimators=100, random_state=42)
risk_model.fit(X_TRAIN, Y_TRAIN)


# ==================================================
# 3. PLACES365 IMAGE MODEL
# ==================================================

def download_file(url, destination):
    """Download a model resource if it is not already cached."""
    if destination.exists() and destination.stat().st_size > 0:
        return

    app.logger.info("Downloading %s", destination.name)
    response = requests.get(url, timeout=(15, 120))
    response.raise_for_status()

    temporary_file = destination.with_suffix(destination.suffix + ".tmp")
    temporary_file.write_bytes(response.content)
    temporary_file.replace(destination)


def load_scene_model():
    """Load the Places365 ResNet18 scene-recognition model."""
    global scene_model, scene_labels

    if scene_model is not None and scene_labels is not None:
        return

    download_file(PLACES_MODEL_URL, MODEL_PATH)
    download_file(PLACES_LABELS_URL, LABELS_PATH)

    # Only load model files downloaded from the trusted URLs above.
    try:
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=False,
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 365)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    labels_by_index = []
    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            try:
                index = int(parts[-1])
                label = " ".join(parts[:-1]).replace("_", " ")
                labels_by_index.append((index, label))
            except ValueError:
                continue

    labels_by_index.sort(key=lambda item: item[0])

    if len(labels_by_index) != 365:
        raise RuntimeError(
            f"Expected 365 scene labels, but found {len(labels_by_index)}. "
            "Check the labels file in model_cache."
        )

    scene_model = model
    scene_labels = [label for _, label in labels_by_index]
    app.logger.info("Places365 model loaded successfully.")


image_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def format_scene_label(label):
    """Convert a Places365 path-like label into a readable name."""
    cleaned = str(label).strip().replace("_", " ")
    parts = [part for part in cleaned.split("/") if part]
    if parts:
        cleaned = parts[-1]
    return cleaned.replace("-", " ").title()


def categorize_scene(label):
    """Group scene labels into broad environmental categories."""
    text = str(label).lower()

    keywords = {
        "Forest": [
            "forest", "woods", "woodland", "rainforest", "bamboo",
        ],
        "Water": [
            "lake", "river", "waterfall", "ocean", "sea", "pond",
            "stream", "beach", "coast", "wetland", "swamp", "marsh",
            "creek", "rainforest",
        ],
        "Mountain": [
            "mountain", "hill", "valley", "cliff", "glacier",
            "volcano", "canyon",
        ],
        "Plants": [
            "garden", "flower", "greenhouse", "meadow", "grassland",
            "orchard", "field", "park",
        ],
        "Desert": ["desert", "sand dune", "badlands"],
        "Urban": [
            "street", "building", "city", "downtown", "industrial",
            "highway", "parking lot",
        ],
    }

    for category, words in keywords.items():
        if any(word in text for word in words):
            return category

    return "Other"


def get_nature_tip(category):
    """Return a conservation tip for a broad scene category."""
    tips = {
        "Forest": (
            "Protect trees, avoid littering, and help conserve forest habitats."
        ),
        "Water": (
            "Keep waterways clean, reduce plastic waste, and conserve water."
        ),
        "Mountain": (
            "Stay on marked trails and avoid disturbing fragile habitats."
        ),
        "Plants": (
            "Protect native plants, conserve water, and care for green spaces."
        ),
        "Desert": (
            "Respect desert plants and wildlife, and avoid disturbing habitats."
        ),
        "Urban": (
            "Support urban trees, reduce waste, and keep public spaces clean."
        ),
        "Other": (
            "Avoid littering and help protect the natural surroundings."
        ),
    }
    return tips.get(category, tips["Other"])


def predict_scene(image_path):
    """Run scene recognition and return the top predictions."""
    load_scene_model()

    with Image.open(image_path) as image:
        image = image.convert("RGB")

    tensor = image_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = scene_model(tensor)
        probabilities = torch.softmax(output[0], dim=0)

    count = min(5, len(scene_labels))
    top_probs, top_indices = torch.topk(probabilities, count)

    top_index = int(top_indices[0].item())
    raw_label = scene_labels[top_index]
    display_label = format_scene_label(raw_label)
    confidence = float(top_probs[0].item()) * 100

    alternatives = []
    for probability, index_tensor in zip(top_probs, top_indices):
        alternative_raw_label = scene_labels[int(index_tensor.item())]
        alternatives.append({
            "label": format_scene_label(alternative_raw_label),
            "confidence": round(float(probability.item()) * 100, 2),
        })

    category = categorize_scene(raw_label)

    if confidence >= 70:
        confidence_message = (
            "The model is relatively confident in this scene prediction."
        )
    elif confidence >= 40:
        confidence_message = "Treat this scene prediction as an estimate."
    else:
        confidence_message = (
            "The model has low confidence. Try a clearer image."
        )

    return {
        "label": display_label,
        "raw_label": raw_label,
        "confidence": round(confidence, 2),
        "nature_category": category,
        "tip": get_nature_tip(category),
        "alternatives": alternatives,
        "confidence_message": confidence_message,
        "note": (
            "Places365 recognizes scene types. It does not scientifically "
            "identify species or measure ecosystem health."
        ),
    }


def allowed_file(filename):
    """Return True when the filename has an allowed image extension."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# ==================================================
# 4. HOME PAGE
# ==================================================

@app.route("/")
def home():
    return render_template("index.html")


# ==================================================
# 5. LAND RISK PREDICTION API
# ==================================================

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            data = request.form.to_dict()

        vegetation = float(data.get("vegetation", 60))
        rainfall = float(data.get("rainfall", 60))
        soil_quality = float(data.get("soil_quality", 60))
        human_activity = float(data.get("human_activity", 40))

        values = [
            vegetation,
            rainfall,
            soil_quality,
            human_activity,
        ]

        if not all(np.isfinite(value) for value in values):
            return jsonify({
                "error": "Please enter valid numeric values.",
            }), 400

        if not all(0 <= value <= 100 for value in values):
            return jsonify({
                "error": "All values must be between 0 and 100.",
            }), 400

        risk = str(risk_model.predict(np.array([values], dtype=float))[0])

        database.save_prediction(
            risk=risk,
            vegetation=vegetation,
            rainfall=rainfall,
            soil_quality=soil_quality,
            human_activity=human_activity,
        )

        recommendations = {
            "Low": (
                "The selected conditions appear relatively favorable. "
                "Continue protecting vegetation, soil, and habitats."
            ),
            "Medium": (
                "Some conditions may need improvement. Protect green spaces "
                "and reduce environmental pressure."
            ),
            "High": (
                "The selected conditions indicate a higher demonstration risk. "
                "Consider protecting vegetation and reducing harmful activities."
            ),
        }

        return jsonify({
            "risk": risk,
            "prediction": risk,
            "message": f"Estimated environmental risk: {risk}",
            "recommendation": recommendations[risk],
            "note": (
                "This educational prototype uses synthetic training data. "
                "It is not a scientifically validated environmental assessment."
            ),
        })

    except (TypeError, ValueError):
        return jsonify({
            "error": "Please provide valid numeric slider values.",
        }), 400
    except Exception as error:
        app.logger.exception("Risk prediction failed")
        return jsonify({
            "error": str(error),
            "error_type": type(error).__name__,
        }), 500


# ==================================================
# 6. ECOBOT CHAT API
# ==================================================

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({
            "reply": "Please enter a question about nature.",
        }), 400

    text = message.lower()
    responses = [
        (
            ["hello", "hi", "hey"],
            "Hello! 🌿 I'm EcoBot. Ask me about forests, water, pollution, "
            "recycling, wildlife, or climate change.",
        ),
        (
            ["forest", "tree", "deforestation"],
            "🌳 Forests provide wildlife habitats, protect soil, and store "
            "carbon. Help protect existing forests and plant suitable native trees.",
        ),
        (
            ["water", "river", "ocean", "rain"],
            "💧 Save water, fix leaks, reduce plastic waste, and keep waterways clean.",
        ),
        (
            ["pollution", "plastic", "waste", "litter"],
            "♻️ Reduce disposable products, reuse items, separate recyclables, "
            "and dispose of waste responsibly.",
        ),
        (
            ["climate", "global warming", "carbon"],
            "🌍 Saving energy, choosing lower-carbon transport, and protecting "
            "ecosystems can help reduce environmental pressure.",
        ),
        (
            ["wildlife", "animal", "biodiversity", "species"],
            "🦋 Protect natural habitats and observe wildlife without disturbing it.",
        ),
        (
            ["plant", "garden", "flower"],
            "🌱 Grow suitable native plants, protect pollinators, and conserve water.",
        ),
        (
            ["recycle", "reuse", "reduce"],
            "♻️ Reduce consumption first, reuse what you can, and recycle according "
            "to your local recycling rules.",
        ),
        (
            ["education", "sdg", "quality education"],
            "📚 SDG 4 promotes inclusive and equitable quality education and "
            "lifelong learning opportunities for everyone.",
        ),
        (
            ["thank", "thanks"],
            "You're welcome! 🌿 Every responsible action can help protect nature.",
        ),
    ]

    reply = (
        "I'm EcoBot. Try asking about forests, water conservation, pollution, "
        "recycling, wildlife, climate change, or quality education."
    )

    for words, answer in responses:
        if any(word in text for word in words):
            reply = answer
            break

    return jsonify({"reply": reply})


# ==================================================
# 7. NATURE IMAGE SCANNER API
# ==================================================

@app.route("/scan", methods=["POST"])
def scan():
    if "image" not in request.files:
        return jsonify({
            "error": "The image field is missing. Please select an image.",
        }), 400

    uploaded_file = request.files["image"]

    if not uploaded_file or not uploaded_file.filename:
        return jsonify({
            "error": "Please choose an image file.",
        }), 400

    if not allowed_file(uploaded_file.filename):
        return jsonify({
            "error": "Use a JPG, JPEG, PNG, or WEBP image.",
        }), 400

    extension = uploaded_file.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{extension}"
    image_path = UPLOAD_FOLDER / filename

    try:
        uploaded_file.save(image_path)

        # Verify the uploaded file before inference.
        with Image.open(image_path) as image:
            image.verify()

        result = predict_scene(str(image_path))

        image_url = url_for(
            "static",
            filename=f"uploads/{filename}",
        )

        database.save_scan(
            label=result["label"],
            nature_category=result["nature_category"],
            confidence=result["confidence"],
            image_url=image_url,
        )

        result["image_url"] = image_url
        return jsonify(result)

    except (UnidentifiedImageError, OSError):
        image_path.unlink(missing_ok=True)
        return jsonify({
            "error": "The file is not a valid or readable image.",
        }), 400
    except Exception as error:
        app.logger.exception("Image scanning failed")
        return jsonify({
            "error": (
                "Image scanning failed. Check the PyCharm terminal "
                "for details."
            ),
            "error_type": type(error).__name__,
        }), 500


# ==================================================
# 8. DASHBOARD STATISTICS API
# ==================================================

@app.route("/dashboard-stats", methods=["GET"])
def dashboard_stats():
    try:
        return jsonify(database.get_dashboard_stats())
    except Exception:
        app.logger.exception("Dashboard statistics failed")
        return jsonify({
            "error": "Unable to load dashboard statistics.",
        }), 500


# ==================================================
# 9. SAVE ECO-FRIENDLY ACTION
# ==================================================

@app.route("/dashboard-eco-action", methods=["POST"])
def dashboard_eco_action():
    try:
        data = request.get_json(silent=True) or {}
        action = str(
            data.get("action", "Completed an eco-friendly action")
        ).strip()

        if not action:
            action = "Completed an eco-friendly action"

        if len(action) > 250:
            return jsonify({
                "error": "The action description must be under 250 characters.",
            }), 400

        database.save_eco_action(action)

        return jsonify({
            "success": True,
            "message": "Your eco-friendly action has been recorded!",
            "stats": database.get_dashboard_stats(),
        })
    except Exception:
        app.logger.exception("Saving eco action failed")
        return jsonify({
            "error": "Unable to save your eco-friendly action.",
        }), 500


# ==================================================
# 10. ERROR HANDLERS
# ==================================================

@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "error": "Image too large. Please choose an image under 5 MB.",
    }), 413


# ==================================================
# 11. START APPLICATION
# ==================================================

if __name__ == "__main__":
    database.init_db()
    app.logger.info("Starting EcoVision AI...")
    app.run(debug=True)
