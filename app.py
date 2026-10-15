
import logging
import os
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from flask import Flask, render_template, request, jsonify, url_for
from sklearn.ensemble import RandomForestClassifier

import database


# --------------------------------------------------
# APP CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}


# --------------------------------------------------
# LAND RISK PREDICTION MODEL
# Demonstration model using illustrative training data
# --------------------------------------------------

X_TRAIN = np.array([
    [85, 80, 90, 15],
    [75, 70, 80, 25],
    [65, 60, 70, 35],
    [55, 50, 60, 45],
    [45, 40, 50, 55],
    [35, 30, 40, 65],
    [25, 20, 30, 75],
    [15, 15, 20, 90],
    [90, 40, 75, 20],
    [30, 80, 45, 70],
    [70, 25, 55, 60],
    [40, 60, 35, 80],
    [80, 85, 85, 10],
    [20, 25, 25, 85],
    [60, 75, 65, 30],
    [50, 45, 45, 60],
], dtype=float)

Y_TRAIN = np.array([
    "Low", "Low", "Low", "Moderate",
    "Moderate", "High", "High", "High",
    "Low", "High", "High", "High",
    "Low", "High", "Low", "Moderate",
])

risk_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
risk_model.fit(X_TRAIN, Y_TRAIN)


# --------------------------------------------------
# GENERAL HELPERS
# --------------------------------------------------

def allowed_file(filename):
    """Check whether the uploaded file has an allowed extension."""
    return (
        bool(filename)
        and "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def format_scene_label(label):
    """Convert a model label into a readable name."""
    cleaned = str(label).strip().replace("_", " ")
    parts = [part for part in cleaned.split("/") if part]

    if parts:
        cleaned = parts[-1]

    return cleaned.replace("-", " ").title()


def categorize_scene(label):
    """Group image labels into broad environmental categories."""
    text = str(label).lower()

    keywords = {
        "Forest": [
            "forest", "woods", "woodland", "rainforest",
            "bamboo", "tree"
        ],
        "Water": [
            "lake", "river", "waterfall", "ocean", "sea",
            "pond", "stream", "beach", "coast", "wetland",
            "swamp", "marsh", "creek"
        ],
        "Mountain": [
            "mountain", "hill", "valley", "cliff",
            "glacier", "volcano", "canyon"
        ],
        "Plants": [
            "garden", "flower", "greenhouse", "meadow",
            "grassland", "orchard", "field", "park",
            "plant", "vegetation"
        ],
        "Desert": ["desert", "sand dune", "badlands"],
        "Wildlife": [
            "bird", "animal", "insect", "butterfly",
            "tiger", "elephant", "wildlife", "mammal",
            "reptile"
        ],
        "Urban": [
            "street", "building", "city", "downtown",
            "industrial", "highway", "parking lot"
        ],
    }

    for category, words in keywords.items():
        if any(word in text for word in words):
            return category

    return "Other"


def get_nature_tip(category):
    """Return an educational conservation tip."""
    tips = {
        "Forest": (
            "Protect trees, avoid littering, and conserve "
            "forest habitats."
        ),
        "Water": (
            "Keep waterways clean, reduce plastic waste, "
            "and conserve water."
        ),
        "Mountain": (
            "Stay on marked trails and avoid disturbing "
            "fragile habitats."
        ),
        "Plants": (
            "Protect native plants, conserve water, "
            "and care for green spaces."
        ),
        "Desert": (
            "Respect desert plants and wildlife, "
            "and protect their habitats."
        ),
        "Wildlife": (
            "Observe wildlife without disturbing it "
            "and protect natural habitats."
        ),
        "Urban": (
            "Support urban trees, reduce waste, "
            "and keep public spaces clean."
        ),
        "Other": (
            "Avoid littering and help protect "
            "the natural surroundings."
        ),
    }

    return tips.get(category, tips["Other"])


# --------------------------------------------------
# AI NATURE SCANNER
# Uses Hugging Face Inference API instead of local
# PyTorch models to reduce deployment memory usage.
# --------------------------------------------------


# Local AI model — loaded only when the scanner is first used
scene_classifier = None


def predict_scene(image_path):
    """Classify an image locally using a pretrained AI model."""
    global scene_classifier

    try:
        # Load the model once, not for every uploaded image.
        if scene_classifier is None:
            import torch
            from transformers import pipeline

            torch.set_num_threads(2)

            scene_classifier = pipeline(
                task="image-classification",
                model="google/vit-base-patch16-224",
                device=-1
            )

        # Run inference on the uploaded image.
        predictions = scene_classifier(
            image_path,
            top_k=5
        )

        if not predictions:
            raise RuntimeError("The model returned no predictions.")

        top = predictions[0]
        raw_label = top["label"]
        confidence = float(top["score"]) * 100

        display_label = format_scene_label(raw_label)
        category = categorize_scene(raw_label)

        alternatives = [
            {
                "label": format_scene_label(item["label"]),
                "confidence": round(float(item["score"]) * 100, 2)
            }
            for item in predictions
        ]

        if confidence >= 70:
            confidence_message = (
                "The model assigns a high score to this image category."
            )
        elif confidence >= 40:
            confidence_message = (
                "Treat this image classification as an estimate."
            )
        else:
            confidence_message = (
                "The model score is low. Try a clearer image."
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
                "This local AI model classifies general image categories. "
                "It does not scientifically measure ecosystem health "
                "or reliably identify every species."
            )
        }

    except Exception:
        app.logger.exception("Local image classification failed.")
        raise



# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------------------------
# LAND RISK PREDICTOR
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True) or request.form

        vegetation = float(data.get("vegetation", 50))
        rainfall = float(data.get("rainfall", 50))
        soil_quality = float(data.get("soil_quality", 50))
        human_activity = float(data.get("human_activity", 50))

        values = [
            vegetation,
            rainfall,
            soil_quality,
            human_activity
        ]

        if not all(0 <= value <= 100 for value in values):
            return jsonify({
                "success": False,
                "error": "All input values must be between 0 and 100."
            }), 400

        input_data = np.array([values], dtype=float)

        prediction = str(risk_model.predict(input_data)[0])
        probabilities = risk_model.predict_proba(input_data)[0]
        confidence = round(float(max(probabilities)) * 100, 2)

        recommendations = {
            "Low": (
                "Continue protecting vegetation, conserving water, "
                "and monitoring the area."
            ),
            "Moderate": (
                "Monitor the area, protect existing vegetation, "
                "and improve soil and water management."
            ),
            "High": (
                "Prioritize habitat protection, reduce damaging "
                "human activities, and seek an expert assessment."
            ),
        }

        messages = {
            "Low": "The model estimates a relatively low land-risk level.",
            "Moderate": "The model estimates a moderate land-risk level.",
            "High": "The model estimates a high land-risk level.",
        }

        result = {
            "success": True,
            "risk": prediction,
            "prediction": prediction,
            "confidence": confidence,
            "message": messages.get(prediction, ""),
            "recommendation": recommendations.get(prediction, ""),
            "note": (
                "Educational demonstration only. This model uses "
                "illustrative training data and is not scientifically "
                "validated for real-world land assessment."
            ),
        }

        # Save the prediction when the database supports it.
        try:
            database.save_prediction(
                risk=prediction,
                vegetation=vegetation,
                rainfall=rainfall,
                soil_quality=soil_quality,
                human_activity=human_activity
            )
        except Exception:
            app.logger.exception(
                "Could not save the land-risk prediction."
            )

        return jsonify(result)

    except (TypeError, ValueError):
        return jsonify({
            "success": False,
            "error": "Please provide valid numeric values."
        }), 400

    except Exception:
        app.logger.exception("Land-risk prediction failed.")
        return jsonify({
            "success": False,
            "error": "Prediction failed. Please try again."
        }), 500


# --------------------------------------------------
# ECOBOT
# Rule-based environmental chatbot
# --------------------------------------------------

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or request.form
    message = str(data.get("message", "")).strip().lower()

    if not message:
        return jsonify({
            "success": False,
            "response": "Please enter a question about the environment."
        }), 400

    responses = [
        (
            ["hello", "hi", "hey"],
            "Hello! I am EcoBot. Ask me about forests, wildlife, "
            "pollution, climate change, or sustainability."
        ),
        (
            ["forest", "deforestation", "trees"],
            "Forests support biodiversity, store carbon, and help "
            "regulate the water cycle. Protect trees and avoid "
            "wasting paper."
        ),
        (
            ["wildlife", "animal", "animals", "biodiversity"],
            "Protect wildlife by preserving natural habitats, "
            "avoiding litter, and never disturbing wild animals."
        ),
        (
            ["pollution", "plastic", "waste", "litter"],
            "Reduce pollution by reusing products, sorting waste, "
            "avoiding single-use plastics, and disposing of rubbish "
            "responsibly."
        ),
        (
            ["climate", "global warming", "carbon"],
            "Climate change is driven mainly by greenhouse gas "
            "emissions. Saving energy, using cleaner transport, "
            "and protecting forests can help reduce emissions."
        ),
        (
            ["water", "river", "ocean", "lake"],
            "Save water, avoid dumping waste into waterways, and "
            "reduce plastic use to help protect aquatic ecosystems."
        ),
        (
            ["plant", "garden", "vegetation"],
            "Grow suitable native plants, protect existing trees, "
            "and avoid harmful chemicals that can damage local "
            "ecosystems."
        ),
        (
            ["sdg 15", "life on land", "sustainable development goal"],
            "UN Sustainable Development Goal 15 focuses on "
            "protecting terrestrial ecosystems, managing forests "
            "sustainably, combating desertification, and halting "
            "biodiversity loss."
        ),
        (
            ["energy", "electricity", "solar"],
            "Switch off unused lights and devices, improve energy "
            "efficiency, and consider renewable energy where possible."
        ),
        (
            ["thank"],
            "You're welcome! Small, consistent actions can help "
            "protect our planet."
        ),
    ]

    response = (
        "I can help with environmental topics such as forests, "
        "wildlife, pollution, water conservation, climate change, "
        "and SDG 15. Try asking me about one of these topics."
    )

    for keywords, answer in responses:
        if any(keyword in message for keyword in keywords):
            response = answer
            break

    return jsonify({
        "success": True,
        "response": response,
        "reply": response,
        "note": "EcoBot uses predefined keyword-based responses."
    })


# --------------------------------------------------
# NATURE SCANNER
# --------------------------------------------------

@app.route("/scan", methods=["POST"])
def scan():
    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "Please select an image to scan."
        }), 400

    uploaded_file = request.files["image"]

    if not uploaded_file or not uploaded_file.filename:
        return jsonify({
            "success": False,
            "error": "Please select an image to scan."
        }), 400

    if not allowed_file(uploaded_file.filename):
        return jsonify({
            "success": False,
            "error": "Unsupported image format. Use JPG, JPEG, PNG, or WEBP."
        }), 400

    unique_name = f"{uuid.uuid4().hex}.jpg"
    image_path = UPLOAD_FOLDER / unique_name

    try:
        # Decode and validate the uploaded image, then save it as JPEG.
        with Image.open(uploaded_file.stream) as image:
            image.verify()

        uploaded_file.stream.seek(0)

        with Image.open(uploaded_file.stream) as image:
            image = image.convert("RGB")
            image.thumbnail((1600, 1600))
            image.save(image_path, format="JPEG", quality=90)

        result = predict_scene(str(image_path))

        result["success"] = True
        result["image_url"] = url_for(
            "static",
            filename=f"uploads/{unique_name}"
        )

        # Store the result if the database operation is available.
        try:
            database.save_scan(
                result["label"],
                result["nature_category"],
                result["confidence"],
                result["image_url"]
            )
        except Exception:
            app.logger.exception(
                "Could not save the nature-scan record."
            )

        return jsonify(result)

    except UnidentifiedImageError:
        if image_path.exists():
            image_path.unlink(missing_ok=True)

        return jsonify({
            "success": False,
            "error": "This file is not a valid image. Please upload another."
        }), 400

    except Exception as error:
        app.logger.exception("Nature scanner failed.")

        # Remove the saved file if inference failed.
        if image_path.exists():
            image_path.unlink(missing_ok=True)

        return jsonify({
            "success": False,
            "error": (
                "The AI scanner could not process this image. "
                "Check the server logs and Hugging Face model availability."
            )
        }), 502


# --------------------------------------------------
# DASHBOARD STATISTICS
# --------------------------------------------------

@app.route("/dashboard-stats", methods=["GET"])
def dashboard_stats():
    try:
        stats = database.get_dashboard_stats()
        return jsonify({
            "success": True,
            **stats
        })

    except Exception:
        app.logger.exception(
            "Dashboard statistics could not be loaded."
        )
        return jsonify({
            "success": False,
            "error": "Could not load dashboard statistics."
        }), 500


# --------------------------------------------------
# RECORD AN ECO ACTION
# --------------------------------------------------

@app.route("/dashboard-eco-action", methods=["POST"])
def dashboard_eco_action():
    data = request.get_json(silent=True) or request.form
    action = str(data.get("action", "")).strip()

    if not action:
        return jsonify({
            "success": False,
            "error": "Please provide an eco-friendly action."
        }), 400

    if len(action) > 200:
        return jsonify({
            "success": False,
            "error": "The action must be 200 characters or fewer."
        }), 400

    try:
        database.save_eco_action(action)

        return jsonify({
            "success": True,
            "message": "Your eco-friendly action has been recorded.",
            "action": action
        })

    except Exception:
        app.logger.exception("Could not save the eco action.")
        return jsonify({
            "success": False,
            "error": "Could not record the action. Please try again."
        }), 500


# --------------------------------------------------
# HANDLE FILES THAT EXCEED THE UPLOAD LIMIT
# --------------------------------------------------

@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "success": False,
        "error": "Image is too large. Please upload an image under 5 MB."
    }), 413


# --------------------------------------------------
# INITIALIZE DATABASE AND START SERVER
# --------------------------------------------------

try:
    database.init_db()
except Exception:
    app.logger.exception(
        "Database initialization failed. Check database.py configuration."
    )


if __name__ == "__main__":
    app.run(debug=True)
