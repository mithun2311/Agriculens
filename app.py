import os
import datetime
from datetime import timedelta
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import onnxruntime as rt
import cv2
import requests
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from bson.objectid import ObjectId
from flask_bcrypt import Bcrypt
import scipy.special
from PIL import Image
import google.generativeai as genai

# --- JWT Imports ---
from flask_jwt_extended import create_access_token, get_jwt_identity, jwt_required, JWTManager

# --- CORS Import ---
from flask_cors import CORS

# --- HF Hub Import ---
from huggingface_hub import HfApi

app = Flask(__name__)
CORS(app)  # Enable CORS for your Vercel frontend
bcrypt = Bcrypt(app)

# =======================
# CONFIGURATIONS
# =======================

# --- JWT CONFIG ---
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY")
jwt = JWTManager(app)
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=1) 
jwt = JWTManager(app)

# --- Model paths ---
CNN_MODEL_PATH = "custom_cnn_final.keras"
VIT_MODEL_PATH = "plant_vit.onnx"

# --- Classes list ---
CLASS_NAMES = [
    'Apple__Apple_scab', 'Apple__Black_rot', 'Apple__Cedar_apple_rust', 'Apple__healthy',
    'Blueberry__healthy', 'Cherry(including_sour)__Powdery_mildew', 'Cherry(including_sour)__healthy',
    'Corn_(maize)__Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)__Common_rust', 'Corn_(maize)__Northern_Leaf_Blight',
    'Corn_(maize)__healthy', 'Grape__Black_rot', 'Grape__Esca(Black_Measles)', 'Grape__Leaf_bht(Isariopsis_Leaf_Spot)',
    'Grape__healthy', 'Orange__Haunglongbing(Citrus_greening)', 'Peach__Bacterial_spot', 'Peach__healthy',
    'Pepper,bell__Bacterial_spot', 'Pepper,bell__healthy', 'Potato__Early_blight', 'Potato__Late_blight',
    'Potato__healthy', 'Raspberry__healthy', 'Soybean__healthy', 'Squash__Powdery_mildew',
    'Strawberry__Leaf_scorch', 'Strawberry__healthy', 'Tomato__Bacterial_spot', 'Tomato__Early_blight',
    'Tomato__Late_blight', 'Tomato__Leaf_Mold', 'Tomato__Septoria_leaf_spot',
    'Tomato__Spider_mites Two-spotted_spider_mite', 'Tomato__Target_Spot',
    'Tomato__Tomato_Yellow_Leaf_Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato__healthy'
]

# --- API Keys & DB ---
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

# --- File Uploads ---
UPLOAD_FOLDER = os.path.join("/tmp", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXT = {"png", "jpg", "jpeg"}

# --- Hugging Face Dataset (FOR PERSISTENT UPLOADS) ---
HF_DATASET_ID = "mithun23112005/AgriculensUploads"
HF_WRITE_TOKEN = os.getenv("HF_WRITE_TOKEN")
hf_api = HfApi(token=HF_WRITE_TOKEN)

# --- DB Connection ---
mongo_client = MongoClient(MONGO_URI)
db = mongo_client.get_default_database()
users_col = db["users"]
predictions_col = db["predictions"]
chats_col = db["chats"]



# =======================
# LOAD MODELS
# =======================
print("Loading models...")
cnn_model = load_model(CNN_MODEL_PATH)
vit_sess = rt.InferenceSession(VIT_MODEL_PATH, providers=['CPUExecutionProvider'])
print("Models loaded.")

# =======================
# HELPERS
# =======================
import requests

def verify_google_id_token(id_token):
    url = "https://oauth2.googleapis.com/tokeninfo"
    resp = requests.get(url, params={"id_token": id_token})
    resp.raise_for_status()
    return resp.json()

def make_unique_username(base):
    base = base.lower().replace(" ", "")
    new = base
    c = 1
    while users_col.find_one({"username": new}):
        new = f"{base}{c}"
        c += 1
    return new


def process_and_predict_image(file_obj, user_id):
    filename = secure_filename(file_obj.filename)
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    saved_name = f"{user_id}_{timestamp}_{filename}"
    img_path = os.path.join(UPLOAD_FOLDER, saved_name)

    try:
        file_obj.save(img_path)
        public_image_url = ""
        try:
            hf_api.upload_file(
                path_or_fileobj=img_path,
                path_in_repo=saved_name,
                repo_id=HF_DATASET_ID,
                repo_type="dataset"
            )
            public_image_url = f"https://huggingface.co/datasets/{HF_DATASET_ID}/resolve/main/{saved_name}"
        except Exception as e:
            print(f"Error uploading to HF Hub: {e}")
            public_image_url = "upload_failed"

        img_pil = Image.open(img_path).convert("RGB")
        img = np.array(img_pil)

        segmented_leaf, health_mask = segment_leaf(img)
        health_index = compute_health_index(health_mask)
        cnn_input = preprocess_for_cnn(img)

        cnn_pred = cnn_model.predict(cnn_input)[0]
        input_name = vit_sess.get_inputs()[0].name
        vit_pred = vit_sess.run(None, {input_name: vit_input})[0][0]
        vit_pred = scipy.special.softmax(vit_pred)

        avg_pred = (cnn_pred + vit_pred) / 2.0
        class_index = int(np.argmax(avg_pred))
        confidence = float(np.max(avg_pred) * 100)
        class_name = CLASS_NAMES[class_index]
        plant, disease = split_plant_disease(class_name)


        prediction_data = {
            "image_url": public_image_url,
            "filename": saved_name,
            "prediction": class_name,
            "plant": plant,
            "disease": disease,
            "confidence": confidence,
            "health_index": health_index,
            "created_at": datetime.datetime.utcnow(),
        }
        return prediction_data
    finally:
        if os.path.exists(img_path):
            os.remove(img_path)
            print(f"Cleaned up temporary file: {img_path}")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def preprocess_for_cnn(img):
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def preprocess_for_vit(img):
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img

def get_weather(lat, lon):
    try:
        if lat is None or lon is None: return None
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
        r = requests.get(url, timeout=6)
        r.raise_for_status()
        data = r.json()
        return {
            "temperature": data.get("main", {}).get("temp"),
            "humidity": data.get("main", {}).get("humidity"),
            "description": data.get("weather", [{}])[0].get("description"),
            "location_name": data.get("name")
        }
    except Exception as e:
        return {"error": "Unable to fetch weather data", "details": str(e)}

def segment_leaf(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # --- Phase 1: detect main green area ---
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Morphological cleanup
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

    # Keep only largest contour = actual leaf
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img, green_mask
    largest = max(contours, key=cv2.contourArea)

    leaf_mask = np.zeros_like(green_mask)
    cv2.drawContours(leaf_mask, [largest], -1, 255, -1)

    # --- Phase 2: Re-include brown/yellow areas inside the same leaf contour ---
    # Any pixel inside the contour, even if not green, counts toward the leaf
    full_leaf_mask = np.zeros_like(leaf_mask)
    cv2.drawContours(full_leaf_mask, [largest], -1, 255, -1)

    # Now combine green mask + interior (to preserve damaged zones)
    final_mask = cv2.bitwise_or(leaf_mask, full_leaf_mask)

    # --- Phase 3: Clean and extract ---
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    segmented_leaf = cv2.bitwise_and(img, img, mask=final_mask)

    return segmented_leaf, final_mask

def compute_health_index(mask):
    # Calculates the percentage of the whole image that is non-zero (green)
    return round((cv2.countNonZero(mask) / mask.size) * 100, 2)
    
def split_plant_disease(name):
    parts = name.split('__')
    plant = parts[0]
    disease = " ".join(parts[1:]).replace("_", " ").strip()
    return plant, disease if disease else "healthy"

# =======================
# AUTH ROUTES
# =======================
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json(force=True)
    username = data.get("username")
    password = data.get("password")
    if not username or not password:
        return jsonify({"error": "username and password required"}), 400
    if users_col.find_one({"username": username}):
        return jsonify({"error": "username already exists"}), 409
    pw_hash = bcrypt.generate_password_hash(password).decode("utf-8")
    user = {"username": username, "password": pw_hash, "created_at": datetime.datetime.utcnow()}
    res = users_col.insert_one(user)
    return jsonify({"message": "user registered", "user_id": str(res.inserted_id)}), 201

@app.route("/auth/google", methods=["POST"])
def google_auth():
    data = request.get_json()
    id_token = data.get("id_token")
    if not id_token:
        return jsonify({"error": "id_token required"}), 400

    try:
        payload = verify_google_id_token(id_token)
    except:
        return jsonify({"error": "Invalid Google token"}), 400

    email = payload.get("email")
    if not email:
        return jsonify({"error": "Google email missing"}), 400

    # Check if user already exists
    user = users_col.find_one({"email": email})
    if not user:
        # Generate username from email localpart
        base_username = email.split("@")[0]
        username = make_unique_username(base_username)

        user_doc = {
            "username": username,
            "email": email,
            "password": None, 
            "created_at": datetime.datetime.utcnow(),
            "google": {"sub": payload.get("sub")}
        }
        res = users_col.insert_one(user_doc)
        user = users_col.find_one({"_id": res.inserted_id})

    access_token = create_access_token(identity=str(user["_id"]))

    return jsonify({
        "message": "Google login success",
        "user_id": str(user["_id"]),
        "username": user["username"],
        "access_token": access_token
    }), 200


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(force=True)
    username = data.get("username")
    password = data.get("password")
    if not username or not password:
        return jsonify({"error": "username and password required"}), 400
    user = users_col.find_one({"username": username})
    if not user or not bcrypt.check_password_hash(user["password"], password):
        return jsonify({"error": "invalid credentials"}), 401
    user_id_str = str(user["_id"])
    access_token = create_access_token(identity=user_id_str)
    return jsonify(access_token=access_token, user_id=user_id_str), 200

@app.route("/weather", methods=["GET"])
def weather():
    lat = request.args.get("lat")
    lon = request.args.get("lon")

    if not lat or not lon:
        return jsonify({"error": "lat and lon are required"}), 400

    try:
        lat = float(lat)
        lon = float(lon)
    except ValueError:
        return jsonify({"error": "lat and lon must be numbers"}), 400

    weather_data = get_weather(lat, lon)
    
    if not weather_data or "error" in weather_data:
        return jsonify({"error": "Unable to fetch weather data", "details": weather_data.get("details")}), 500

    # --- UPDATED RESPONSE ---
    return jsonify({
        "lat": lat,
        "lon": lon,
        "location_name": weather_data.get("location_name"), # <-- ADDED
        "weather": weather_data
    }), 200
    

# =======================
# PREDICTION / TIMELINE ROUTES
# =======================
@app.route("/predict", methods=["POST"])
@jwt_required()
def predict():
    user_id = get_jwt_identity()
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    file = request.files['image']
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    try:
        prediction_data = process_and_predict_image(file, user_id)
    except Exception as e:
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

    lat = request.form.get("lat")
    lon = request.form.get("lon")
    weather_data = get_weather(float(lat), float(lon)) if lat and lon else None
    
    is_cured = prediction_data["disease"].lower() == "healthy"

    case_doc = {
        "user_id": ObjectId(user_id),
        "plant": prediction_data["plant"],
        "latest_disease": prediction_data["disease"],
        "latest_health_index": prediction_data["health_index"], # <-- ADDED
        "created_at": datetime.datetime.utcnow(),
        "cured": is_cured,
        "cure_date": datetime.datetime.utcnow() if is_cured else None,
        "weather_on_creation": weather_data,
        "timeline": [ prediction_data ]
    }
    res = predictions_col.insert_one(case_doc)

    return jsonify({
        "message": "New prediction case created",
        "history_id": str(res.inserted_id),
        "prediction": prediction_data,
        "weather": weather_data,
        "lat": lat,
        "lon": lon
    }), 201

@app.route("/history/<case_id>/add_update", methods=["POST"])
@jwt_required()
def add_history_update(case_id):
    user_id = get_jwt_identity()
    if 'image' not in request.files: return jsonify({"error": "No image file provided"}), 400
    file = request.files['image']
    if file.filename == "" or not allowed_file(file.filename): return jsonify({"error": "File type not allowed"}), 400
    
    try:
        prediction_data = process_and_predict_image(file, user_id)
    except Exception as e:
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

    # --- ADD THIS ---
    # Get weather data, just like in the /predict route
    lat = request.form.get("lat")
    lon = request.form.get("lon")
    weather_data = get_weather(float(lat), float(lon)) if lat and lon else None
    # --- END ADD ---

    is_cured = prediction_data["disease"].lower() == "healthy"
    
    update = {
        "$push": {
            "timeline": prediction_data  # Note: You might want to add weather_data to this timeline entry too
        },
        "$set": {
            "latest_disease": prediction_data["disease"],
            "latest_health_index": prediction_data["health_index"], 
            "cured": is_cured,
            "cure_date": datetime.datetime.utcnow() if is_cured else None
            # Note: You could also update the weather here if you want
            # "latest_weather": weather_data 
        }
    }
    
    res = predictions_col.update_one({"_id": ObjectId(case_id), "user_id": ObjectId(user_id)}, update)
    
    if res.matched_count == 0: return jsonify({"error": "Case not found or not authorized"}), 404
    
    # --- UPDATE THIS RESPONSE ---
    # Now it includes both prediction and weather
    return jsonify({
        "message": "Case timeline updated", 
        "new_prediction": prediction_data,
        "weather": weather_data,
        "lat": lat,
        "lon": lon  
    }), 200

@app.route("/history", methods=["GET"])
@jwt_required()
def get_history_list():
    user_id = get_jwt_identity() 
    try:
        docs = predictions_col.find(
            {"user_id": ObjectId(user_id)},
            {"timeline": 0}
        ).sort("created_at", -1)
        
        out = []
        for d in docs:
            out.append({
                "history_id": str(d["_id"]),
                "plant": d.get("plant"),
                "latest_disease": d.get("latest_disease"),
                "latest_health_index": d.get("latest_health_index"), # <-- ADDED
                "created_at": d.get("created_at").isoformat(),
                "cured": d.get("cured", False)
            })
        return jsonify({"history": out}), 200
    except Exception as e: return jsonify({"error": "Failed to fetch history list", "details": str(e)}), 500

@app.route("/history/<case_id>", methods=["GET"])
@jwt_required()
def get_history_detail(case_id):
    user_id = get_jwt_identity()
    try:
        doc = predictions_col.find_one({"_id": ObjectId(case_id), "user_id": ObjectId(user_id)})
        if not doc: return jsonify({"error": "Case not found or not authorized"}), 404
        timeline_json = [{"image_url": t.get("image_url"), "filename": t.get("filename"), "prediction": t.get("prediction"), "plant": t.get("plant"), "disease": t.get("disease"), "confidence": t.get("confidence"), "health_index": t.get("health_index"), "created_at": t.get("created_at").isoformat()} for t in doc.get("timeline", [])]
        response_data = {
            "history_id": str(doc["_id"]),
            "user_id": str(doc["user_id"]),
            "plant": doc.get("plant"),
            "latest_disease": doc.get("latest_disease"),
            "latest_health_index": doc.get("latest_health_index"), # <-- ADDED
            "created_at": doc.get("created_at").isoformat(),
            "cured": doc.get("cured", False),
            "cure_notes": doc.get("cure_notes"),
            "weather_on_creation": doc.get("weather_on_creation"),
            "timeline": timeline_json
        }
        return jsonify(response_data), 200
    except Exception as e: return jsonify({"error": "Failed to fetch case details", "details": str(e)}), 500

# --- (The rest of your routes are unchanged) ---
@app.route("/history/<history_id>/mark_cured", methods=["POST"])
@jwt_required()
def mark_cured(history_id):
    user_id = get_jwt_identity()
    notes = request.form.get("notes", None)
    update = {"$set": {"cured": True, "cure_date": datetime.datetime.utcnow(), "cure_notes": notes}}
    res = predictions_col.update_one({"_id": ObjectId(history_id), "user_id": ObjectId(user_id)}, update)
    if res.matched_count == 0: return jsonify({"error": "Record not found or not authorized"}), 404
    return jsonify({"message": "Marked cured"}), 200

@app.route("/stats/cured_count", methods=["GET"])
@jwt_required()
def cured_count():
    user_id = get_jwt_identity()
    total = predictions_col.count_documents({"user_id": ObjectId(user_id)})
    cured = predictions_col.count_documents({"user_id": ObjectId(user_id), "cured": True})
    not_cured = total - cured
    return jsonify({"total": total, "cured": cured, "not_cured": not_cured}), 200

@app.route("/chat", methods=["POST"])
@jwt_required()
def create_chat():
    user_id = get_jwt_identity()
    body = request.get_json(force=True)
    title = body.get("title", "New Chat")
    context_id = body.get("context_id", None)
    messages = body.get("messages", [])
    doc = {"user_id": ObjectId(user_id), "title": title, "context_id": context_id, "created_at": datetime.datetime.utcnow(), "messages": messages}
    res = chats_col.insert_one(doc)
    return jsonify({"message": "Chat created", "chat_id": str(res.inserted_id), "title": title, "created_at": doc["created_at"].isoformat()}), 201

@app.route("/chat", methods=["GET"])
@jwt_required()
def get_chat_list():
    user_id = get_jwt_identity()
    docs = chats_col.find({"user_id": ObjectId(user_id)}, {"messages": 0}).sort("created_at", -1)
    out = [{"chat_id": str(d["_id"]), "title": d.get("title", "Chat"), "context_id": d.get("context_id"), "created_at": d.get("created_at").isoformat()} for d in docs]
    return jsonify({"chats": out}), 200

@app.route("/chat/<chat_id>", methods=["GET"])
@jwt_required()
def get_chat_messages(chat_id):
    user_id = get_jwt_identity()
    doc = chats_col.find_one({"_id": ObjectId(chat_id), "user_id": ObjectId(user_id)})
    if not doc: return jsonify({"error": "Chat not found or not authorized"}), 404
    messages_json = []
    for msg in doc.get("messages", []):
        if isinstance(msg.get("created_at"), datetime.datetime): msg["created_at"] = msg["created_at"].isoformat()
        messages_json.append(msg)
    return jsonify({"chat_id": str(doc["_id"]), "title": doc.get("title"), "context_id": doc.get("context_id"), "created_at": doc.get("created_at").isoformat(), "messages": messages_json}), 200

@app.route("/chat/<chat_id>", methods=["PUT"])
@jwt_required()
def update_chat_messages(chat_id):
    user_id = get_jwt_identity()
    body = request.get_json(force=True)
    messages = body.get("messages")
    if messages is None: return jsonify({"error": "A 'messages' array is required"}), 400
    update = {"$set": {"messages": messages, "last_updated": datetime.datetime.utcnow()}}
    res = chats_col.update_one({"_id": ObjectId(chat_id), "user_id": ObjectId(user_id)}, update)
    if res.matched_count == 0: return jsonify({"error": "Chat not found or not authorized"}), 404
    return jsonify({"message": "Chat updated successfully"}), 200

@app.route("/chat/<chat_id>", methods=["DELETE"])
@jwt_required()
def delete_chat(chat_id):
    user_id = get_jwt_identity()
    
    # Attempt to delete the chat document, ensuring it matches both the chat_id and the user_id
    res = chats_col.delete_one({"_id": ObjectId(chat_id), "user_id": ObjectId(user_id)})

    if res.deleted_count == 0:
        # Returns 404 if the chat_id was invalid or the chat did not belong to the user
        return jsonify({"error": "Chat not found or not authorized to delete"}), 404

    return jsonify({"message": "Chat deleted successfully"}), 200


    # ==========================================================
# NEW ROUTE → /chat/generate  (uses Gemini + chat history)
# ==========================================================
@app.route("/chat/generate", methods=["POST"])
@jwt_required()
def chat_generate():
    user_id = get_jwt_identity()
    data = request.get_json() or {}

    message = data.get("message")
    chat_id = data.get("chat_id")
    lang_code = data.get("language", "en")

    if not message:
        return jsonify({"error": "Message is required"}), 400

    # 🌍 Language map
    language_map = {
        "en": "English",
        "hi": "Hindi",
        "kn": "Kannada",
        "ta": "Tamil",
        "te": "Telugu",
        "ml": "Malayalam"
    }

    language = language_map.get(lang_code, "English")

    # =========================
    # ✅ FIXED HISTORY HANDLING
    # =========================
    history = []

    if chat_id:
        chat_doc = chats_col.find_one({
            "_id": ObjectId(chat_id),
            "user_id": ObjectId(user_id)
        })

        if chat_doc:
            for msg in chat_doc.get("messages", []):
                role = msg.get("role", "user")

                # ✅ Fix invalid roles
                if role not in ["user", "assistant", "system"]:
                    role = "user"

                # ✅ Fix content
                content = msg.get("content")

                # Handle old Gemini format (parts)
                if not content and "parts" in msg:
                    content = msg["parts"][0]

                # Only valid messages
                if content:
                    history.append({
                        "role": role,
                        "content": content
                    })

    # =========================
    # 🔥 GROQ API CALL
    # =========================
    try:
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",  # ✅ Working model
                "messages": [
                    {"role": "system", "content": f"Respond in {language}"},
                    *history,
                    {"role": "user", "content": message}
                ]
            }
        )

        result = response.json()
        print("Groq response:", result)

        # ✅ Safe response handling
        if "choices" in result:
            reply = result["choices"][0]["message"]["content"]
        else:
            return jsonify({"reply": f"Groq error: {result}"}), 200

        return jsonify({"reply": reply}), 200

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"reply": "AI service unavailable"}), 200
    
@app.route('/')
def home():
    return jsonify({
        "message": "Crop Leaf Disease Identification API is running.",
        "notes": "Use /register and /login to obtain user_id, then call endpoints with user_id parameter."
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)

