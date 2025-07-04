import os
import json
import requests
import base64
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from google.oauth2 import service_account
from google.auth.transport.requests import Request
import tensorflow as tf
import numpy as np
from PIL import Image

# Cargar variables de entorno
load_dotenv()
MODEL_PATH = 'modelo_nuevas_razas_2.h5'
app = Flask(__name__)
CORS(app)  # Habilita CORS para todas las rutas y orígenes
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Inicializar modelo ---
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Modelo cargado correctamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    model = None  

razas = ['Brangus', 'Charolesa', 'Holstein', 'Jersey', 'Nelore', 'Salers', 'Simmental']

def procesar_imagen(image_file):
    img = Image.open(image_file).convert('RGB').resize((150, 150))  
    arr = np.array(img) / 255.0  
    arr = np.expand_dims(arr, axis=0) 
    return arr.astype(np.float32)


def get_access_token():
    path_to_secret = os.getenv("PATH_TO_SECRET")
    scopes = ["https://www.googleapis.com/auth/firebase.messaging"]

    credentials = service_account.Credentials.from_service_account_file(
        path_to_secret,
        scopes=scopes,
    )
    credentials.refresh(Request())
    return credentials.token

def send_push_notification(device_token, title, body, data=None):
    access_token = get_access_token()
    project_id = os.getenv("PROJECT_ID")

    url = f"https://fcm.googleapis.com/v1/projects/{project_id}/messages:send"

    message = {
        "message": {
            "token": device_token,
            "notification": {
                "title": title,
                "body": body
            },
            "data": data or {}
        }
    }

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json; UTF-8",
    }

    response = requests.post(url, headers=headers, json=message)

    return response

@app.route("/send-notification", methods=["POST"])
def notify():
    body = request.json
    device_token = body.get("token")
    title = body.get("title")
    message_body = body.get("body")
    data = body.get("data")

    if not all([device_token, title, message_body]):
        return jsonify({"error": "Missing required fields"}), 400

    response = send_push_notification(device_token, title, message_body, data)
    
    if response.status_code == 200:
        return jsonify({"success": True, "message": "Notification sent"})
    else:
        return jsonify({"success": False, "error": response.text}), response.status_code

def analizar_img(imagen_path):
    # Tu lógica de análisis real aquí
    return True  # Cambia según tu lógica


def codificar_imagen_a_base64(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return 'No image part', 400

    file = request.files['image']
    if file.filename == '':
        return 'No selected file', 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    print(f"Imagen guardada en: {filepath}")

    try:
        # Paso 1: codificar imagen en base64
        img_b64 = codificar_imagen_a_base64(filepath)

        # Paso 2: enviar a endpoint externo
        with open(filepath, "rb") as img_file:
            response = requests.post(
                "https://v9k5scrk-8000.brs.devtunnels.ms/analizar",
                files={"file": img_file},
                timeout=10
            )

        if response.status_code != 200:
            print("Error desde el analizador de ganado:", response.text)
            return jsonify({'error': 'Fallo al analizar si hay ganado'}), 400

        resultado = response.json().get("descripcion", "").strip().lower()
        if resultado != "true":
            print("No se detectó ganado en la imagen")
            os.remove(filepath)  # eliminar imagen
            return jsonify({'mensaje': 'No se detectó ganado'}), 204

        # Paso 3: continuar con predicción de raza
        input_data = procesar_imagen(file)
        prediction = model.predict(input_data)[0]
        raza_predicha = razas[np.argmax(prediction)]

        # Paso 4: notificación
        send_push_notification(
            device_token='fUC_R1b8TsqxFq3RrSNi59:APA91bFY9dlSAWhzZnUxXx6VactxP5i5aJsxx34MBADSW6UWOCwgK_EwtjNCuAi4yqBb_KX9Gj44CPXxpwz_AAYMBGKUaqzLd2nBuBeJn-eRmct5Ss3-4sg',
            title="Raza de vaca predicha",
            body=f"La raza de la vaca es: {raza_predicha}",
            data={"raza": raza_predicha}
        )

        return jsonify({'raza': raza_predicha})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error procesando la imagen: {str(e)}'}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)

# e0SexImkQNSt_NZFQ3FUnF:APA91bHfHT68s9Kgfuqzcdx3HcctWodmQzhW0kcvwi1jfuX3itS1RWUKZnMBfW4qwqTjzjcJffFhsGzCa3HdgGO4nKb_NEHj776B4zN0Rr26g-cUbrl4K28
# evBICKeCTNm5JK7oOyG-tU:APA91bFz5ArCJQ1T69qGVgI3QVgqBZI9CmrmdlV_QYDaZNBkMvmTHfS_cztM2pGdax3kY_JvnL9pd6_OYaGD-ryfg3FQNJ4Bi9vqu3oimMzH_6vs98d_oQc