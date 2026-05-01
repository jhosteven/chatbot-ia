from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re
import nltk
from nltk.corpus import stopwords
import nltk
import os
nltk.download('stopwords', quiet=True)

def cargar_modelo():
    global modelo, vectorizer

    if modelo is None or vectorizer is None:
        print("🔄 Cargando modelo IA...")
        modelo = pickle.load(open("modelo_chatbot.pkl", "rb"))
        vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
        print("✅ Modelo cargado")
    
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Cargar modelo y vectorizador
modelo = None
vectorizer = None

stop_words = set(stopwords.words('spanish'))

# Limpieza de texto
def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^a-záéíóúñ\s]', '', texto)
    palabras = texto.split()
    palabras = [p for p in palabras if p not in stop_words]
    return " ".join(palabras)

# Predicción
def predecir(texto):
    texto_limpio = limpiar_texto(texto)
    texto_vec = vectorizer.transform([texto_limpio])
    
    prediccion = modelo.predict(texto_vec)[0]
    probabilidad = modelo.predict_proba(texto_vec).max()
    
    return prediccion, probabilidad

# Endpoint raíz (PRIMERO)
@app.route('/')
def home():
    return "Chatbot IA activo"
    
@app.route('/warmup')
def warmup():
    cargar_modelo()
    return jsonify({"status": "warm", "message": "Modelo listo"})

# Endpoint chatbot
@app.route('/chatbot', methods=['POST'])
def chatbot():
    cargar_modelo()
    data = request.get_json(force=True)
    mensaje = data.get("mensaje", "")
    
    intencion, prob = predecir(mensaje)

    # Respuestas
    respuestas = {
        "informacion": "Te puedo brindar información sobre nuestros servicios.",
        "precio": "Nuestros precios varían según el servicio.",
        "compra": "Perfecto, te ayudaré a contratar el servicio."
    }

    respuesta = respuestas.get(intencion, "No entendí tu mensaje.")

    # Lead scoring
    if intencion == "compra" and prob > 0.7:
        nivel = "alto"
    elif prob > 0.4:
        nivel = "medio"
    else:
        nivel = "bajo"

    return jsonify({
        "respuesta": respuesta,
        "intencion": intencion,
        "confianza": float(prob),
        "nivel_interes": nivel
    })

