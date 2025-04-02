import os
import requests
import base64
import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor

# Configuración de la aplicación Flask
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Configuración de las APIs (usar variables de entorno en producción)
CLASSIFIER_API_URL = os.environ.get('CLASSIFIER_API_URL', "https://apiclassifier.onrender.com/predict")
SEGMENTATION_API_URL = os.environ.get('SEGMENTATION_API_URL', "https://tumor-segmentation-api-latest.onrender.com/segment")

# Parámetros de optimización
MAX_IMAGE_SIZE = (512, 512)  # Tamaño máximo para procesamiento
PREVIEW_SIZE = (500, 500)    # Tamaño para visualización
IMAGE_QUALITY = 80           # Calidad de compresión (80%)

# Configuración de reintentos para las APIs
session = requests.Session()
retry_strategy = Retry(
    total=3,                  # Número máximo de reintentos
    backoff_factor=1,         # Tiempo de espera entre reintentos (segundos)
    status_forcelist=[408, 429, 500, 502, 503, 504]  # Códigos para reintentar
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)

def optimize_image(file, max_size=MAX_IMAGE_SIZE, quality=IMAGE_QUALITY):
    """Optimiza la imagen reduciendo tamaño y calidad"""
    try:
        img = Image.open(file)
        
        # Convertir a RGB si es necesario
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Redimensionar manteniendo aspect ratio
        img.thumbnail(max_size, Image.LANCZOS)
        
        # Guardar en buffer con compresión
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality, optimize=True)
        buffer.seek(0)
        
        return buffer
    except Exception as e:
        raise ValueError(f"Error al optimizar imagen: {str(e)}")

def encode_image(img, format='PNG'):
    """Codifica una imagen PIL a base64"""
    try:
        buffer = io.BytesIO()
        img.save(buffer, format=format)
        return f"data:image/{format.lower()};base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"
    except Exception as e:
        raise ValueError(f"Error al codificar imagen: {str(e)}")

def process_for_display(file, max_size=PREVIEW_SIZE):
    """Procesa imagen para visualización en frontend"""
    try:
        img = Image.open(file)
        img.thumbnail(max_size, Image.LANCZOS)
        return encode_image(img, 'JPEG')
    except Exception as e:
        raise ValueError(f"Error al procesar para visualización: {str(e)}")

def call_classifier_api(file):
    """Envía imagen al clasificador con manejo de errores"""
    try:
        response = session.post(
            CLASSIFIER_API_URL,
            files={'image': file},
            timeout=20
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise ConnectionError(f"Error en API de clasificación: {str(e)}")

def call_segmentation_api(file):
    """Envía imagen al segmentador con manejo de errores"""
    try:
        response = session.post(
            SEGMENTATION_API_URL,
            files={'image': file},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise ConnectionError(f"Error en API de segmentación: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return render_template('index.html', error="No se proporcionó imagen")
            
        file = request.files['image']
        
        if file.filename == '':
            return render_template('index.html', error="Nombre de archivo vacío")
        
        # 1. Optimizar imagen (reduce tamaño antes de enviar a APIs)
        optimized_buffer = optimize_image(file)
        
        # 2. Procesar para visualización
        optimized_buffer.seek(0)
        encoded_original = process_for_display(optimized_buffer)
        
        # 3. Enviar al clasificador (con reintentos automáticos)
        optimized_buffer.seek(0)
        classifier_data = call_classifier_api(optimized_buffer)
        
        tumor_prob = classifier_data.get('tumor_probability', 0)
        
        # 4. Si no hay tumor, retornar temprano
        if tumor_prob <= 0.5:
            return render_template('index.html', 
                               result="No hay tumor",
                               probability=float(tumor_prob),
                               original_image=encoded_original)

        # 5. Si hay tumor, enviar a segmentación (en paralelo si fuera necesario)
        optimized_buffer.seek(0)
        segmentation_data = call_segmentation_api(optimized_buffer)
        
        # 6. Procesar máscara de segmentación
        mask = np.array(segmentation_data.get('segmentation_mask', []), dtype=np.uint8)
        if mask.size == 0:
            raise ValueError("Máscara de segmentación vacía")
        
        mask = mask * 255  # Escalar a 0-255
        mask_img = Image.fromarray(mask)
        
        # 7. Crear superposición visual
        original_img = Image.open(optimized_buffer)
        red_mask = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        red_mask[mask > 0] = [255, 0, 0, 180]  # RGBA: Rojo con transparencia
        
        overlay = Image.fromarray(red_mask)
        original_rgba = original_img.convert('RGBA')
        combined = Image.alpha_composite(original_rgba, overlay)
        
        # 8. Codificar resultados
        return render_template('index.html',
                            result="Tumor detectado",
                            probability=float(tumor_prob),
                            original_image=encoded_original,
                            mask=encode_image(mask_img),
                            overlay_image=encode_image(combined))

    except ConnectionError as e:
        return render_template('index.html', 
                            error=f"Error de conexión: {str(e)}. Intente nuevamente más tarde.")
    except ValueError as e:
        return render_template('index.html', 
                            error=f"Error al procesar imagen: {str(e)}")
    except Exception as e:
        return render_template('index.html', 
                            error=f"Error inesperado: {str(e)}")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, threaded=True)