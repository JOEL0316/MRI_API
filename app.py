import os
import requests
import base64
import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Configuración básica de la aplicación
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# URLs de las APIs (mejor como variables de entorno)
CLASSIFIER_API_URL = os.environ.get('CLASSIFIER_API_URL', "https://apiclassifier.onrender.com/predict")
SEGMENTATION_API_URL = os.environ.get('SEGMENTATION_API_URL', "https://tumor-segmentation-api-latest.onrender.com/segment")

# Configuración de tamaño máximo de imagen
MAX_IMAGE_SIZE = (500, 500)
IMAGE_QUALITY = 85

def process_image(file):
    """Procesa la imagen y devuelve versión compatible para visualización"""
    try:
        img = Image.open(file)
        
        # Convertir a RGB si es necesario
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Redimensionar manteniendo aspect ratio
        img.thumbnail(MAX_IMAGE_SIZE, Image.LANCZOS)
        
        # Guardar en buffer
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=IMAGE_QUALITY)
        buffer.seek(0)
        
        return buffer
    except Exception as e:
        raise ValueError(f"Error al procesar imagen: {str(e)}")

def encode_image(img, format='PNG'):
    """Codifica una imagen PIL a base64"""
    try:
        buffer = io.BytesIO()
        img.save(buffer, format=format)
        return f"data:image/{format.lower()};base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"
    except Exception as e:
        raise ValueError(f"Error al codificar imagen: {str(e)}")

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
            
        # Procesar imagen original para visualización
        processed_img_buffer = process_image(file)
        encoded_original = encode_image(Image.open(processed_img_buffer), 'JPEG')
        
        # Enviar al clasificador
        file.seek(0)
        classifier_response = requests.post(
            CLASSIFIER_API_URL, 
            files={'image': file},
            timeout=10
        )
        classifier_response.raise_for_status()
        classifier_data = classifier_response.json()

        tumor_prob = classifier_data.get('tumor_probability', 0)

        if tumor_prob <= 0.5:
            return render_template('index.html', 
                               result="No hay tumor",
                               probability=float(tumor_prob),
                               original_image=encoded_original)

        # Enviar a segmentación
        file.seek(0)
        segmentation_response = requests.post(
            SEGMENTATION_API_URL, 
            files={'image': file},
            timeout=10
        )
        segmentation_response.raise_for_status()
        segmentation_data = segmentation_response.json()

        # Procesar máscara
        mask = np.array(segmentation_data.get('segmentation_mask', []), dtype=np.uint8) * 255
        if mask.size == 0:
            raise ValueError("Máscara de segmentación vacía")
            
        mask_img = Image.fromarray(mask)
        
        # Crear imagen superpuesta en ROJO
        original_img = Image.open(processed_img_buffer)
        red_mask = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        red_mask[mask > 0] = [255, 0, 0, 180]  # RGBA: Rojo con 70% transparencia
        
        # Combinar con la imagen original
        overlay = Image.fromarray(red_mask)
        original_rgba = original_img.convert('RGBA')
        overlay = Image.alpha_composite(original_rgba, overlay)
        
        return render_template('index.html',
                            result="Tumor detectado",
                            probability=float(tumor_prob),
                            original_image=encoded_original,
                            mask=encode_image(mask_img),
                            overlay_image=encode_image(overlay))

    except requests.exceptions.RequestException as e:
        return render_template('index.html', error=f"Error de conexión con API: {str(e)}")
    except Exception as e:
        return render_template('index.html', error=f"Error interno: {str(e)}")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)