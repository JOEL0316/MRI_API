import collections
import collections.abc
import os
import requests
import base64
import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Restaurar compatibilidad
for type_name in ('Iterable', 'Mapping', 'MutableSet', 'Sequence'):
    if not hasattr(collections, type_name) and hasattr(collections.abc, type_name):
        setattr(collections, type_name, getattr(collections.abc, type_name))

# URLs de las APIs
CLASSIFIER_API_URL = "https://apiclassifier.onrender.com/predict"
SEGMENTATION_API_URL = "https://tumor-segmentation-api-latest.onrender.com/segment"

app = Flask(__name__, template_folder='templates')
CORS(app)

def process_image(file):
    """Procesa la imagen y devuelve versión compatible para visualización"""
    img = Image.open(file)
    
    # Convertir a RGB si es necesario
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Redimensionar manteniendo aspect ratio
    max_size = (500, 500)
    img.thumbnail(max_size, Image.LANCZOS)
    
    # Guardar en buffer
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=85)
    buffer.seek(0)
    
    return buffer

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        
        # Procesar imagen original para visualización
        processed_img_buffer = process_image(file)
        encoded_original = f"data:image/jpeg;base64,{base64.b64encode(processed_img_buffer.getvalue()).decode('utf-8')}"
        
        # Enviar al clasificador
        file.seek(0)
        classifier_response = requests.post(CLASSIFIER_API_URL, files={'image': file})
        classifier_data = classifier_response.json()

        if 'error' in classifier_data:
            raise Exception(f"Error en clasificación: {classifier_data['error']}")

        tumor_prob = classifier_data['tumor_probability']

        if tumor_prob <= 0.5:
            return render_template('index.html', 
                               result="No hay tumor",
                               probability=float(tumor_prob),
                               original_image=encoded_original)

        # Enviar a segmentación
        file.seek(0)
        segmentation_response = requests.post(SEGMENTATION_API_URL, files={'image': file})
        segmentation_data = segmentation_response.json()

        if 'error' in segmentation_data:
            raise Exception(f"Error en segmentación: {segmentation_data['error']}")

        # Procesar máscara
        mask = np.array(segmentation_data['segmentation_mask'], dtype=np.uint8) * 255
        mask_img = Image.fromarray(mask)
        
        # Crear imagen superpuesta en ROJO
        original_img = Image.open(processed_img_buffer)
        
        # Crear una máscara roja con transparencia
        red_mask = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        red_mask[mask > 0] = [255, 0, 0, 180]  # RGBA: Rojo con 70% transparencia
        
        # Combinar con la imagen original
        overlay = Image.fromarray(red_mask)
        original_rgba = original_img.convert('RGBA')
        overlay = Image.alpha_composite(original_rgba, overlay)
        
        # Codificar imágenes
        def encode_image(img, format='PNG'):
            buffer = io.BytesIO()
            img.save(buffer, format=format)
            return f"data:image/{format.lower()};base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"

        return render_template('index.html',
                            result="Tumor detectado",
                            probability=float(tumor_prob),
                            original_image=encoded_original,
                            mask=encode_image(mask_img),
                            overlay_image=encode_image(overlay))

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5005))
    app.run(host="0.0.0.0", port=port)