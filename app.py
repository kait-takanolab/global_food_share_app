import os
import io

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import keras
from keras.models import load_model
from keras.preprocessing.image import img_to_array

app = Flask(__name__)

# モデルの読み込み
try:
    # 2.15環境で古いモデルを読み込む際の互換性フラグ
    model = load_model('model_mini.keras', compile=False)
    print("--- モデルの読み込みに成功しました ---")
except Exception as e:
    print(f"読み込み失敗: {e}")

categories = ['30', '60', '100']

@app.route('/', methods=['POST'])
def top():
    image_bytes = request.data
    if not image_bytes:
        return jsonify({"error": "No data received"}), 400
    
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert('RGB')
        img = img.resize((150, 150))
        
        img_array = img_to_array(img) / 255.0
        img_expand = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_expand)
        predicted_class_index = np.argmax(predictions[0])
        predicted_label = categories[predicted_class_index]
        
        return jsonify({"result": predicted_label})
    
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)