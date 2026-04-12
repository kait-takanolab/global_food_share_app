import os
import io

from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf
import numpy as np
from PIL import Image
import keras
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import json
import base64

app = Flask(__name__)

# モデルの読み込み
try:
    model = load_model('model/model_large.keras', compile=False)
    print("--- モデルの読み込みに成功しました ---")
except Exception as e:
    print(f"読み込み失敗: {e}")

#モデルのラベル設定
categories = ['30', '60', '100']

#画像枚数カウンタの設定
COUNTER_FILE = "count.txt"


#画像枚数カウンタ用_カウントアップ関数
def get_next_count():
    if os.path.exists(COUNTER_FILE):
        with open(COUNTER_FILE, "r") as f:
            return int(f.read().strip())
    return 0

#画像枚数カウンタ用_カウント保存関数
def save_count(count):
    with open(COUNTER_FILE, "w") as f:
        f.write(str(count))

@app.route('/predict', methods=['POST'])
def predict():
    current_count = get_next_count() + 1
    image = request.get_data()

    if len(image) == 0:
        return "サーバーには何も届いていません(0 bytes)", 400
    
    try:
        img = Image.open(io.BytesIO(image))
        img = img.convert('RGB')
        img = img.resize((150, 150))
        
        img_array = img_to_array(img) / 255.0
        img_expand = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_expand)
        predicted_class_index = np.argmax(predictions[0])
        predicted_label = categories[predicted_class_index]

        #画像を保存
        # バイナリデータを画像オブジェクトに変換
        img = Image.open(io.BytesIO(image))
        
        # PNGとして保存
        name = f"./picture/image_{current_count}.png"
        img.save(name, "PNG")
        
        print("PNG形式で保存しました。")
        
        return jsonify({"image_path" : f"./picture/image_{current_count}.png", "predicted_label": predicted_label}),200
    
    except Exception as e:
        return "error"+str(e), 500

@app.route('/returnPNG', methods=['POST'])
def returnPNG():
    file_path = request.get_data()
    
    with open(file_path, "rb") as image_file:
        # 1. 画像をバイナリで読み込む
        binary_data = image_file.read()
        
        # 2. Base64でエンコードし、さらに文字列(utf-8)に変換
        base64_string = base64.b64encode(binary_data).decode('utf-8')
    
    # 3. 辞書形式にして返す（Flask等ならそのままreturn可能）
    return {
        "image_data": base64_string
    }

@app.route("/images/<filename>")
def get_image(filename):
    return send_from_directory("picture", filename)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)