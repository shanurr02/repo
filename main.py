#   FINAL Code updates
from flask import Flask, request, jsonify
import numpy as np
from funcd import create_final_dataframe
from funcd import find_matching_rows
import os
import json
import pandas as pd
import easyocr
from werkzeug.utils import secure_filename
# from tensorflow import keras
import joblib
import sklearn
import pickle
import base64
from PIL import Image
from io import BytesIO



app = Flask(__name__)

# UPLOAD_FOLDER = './'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model and vectorizer
# script_dir = os.path.dirname(os.path.realpath(__file__))
# vector_path = os.path.join(script_dir, 'model.pkl')
# model_path = os.path.join(script_dir, 'vec.pkl')
# Load the model and vectorizer
model = pickle.load(open('dectree.pkl', 'rb'))
vectorizer = pickle.load(open('decvec.pkl', 'rb'))
# vectorizer = joblib.load(vector_path)
# model = joblib.load(model_path)

@app.route('/predict', methods=['POST'])

def predict_endpoint():
    request_data = request.get_json()
    print(request_data['text'])
    if 'text' in request_data and 'url' in request_data:
        print("inside if else")
        text_data = request_data['text']
        print("text extracted")
        
        image_url = request_data['url']
        image_data = base64.b64decode(image_url)
        print("image extracted and decoded")

        # Create a PIL Image object from binary data
        image = Image.open(BytesIO(image_data))
        # Save the image to a .png file
        image.save('./image.png', format='PNG',quality=95)
        print("image saved in root")
        

        # Process text data
        df_text = pd.DataFrame(text_data, columns=['id', 'text'])
        print("making frame")
        text_values = df_text['text'].values
        X_pred_text = vectorizer.transform(text_values)
        predictions_text = model.predict(X_pred_text)
        results_text = pd.DataFrame({'id': df_text['id'], 'text': df_text['text'], 'prediction_text': predictions_text})
        print("3")

        # Process image data
        current_directory = os.getcwd()
        image_path = os.path.join(current_directory, 'image.png')
        # loaded_image = Image.open(image_path)
        
        print("path:",image_path)
        print("4")


        # Use EasyOCR on the image
        reader = easyocr.Reader(['en'])
        print("5")

        result_image = reader.readtext(image_path, paragraph=True)
        print("6")

        df_image = pd.DataFrame(result_image, columns=['bbox', 'text'])
        print("7")

        df_image['text'] = df_image['text'].apply(lambda x: x.lower())
        text_values_image = df_image['text'].values
        X_pred_image = vectorizer.transform(text_values_image)
        predictions_image = model.predict(X_pred_image)
        results_image = pd.DataFrame({'id': df_text['id'], 'text': df_image['text'], 'prediction_image': predictions_image})

        matching_rows = find_matching_rows(results_image, results_text)

        #Create the final DataFrame
        final_df = create_final_dataframe(results_image, results_text, matching_rows)

        # Combine results
        combined_results = {
            'predictions': final_df.to_dict(orient='records')
        }
        
        print(combined_results)

        return jsonify(combined_results)
    
   

    return jsonify({'error': 'No text or image data provided'}), 400

if __name__ == '__main__':
    app.run(debug=True)
