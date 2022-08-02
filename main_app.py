import os
import numpy as np
# from flask_ngrok import run_with_ngrok
from flask import Flask, request, render_template, send_from_directory, redirect, url_for
# from werkzeug.utils import secure_filename
# import pickle
import cv2
import tensorflow as tf
# from tensorflow.keras.preprocessing import image

name_animal = ["Lion","Jaguar","Panda","Arctic Fox","Rhino","Orangutan","African Elephant"]
animals = name_animal

with open('all-animal-desc-archive-7.txt', encoding="utf-8") as f:
    animal_desc_list = f.readlines()

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static\\"

path = './animal-model.h5'
model= tf.keras.models.load_model(path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/getprediction',methods=['POST'])
def getprediction():    
    img = request.files['img']
    filename = img.filename
    print("FILENAMEEE: ", filename)
    img.save(os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], filename))
    
    print('imggg: ', os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], img.filename))
    img_arr = cv2.imread(os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], img.filename))
    img_arr = cv2.resize(img_arr, (224,224))
    img_arr = img_arr / 255.0
    img_arr = img_arr.reshape(1, 224,224,3)

    prediction = model.predict(img_arr)
    print('prediction: ', prediction[0])
    
    return render_template('index.html', height0=0, height=450, vis0='hidden', visibility='visible', filename=filename, output1='{}'.format(animals[np.where(prediction[0].max() == prediction[0])[0][0]]), output2='{}'.format(animal_desc_list[np.where(prediction[0].max() == prediction[0])[0][0]]))

# @app.route('/display/<filename>')
# def display_image(filename):
# 	print('display_image filename: ' + filename)
# 	return redirect(url_for('static', filename=os.path.join(filename)))


if __name__ == "__main__":
    app.run(debug=True)