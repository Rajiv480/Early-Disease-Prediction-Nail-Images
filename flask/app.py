
import numpy as np
import os
from flask import Flask, app, request, render_template
from keras import models
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.python.ops.gen_array_ops import concat
from keras.applications.inception_v3 import preprocess_input
import requests
from flask import Flask, request, render_template, redirect, url_for
import cv2



# Loading the model
# modeln=load_model("model_vgg16.h5")
# modeln=load_model("model_v3_vgg16.h5")

# Build an absolute path to the model file relative to this script's directory
BASE_DIR = os.path.dirname(__file__)
MODEL_FILENAME = "Vgg-16-nail-disease.h5"
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)

modeln = load_model(MODEL_PATH)


app=Flask(__name__)

#default home page or route
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/index')
def inde1():
    return render_template('index.html')



@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/nailhome')
def nailhome():
    return render_template('nailhome.html')

@app.route('/nailpred')
def nailpred():
    return render_template('nailpred.html')

@app.route('/nailresult',methods=["GET","POST"])
def nres():
    if request.method=="POST":
        f=request.files['image']
        basepath=os.path.dirname(__file__)

        # Ensure the uploads directory exists
        upload_dir = os.path.join(basepath, 'uploads')
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir, exist_ok=True)

        filepath=os.path.join(upload_dir, f.filename)

        f.save(filepath)
        img=image.load_img(filepath,target_size=(224,224))
        x=image.img_to_array(img)
        img_data = cv2.resize(x, (64,64))
        img_data = np.expand_dims(img_data, axis = 0)
        # x=np.expand_dims(x,axis=0)
        # img_data=preprocess_input(x)
        prediction=np.argmax(modeln.predict(img_data))
        
        index=['Darier_s disease', 'Muehrck-e_s lines', 'aloperia areata', 'beau_s lines', 'bluish nail',
               'clubbing','eczema','half and half nailes (Lindsay_s nails)','koilonychia','leukonychia',
               'onycholycis','pale nail','red lunula','splinter hemmorrage','terry_s nail','white nail','yellow nails']
        nresult = str(index[prediction])
        
        return render_template('nailpred.html',prediction=nresult)
        



""" Running our application """
if __name__ == "__main__":
    app.run(debug =True, port = 8080)