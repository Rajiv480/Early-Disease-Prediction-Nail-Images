
import numpy as np
import os
from flask import Flask, app,request,render_template
from tensorflow.keras import models
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.python.ops.gen_array_ops import concat
from tensorflow.keras.applications.inception_v3 import preprocess_input
import requests
from flask import Flask, request, render_template, redirect, url_for


#Loading the model

modeln=load_model("Vgg-16-nail-disease.h5")


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
        filepath=os.path.join(basepath,'uploads',f.filename) 
        f.save(filepath)
        img=image.load_img(filepath,target_size=(224,224))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        img_data=preprocess_input(x)
        prediction=np.argmax(modeln.predict(img_data))
        
        index=['Darier_s disease', 'Muehrck-e_s lines', 'aloperia areata', 'beau_s lines', 'bluish nail',
               'clubbing','eczema','half and half nailes (Lindsay_s nails)','koilonychia','leukonychia',
               'onycholycis','pale nail','red lunula','splinter hemmorrage','terry_s nail','white nail','yellow nails']
        nresult = str(index[prediction])
        nresult        
        
        return render_template('nailpred.html',prediction=nresult)
        



""" Running our application """
if __name__ == "__main__":
    app.run(debug =False, port = 8080)
