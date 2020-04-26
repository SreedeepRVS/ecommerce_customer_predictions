from flask import Flask,render_template,url_for,request
from flask_bootstrap import Bootstrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pickle


app=Flask(__name__)

Bootstrap(app)
filename='model/ecom_model.h5'
model=pickle.load(open(filename,'rb'))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    if request.method=='POST':
        params=[np.double(x) for x in request.form.values()]
        input_params=[np.array(params)]
        prediction=model.predict(input_params)
        output=round(prediction[0],2)
        return render_template("index.html",tag='Yearly Amount Spent: $',prediction_text=output)





if __name__=='__main__':
    app.run(debug=True)
