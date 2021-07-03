#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("covid19_model.sav")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def result():
    s_length = request.form['sepal_length']
    s_width = request.form['sepal_width']

    
    pred = model.predict([[s_length, s_width]])
    
    
    return render_template("index.html", result = pred[0])


app.run()
   


# In[ ]:




