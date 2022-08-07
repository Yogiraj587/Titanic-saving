import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('ML/Reservation Saving/Reservation_Saving.pkl','rb'))

@app.route("/")

def home():
    return render_template('body.html')

@app.route('/predict',methods=["POST"])

def predict():
    if request.method == 'POST':
        Pclass = int(request.form['Pclass'])
        Age = int(request.form['Age'])
        Fare = int(request.form['Fare'])
        Sex =  int(request.form["Sex"])

        arr = np.array([[Pclass,Age,Fare,Sex]])
        pred=model.predict(arr)
    return render_template("predicted.html",pred=pred)

if __name__ =="__main__":
    app.run(debug=True)