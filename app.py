from flask import Flask,render_template,request,jsonify
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    cp = float(request.form['cp'])
    trestbps = float(request.form['trestbps'])
    chol = float(request.form['chol'])
    restecg = float(request.form['restecg'])
    thalach = float(request.form['thalach'])

    arr = np.array([[cp,trestbps,chol,restecg,thalach]])
    pred = model.predict(arr)
    output = np.round(pred,decimals=2)
    #output=round(pred[0],2)


    return render_template('after.html', data = output)


if __name__ == "__main__":
    app.run(debug=True)
