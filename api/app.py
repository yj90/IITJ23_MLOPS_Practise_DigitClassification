from flask import Flask, request, jsonify
from joblib import load

app = Flask(__name__)

@app.route('/hello/<name>')
def index(name):
    return "Hello, "+name+"!"

@app.route('/model', methods=['POST'])
def pred_model():
    js = request.get_json()
    image1 = js['image1']
    image2 = js['image2']
    #Assuming this is the path of our best trained model
    model = load('/Users/pranjal/Desktop/ml-ops/models/svmgamma:0.001_C:1.joblib')
    prediction_image_1 = model.predict(image1)
    prediction_image_2 = model.predict(image2)
    if(prediction_image_1 == prediction_image_2):
        return "True"
    else:
        return "False"