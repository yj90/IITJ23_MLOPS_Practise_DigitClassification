from flask import Flask, request, jsonify
from joblib import load
import os
from markupsafe import escape

app = Flask(__name__)

def load_model(model_name):
    models_path = '/home/yjlinuxubu/mlops23/IITJ23_MLOPS_Practise_DigitClassification/models'

    if model_name == 'svm':
        filename = os.path.join(models_path, 'svm_gamma_0.0001_C_10.joblib')
    elif model_name == 'tree':
        filename = os.path.join(models_path, 'tree_max_depth_15.joblib')
    elif model_name == 'lr':
        filename = os.path.join(models_path, 'M22AIE236_lr_lbfgs.joblib')
    else:
        return None

    return load(filename)

@app.route('/')
def index():
    return "Hello, World"

@app.route("/", methods=["POST"])
def hello_world_post():    
    return {"op": "Hello, World POST " + request.json["suffix"]}

@app.route('/predict/<model_name>', methods=['POST'])
def pred_model(model_name):
    js = request.get_json()
    model_name = escape(model_name)
    
    # Assuming this is the path of our best trained model
    model = load_model(model_name)
    
    if model is None:
        return jsonify(error="Invalid model name"), 400

    image1 = [js['image']]
    pred1 = model.predict(image1)
    
    # Return pred1 in json
    return jsonify(prediction=pred1.tolist())

if __name__ == '__main__':
    app.run(debug=True)
