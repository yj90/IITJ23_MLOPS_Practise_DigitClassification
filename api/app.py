from flask import Flask, request, jsonify
from joblib import load
import os
from markupsafe import escape


app = Flask(__name__)

def load_model(model_name):
    dirname = os.path.dirname(__file__)
    filename_svm = os.path.join(dirname, '../models/svmgamma:0.0001_C:1.joblib')
    svm = load(filename_svm)
    filename_tree = os.path.join(dirname, '../models/tree_max_depth:100.joblib')
    tree = load(filename_tree)
    filename_lr = os.path.join(dirname, '../models/M22AIE236_lr_lbfgs.joblib')
    lr = load(filename_lr)
    if model_name == 'svm':
        return svm
    elif model_name == 'tree':
        return tree
    elif model_name == 'lr':
        return lr
    else:
        return None

@app.route('/')
def index():
    return "Hello, World"

@app.route("/", methods=["POST"])
def hello_world_post():    
    return {"op" : "Hello, World POST " + request.json["suffix"]}

@app.route('/predict/<model_name>', methods=['POST'])
def pred_model(model_name):
    js = request.get_json()
    model_name = escape(model_name)
    image1 = [js['image']]
    #Assuming this is the path of our best trained model
    model = load_model(model_name)
    pred1 = model.predict(image1)
    #reurn pred1 in json
    return jsonify(prediction=pred1.tolist())
    
if __name__ == '__main__':
    app.run(debug=True)
