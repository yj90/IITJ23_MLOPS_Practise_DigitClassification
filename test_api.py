from api.app import app
import pytest
from utils import read_digits, preprocess_data

# Basic example testing
def test_get_root():
    response = app.test_client().get("/")
    assert response.status_code == 200
    assert response.get_data() == b"Hello, World"

def test_post_root():
    suffix = "post suffix"
    response = app.test_client().post("/", json={"suffix": suffix})
    assert response.status_code == 200
    assert response.get_json()['op'] == "Hello, World POST " + suffix

def get_processed_data():
    X, y = read_digits()
    X = X[:100, :, :]
    y = y[:100]
    X = preprocess_data(X)
    return X, y

def test_predict_svm():
    X, y = get_processed_data()
    response = app.test_client().post("/predict/svm", json={"image": X[9].tolist()})
    assert response.status_code == 200
    assert response.get_json()['prediction'] == [2]

def test_predict_tree():
    X, y = get_processed_data()
    response = app.test_client().post("/predict/tree", json={"image": X[0].tolist()})
    assert response.status_code == 200
    assert response.get_json()['prediction'] == [0]

def test_predict_lr():
    X, y = get_processed_data()
    response = app.test_client().post("/predict/lr", json={"image": X[0].tolist()})
    assert response.status_code == 200
    assert response.get_json()['prediction'] == [0]
