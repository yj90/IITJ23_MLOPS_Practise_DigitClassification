
# Import datasets, classifiers and performance metrics
from sklearn import svm,datasets
from sklearn.model_selection import train_test_split
<<<<<<< HEAD
from sklearn import svm, tree, datasets, metrics
from joblib import dump, load
# we will put all utils here

def get_combinations(param_name, param_values, base_combinations):    
    new_combinations = []
    for value in param_values:
        for combination in base_combinations:
            combination[param_name] = value
            new_combinations.append(combination.copy())    
    return new_combinations

def get_hyperparameter_combinations(dict_of_param_lists):    
    base_combinations = [{}]
    for param_name, param_values in dict_of_param_lists.items():
        base_combinations = get_combinations(param_name, param_values, base_combinations)
    return base_combinations

def tune_hparams(X_train, y_train, X_dev, y_dev, h_params_combinations, model_type="svm"):
    best_accuracy = -1
    best_model_path = ""
    for h_params in h_params_combinations:
        # 5. Model training
        model = train_model(X_train, y_train, h_params, model_type=model_type)
        # Predict the value of the digit on the test subset        
        cur_accuracy, _, _ = predict_and_eval(model, X_dev, y_dev)
        if cur_accuracy > best_accuracy:
            best_accuracy = cur_accuracy
            best_hparams = h_params
            best_model_path = "./models/{}_".format(model_type) +"_".join(["{}:{}".format(k,v) for k,v in h_params.items()]) + ".joblib"
            best_model = model

    # save the best_model    
    dump(best_model, best_model_path) 


    return best_hparams, best_model_path, best_accuracy 


=======
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from joblib import dump,load
>>>>>>> main

#read gigits
def read_digits():
    digits = datasets.load_digits()
    x = digits.images
    y = digits.target 
    return x,y

# We will define utils here :
def preprocess_data(data):
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    return data

# Split data into 50% train and 50% test subsets

def split_data(X,y,test_size=0.5,random_state=1):
    X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=test_size, shuffle=False
    )
    return X_train, X_test, y_train, y_test

# Create a classifier: a support vector classifier
def train_model(X, y, model_params,model_type = 'svm'):
    if model_type == 'svm':
        clf = svm.SVC(**model_params)
    if model_type == 'tree':
        clf = tree.DecisionTreeClassifier(**model_params)
    clf.fit(X, y)
    return clf

def split_train_dev_test(X, y, test_size, dev_size):
    X_train_dev, X_test, y_train_dev, y_test = split_data(X, y, test_size=test_size)
    X_train, X_dev, y_train, y_dev = split_data(X_train_dev, y_train_dev, test_size=dev_size/(1-test_size))
    return X_train, X_test,X_dev, y_train, y_test, y_dev

def predict_and_eval(model, X, y):
    
    predicted = model.predict(X)
    accuracy = accuracy_score(y, predicted)

<<<<<<< HEAD
# Question 2:
def predict_and_eval(model, X_test, y_test):
    predicted = model.predict(X_test)
    return metrics.accuracy_score(y_test, predicted), metrics.f1_score(y_test, predicted, average="macro"), predicted
=======
    return accuracy

def tune_hparams(X_train, Y_train, X_dev, y_dev, list_of_all_param_combination, model_type='svm'):
    best_accuracy_so_far = -1
    best_model = None
    best_model_path = ""

    for param_combination in list_of_all_param_combination:
        if model_type == 'svm':
            cur_model = train_model(X_train, Y_train, {'gamma': param_combination['gamma'],'C':param_combination['C']}, model_type='svm')
        if model_type == 'tree':
            cur_model = train_model(X_train, Y_train, {'max_depth': param_combination['max_depth']}, model_type='tree')

        cur_accuracy = predict_and_eval(cur_model, X_dev, y_dev)
        if cur_accuracy > best_accuracy_so_far:
            best_accuracy_so_far = cur_accuracy
            if model_type == 'svm':
                optimal_gamma = param_combination['gamma']
                optimal_C = param_combination['C']
                best_hparams = {'gamma': optimal_gamma,'C':optimal_C}
            if model_type == 'tree':
                optimal_max_depth = param_combination['max_depth']
                best_hparams = {'max_depth': optimal_max_depth}
            best_model_path = "./models/{}".format(model_type)+"_".join(["{}:{}".format(k,v) for k,v in best_hparams.items()])+".joblib"

            best_model = cur_model

    # save the best model
    dump(best_model,best_model_path)
    #best_model_path = 
    return best_hparams, best_model_path, best_accuracy_so_far
>>>>>>> main
