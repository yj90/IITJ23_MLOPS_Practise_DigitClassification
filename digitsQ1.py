# Import necessary libraries
from sklearn.preprocessing import StandardScaler
from utils import preprocess_data, tune_hparams, split_train_dev_test, read_digits, predict_and_eval
from joblib import load
import argparse
from sklearn import metrics

# 1. Data Loading
x, y = read_digits()

parser = argparse.ArgumentParser()

parser.add_argument("--runs", default=5, help="number of runs")
parser.add_argument("--test_sizes", default="0.1,0.2", help="comma separated value of test sizes")
parser.add_argument("--dev_sizes", default="0.1,0.2", help="comma separated value of dev sizes")
parser.add_argument("--prod", default="svm", help="model to be used for production")
parser.add_argument("--candidate", default="tree", help="model to be used as candidate")

args = parser.parse_args()

max_runs = int(args.runs)
test_sizes = args.test_sizes.split(',')
test_sizes = [float(i) for i in test_sizes]
dev_sizes = args.dev_sizes.split(',')
dev_sizes = [float(i) for i in dev_sizes]
models = []
models.append(args.prod)
models.append(args.candidate)

results = []

for i in range(max_runs):
    for test_size in test_sizes:
        for dev_size in dev_sizes:
            # 3. Data splitting
            X_train, X_test, X_dev, y_train, y_test, y_dev = split_train_dev_test(x, y, test_size=test_size, dev_size=dev_size)

            # Reshape the image data to 2D arrays
            X_train = X_train.reshape((X_train.shape[0], -1))
            X_test = X_test.reshape((X_test.shape[0], -1))
            X_dev = X_dev.reshape((X_dev.shape[0], -1))

            # 4. Data Preprocessing
            scaler = StandardScaler()

            # Fit the scaler on the training data and transform training, validation, and test data
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            X_dev = scaler.transform(X_dev)

            classifer_hparam = {}

            gama_ranges = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
            C_ranges = [0.1, 1, 2, 5, 10]
            classifer_hparam['svm'] = [{'gamma': gamma, 'C': C} for gamma in gama_ranges for C in C_ranges]

            max_depth = [5, 10, 15, 20, 50, 100]
            classifer_hparam['tree'] = [{'max_depth': depth} for depth in max_depth]

            # Predict the value of the digit on the test subset
            # 6.Predict and Evaluate
            for model in models:
                best_hparams, best_model_path, best_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev, classifer_hparam[model], model_type=model)
                best_model = load(best_model_path)

                accuracy_test, predicted_test = predict_and_eval(best_model, X_test, y_test)
                accuracy_dev, _ = predict_and_eval(best_model, X_dev, y_dev)
                accuracy_train, _ = predict_and_eval(best_model, X_train, y_train)

                print("Production accuracy " + f" model={model} run_index={i} test_size={test_size} dev_size={dev_size} train_size={1 - (dev_size + test_size)} train_acc={accuracy_train} dev_acc={accuracy_dev} test_acc={accuracy_test}" if args.prod == model else "Candidate Accuracy" + f" model={model} run_index={i} test_size={test_size} dev_size={dev_size} train_size={1 - (dev_size + test_size)} train_acc={accuracy_train} dev_acc={accuracy_dev} test_acc={accuracy_test}")
                disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted_test)
                disp.figure_.suptitle("Confusion Matrix")
                print(f"Confusion matrix:\n{disp.confusion_matrix}")
                # results.append([{'model': model, 'run_index': i, 'test_size': test_size, 'dev_size': dev_size, 'train_size': 1 - (dev_size + test_size), 'train_acc': accuracy_train, 'dev_acc': accuracy_dev, 'test_acc': accuracy_test}])
            # print(f"best_gamma={best_hparams['gamma']},best_C={best_hparams['C']}")
