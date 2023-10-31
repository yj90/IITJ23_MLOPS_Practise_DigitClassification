**# Indian Institute of Technology Jodhpur MTECH AI | IITJ23_MLOPS_Practise_DigitClassification**

__#This is Practise Digit Classification__


System Requirements:
OS
Hardware

dummy commit to make the feature branch ahead of main branch 

e.g. 100 samples, 2 or binary class classification: image of carrot or turnip
      50 samples : carrots
      50 samples : turnips

      Data distrubution: balanced/uniform

      x amount of data for training 
      n-x amount of data for testing

      calculate some eval metric(train model (70 samples in training:35 carrots, 35 turnips), 30 samples in testing) == performance

In Practise:
     Train, Development, Test

     Train = Training the model(model type, model hyper params, model iteration)

     Dev = Selecting the model

     Test = Reporting the performance

How to set up:

install conda
conda create -n digits python=3.9
conda activate digits
pip install -r requirements.txt

How to Run:

Python exp.py


Feature:

Vary model hyper parameter

Meaning of Failure:

Poor performance metrics
Coding Runtime/Compile Error
The Model gavee wrong result on test sample during demo


Randomness In ML model:

1- When you create a split(means different training set & test set) so accuracy my have sliglty change

So in order to avoid this  we need to FREEZ the Data .i.e shuffle =False

2- Data Order (when learning is iterative means in DL model)

3- Weight initialization in the ML Model


HYPERPARAMTER TUNNING:(svm) 
  - take all the combinations of gamma and c
  - train model over all these parameter
  - get some performance metric on DEV test
  - test dataset should be completley unseen
  - select the hparams that yeilds the best performance on DEV set
  - so hyperparamter done only by looking at train and dev set only 
  - we dont have to touch test dataset while doing hyperparameter tunning

In our code hyperparameters are :
    - tain:dev:test split
    - gamma and choice of classifier

Hyperparameter meaning:
    - knob in our hand
    - manually tuunable parameter


    In machine learning, hyperparameters are settings or configurations that are not learned from the data but are set prior to training a machine learning model. These parameters are essential because they control the learning process but are not learned through the standard training procedures, which involve adjusting the model's weights or coefficients based on the data.

    Here are some common examples of hyperparameters in machine learning:

    Learning Rate: In optimization algorithms like gradient descent, the learning rate hyperparameter controls the step size at each iteration. It determines how quickly or slowly a model learns.

    Number of Hidden Layers and Units: In neural networks, you often specify the architecture of the network, including the number of hidden layers and the number of units (neurons) in each layer.

    Activation Functions: Choosing activation functions for neural network layers, such as sigmoid, ReLU (Rectified Linear Unit), or tanh, is another example of hyperparameter tuning.

    Batch Size: During training, models are often trained on batches of data rather than the entire dataset at once. The batch size hyperparameter determines how many data points are used in each training batch.

    Number of Trees (Ensemble Methods): In ensemble methods like Random Forest or Gradient Boosting, you specify the number of trees in the ensemble.

    Regularization Strength: Hyperparameters like L1 and L2 regularization coefficients control the amount of regularization applied to prevent overfitting.

    Kernel Type and Parameters (SVM): For Support Vector Machines, you might choose the kernel type (e.g., linear, polynomial, radial basis function) and associated hyperparameters.

    Depth and Splitting Criteria (Decision Trees): For decision trees, hyperparameters like maximum depth and splitting criteria (e.g., Gini impurity or entropy) are specified.

    Number of Clusters (Clustering Algorithms): In clustering algorithms like k-means, you need to specify the number of clusters in advance.

    Window Size (Time Series Analysis): In time series forecasting, the window size for moving averages or other time-based features can be considered a hyperparameter.

    Hyperparameter tuning is a critical part of the machine learning workflow. It often involves using techniques like grid search, random search, or Bayesian optimization to find the best set of hyperparameters that result in the optimal performance of the model on a validation dataset. Properly tuned hyperparameters can significantly impact the model's accuracy and generalization to new data.



