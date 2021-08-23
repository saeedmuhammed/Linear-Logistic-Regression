# import Libraries.
import numpy as np;
import pandas as pd;
import scipy.optimize as opt;


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Cost function
def costFunction(x, y, theta):
    z = np.power(((x * theta.T) - y), 2);
    return np.sum(z) / (2 * len(x));


# ----------------------------------------------------------------------------------------------------

# Gradient Descent
def gD(x, y, theta, alpha, iter):
    temp = np.matrix(np.zeros(theta.shape));
    thetas = int(theta.ravel().shape[1]);  # number of thetas
    cost = np.zeros(iter);  # cost of all iterations

    for i in range(iter):
        error = (x * theta.T) - y;
        for j in range(thetas):
            term = np.multiply(error, x[:, j]);  # error * Xi
            temp[0, j] = theta[0, j] - ((alpha / len(x) * np.sum(term)));  # all function
        theta = temp;
        cost[i] = costFunction(x, y, theta);
    return theta, cost;


# ----------------------------------------------------------------------------------------------------

def sigmoid(z):
    return (1 / (1 + np.exp(-z)));


# ----------------------------------------------------------------------------------------------------

def costFn(thetav, X, Y):
    thetav = np.matrix(thetav);
    X = np.matrix(X);
    Y = np.matrix(Y);
    first = np.multiply(-Y, np.log(sigmoid(X * thetav.T)));
    second = np.multiply((1 - Y), np.log(1 - sigmoid(X * thetav.T)));
    return np.sum(first - second) / (len(X));


# ----------------------------------------------------------------------------------------------------

def gradient(thetav, X, Y):
    thetav = np.matrix(thetav);
    X = np.matrix(X);
    Y = np.matrix(Y);
    parameters = int(thetav.ravel().shape[1]);
    grad = np.zeros(parameters);
    error = sigmoid(X * thetav.T) - Y;  # Error Calculation -> for Every Iteration.
    for i in range(parameters):
        term = np.multiply(error, X[:, i]);
        grad[i] = np.sum(term) / len(X);
    return grad;


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------

def predict(thetav, X):
    probability = sigmoid(X * thetav.T);
    return [1 if X >= 0.5 else 0 for X in probability];


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------
check = False;
while (not check):

    # Get Value Form User as Input To Select Needed Operation
    choice = input(
        "1- Linear Regression For one variable. \n2- Linear Regression For multi variable.\n3- Logositic Regression For Multi vairable.\n");
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # Data Files Paths
    linearPath = 'house_data.csv';
    logisticePath = 'heart.csv';
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # Choice = 1 -> Linear Regression For One Variable.
    if (choice == '1'):

        # Read Data From CSV File
        data = pd.read_csv(linearPath, header=0, usecols=['sqft_living', 'price'])

        # Rescaling data (Feature Scaling) -> (value - mean) divided by whole Count.
        data = (data - data.mean()) / data.std();
        data.insert(1, 'ones', 1);  # For Matrixing Data and in Form of Matrix We Add Column of Ones.

        # Separating X (Training Set) from Y (Target Variable).
        cols = data.shape[1];
        y = data.iloc[:, 0:cols - 2];
        x = data.iloc[:, cols - 2:cols];

        # Matrix X and Y To make -> Convert From Data Frames into numpy Matrices
        x = np.matrix(x.values);
        y = np.matrix(y.values);
        theta = np.matrix(np.array([0, 0]));

        # Run Program
        alpha = 0.01;
        iter = 100;
        lastTheta, cost = gD(x, y, theta, alpha, iter);
        print(lastTheta);
        choice2 = int(input("enter the squft_living to predict the price \n"))

        input = np.matrix(np.array([1,choice2]))
        output = input.dot(lastTheta.T)
        print("The price is", output);

        # Variable To Close Program.
        check = True;

    # Choice = 2 -> Linear Regression For Multi Variable.
    elif (choice == '2'):

        # Read Data From CSV File for Multi X.
        data = pd.read_csv(linearPath, header=0, usecols=['price', 'sqft_living', 'bathrooms', 'grade', 'lat', 'view'])

        # Rescaling Data (Feature Scaling) -> (value - mean) divided by whole Count.
        data = (data - data.mean()) / data.std();
        data.insert(1, 'ones', 1);  # For Matrixing Data and in Form of Matrix We Add Column of Ones.

        # Separating X (Training Set) from Y (Target Variable).
        cols = data.shape[1];
        y = data.iloc[:, 0: cols - 6];
        x = data.iloc[:, cols - 6:cols];

        # Matrix X and Y To make -> Convert From Data Frames into numpy Matrices
        x = np.matrix(x.values);
        y = np.matrix(y.values);
        theta = np.matrix(np.array([0, 0, 0, 0, 0, 0]));

        # Run Program
        alpha = 0.01;
        iter = 100;
        lastTheta, cost = gD(x, y, theta, alpha, iter);
        print(lastTheta);


        data1 = int(input("enter the squft_living to predict the price \n"))
        data2 = int(input("enter the bathrooms to predict the price \n"))
        data3 = int(input("enter the grade to predict the price \n"))
        data4 = int(input("enter the lat to predict the price \n"))
        data5 = int(input("enter the view to predict the price \n"))

        input = np.matrix(np.array([1, data1,data2,data3,data4,data5]))
        output = input.dot(lastTheta.T)
        print("The price is", output);

        # Variable To Close Program.
        check = True;

    # Choice = 3 -> Logistic Regression For Mutli Variable.
    elif (choice == '3'):

        # Read Data From CSV File for 4 Predicators (X) and Classification Variable (Y).
        data = pd.read_csv(logisticePath, header=0, usecols=['target', 'trestbps', 'chol', 'thalach', 'oldpeak']);

        # Classify Data into Two Category either Positive or Negative
        positive = data[data['target'].isin([1])];
        negative = data[data['target'].isin([0])];

        # Add a ones column - this makes the matrix multiplication work out easier
        data.insert(4, 'Ones', 1);

        # set X (training data) and y (target variable)
        cols = data.shape[1];
        y = data.iloc[:, cols - 1: cols];
        x = data.iloc[:, 0: cols - 1];

        # convert to numpy arrays and initalize the parameter array theta
        x = np.array(x.values);
        y = np.array(y.values);
        theta = np.zeros(5);  # For n = 4 -> Number of Thetas = n + 1;
        cost = costFn(theta, x, y);

        # get Min of cost using fprime fn (gradient) with initial value thetas (X0) and arguments x,y.
        result = opt.fmin_tnc(func=costFn, x0=theta, fprime=gradient, args=(x, y));

        # Get Cost After Optimization
        costAfteroptimization = costFn(result[0], x, y);

        # Predict Result from the output Value;
        theta_min = np.matrix(result[0]);
        predictions = predict(theta_min, x);
        correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)];
        accuracy = (sum(map(int, correct)) % len(correct));
        print("Number Of Correct Predictions are", accuracy);
        accuracy = (accuracy / len(correct)) * 100;
        print("With Percent: ", accuracy);
        print("Last Theta = ", theta_min);

        data11 = int(input("enter the trestbps \n"))
        data21 = int(input("enter the chol  \n"))
        data31 = int(input("enter the thalach \n"))
        data41 = int(input("enter the oldpeak  \n"))

        input = np.matrix(np.array([1, data11, data21, data31, data41]))
        output = predict(theta_min,input)
        if(output == 0):
            print("patient don't have heart disease")
        else:
            print("patient have heart disease")

        check = True;

    # Choice = None Of The Above So We Repeat
    else:
        print("Invalid Input For Operation....\n");
        check = False;
# ----------------------------------------------------------------------------------------------------
