from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np
import cv2

np.random.seed(0)
NEURAL_SHAPE = [784, 16, 16, 10]

def load_dataset():
    try:
        data = pd.read_pickle("mnist_data.pickle")
        target = pd.read_pickle("mnist_target.pickle")
    except FileNotFoundError:
        ### MNIST database is stored as a dictionary with keys "data" and "target".
        ### "data"   -> Pandas DataFrame where each row represents a 28 x 28 image, and where the value of each of the 784 pixels is stored in a
        ###             a column
        ### "target" -> Each row corresponds to the digit shown in the corresponding row/image of "data"
        mnist = fetch_openml("mnist_784")
        data = mnist["data"]
        target = mnist["target"]

        data.to_pickle("mnist_data.pickle")
        target.to_pickle("mnist_target.pickle")
    return data, target

def get_weights(neural_shape):
    weights = [np.zeros((1, neural_shape[0]))]
    weights = weights + [np.random.rand(neural_shape[i], neural_shape[i+1]) for i in range(len(neural_shape) - 1)]
    return weights

def get_biases(neural_shape):
    biases = [np.zeros(neural_shape[0])]
    biases = biases + [np.random.rand(n) for n in neural_shape[1:]]
    return biases

### Convert row of MNIST DataFrame into 28x28 NumPy array
def get_image(series):
    im = np.zeros((28, 28), dtype=np.uint8)
    px = 1
    for i in range(28):
        for j in range(28):
            im[i, j] = series[f"pixel{px}"]
            px += 1
    return im

def display_image(im):
    im_display = cv2.resize(im, (224, 224))
    cv2.imshow("im", im_display)
    cv2.waitKey()

# Load dataset
data, target = load_dataset()

# Define training and testing datapoints
data_training = data[:60000]
target_training = target[:60000]

data_testing = data[60000:70000]
target_testing = target[60000:70000]

# Initialize weights and biases
weights = get_weights(NEURAL_SHAPE)
biases = get_biases(NEURAL_SHAPE)