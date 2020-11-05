import numpy as np
from numpy import array
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd


def one_hot_encode(arr):
    values = array(arr)
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded


def compute_softmax(W_matrix, X_matrix):
    y_values = np.dot(X_matrix, W_matrix)  # n * c

    y_values = y_values - np.max(y_values, axis=1, keepdims=True)
    exps = np.exp(y_values)

    softmax = exps / exps.sum(axis=1)[:, None]

    return softmax


def loss_function(soft_labels, actual_one_hot_encoded_label):
    return - np.sum(actual_one_hot_encoded_label * np.log(soft_labels + 1e-6))


def loss_grad(W_matrix, X_matrix, labels):
    p_y_given_x = compute_softmax(W_matrix, X_matrix)
    d_y = labels - p_y_given_x
    return np.dot(X_matrix.T, d_y)


# Read Data
sc = StandardScaler()

test_data = pd.read_csv('drive_test.csv')
train_data = pd.read_csv('drive_train.csv')
X = train_data.drop(['48'], axis=1)
X = X.to_numpy()

sc.fit(X)

X = sc.transform(X)

original_Y = list(train_data['48'])
Y = one_hot_encode(original_Y)

n = X.shape[0]
d = X.shape[1]
c = Y.shape[1]

W = np.zeros((d, c))

number_of_classes = c

number_of_iterations = 10000
# lam = 1
step_size = 0.001
print("---------------------")
for _ in range(number_of_iterations):
    loss_function_grad = loss_grad(W, X, Y)

    total_grad = loss_function_grad

    W -= step_size * total_grad


test_data = pd.read_csv('drive_test.csv')
X_test = test_data.drop(['48'], axis=1)
X_test = X_test.to_numpy()

X_test = sc.transform(X_test)

Y_test = test_data[['48']].to_numpy()

soft_predictions = compute_softmax(W, X_test)

number_of_tests = X_test.shape[0]
hard_predictions = np.argmax(soft_predictions, axis=1) + 1

hard_predictions = np.array(hard_predictions).reshape((number_of_tests, 1))
print(hard_predictions)
print(Y_test)

check = Y_test == hard_predictions
print(sum(check) / number_of_tests)
