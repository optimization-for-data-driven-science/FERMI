import numpy as np
from sklearn.preprocessing import normalize, StandardScaler
import csv
from math import sqrt
import math
from sklearn.metrics import normalized_mutual_info_score
import pandas as pd


def grad_sigmoid(x):
    return np.exp(-x) / ((1.0 + np.exp(-x)) * (1.0 + np.exp(-x)))


def sigmoid(x):  # P(Y = 1 | X, \theta) Input: X\theta
    return 1.0 / (1.0 + np.exp(-x))


def loss1(predictions, labels):
    return (-labels * np.log(predictions) - (1 - labels) * np.log(1 - predictions)).mean()


with open('adult.data') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    y = []
    s1 = []
    s2 = []

    counter = 0
    i = 0
    for row in csv_reader:
        if counter > 9999:
            break

        if i == 0:
            i += 1
            continue

        if row[9] == "Male":
            s1.append(1)
        else:
            s1.append(0)

        if row[8] == "White":
            s2.append(1)
        else:
            s2.append(0)

        if row[14] == '>50K':
            y.append(1)
        else:
            y.append(0)

        counter += 1

    Y_ = np.array(y).reshape((len(y), 1))

P00 = 0
P01 = 0
P10 = 0
P11 = 0

S00 = []
S11 = []
S01 = []
S10 = []

S = []
for i in range(len(s1)):

    if s1[i] == 0 and s2[i] == 0:
        S.append(0)
    elif s1[i] == 0 and s2[i] == 1:
        S.append(1)
    elif s1[i] == 1 and s2[i] == 0:
        S.append(2)
    else:
        S.append(3)
    S11.append(s1[i] * s2[i])
    S10.append(s1[i] * (1 - s2[i]))
    S01.append((1 - s1[i]) * s2[i])
    S00.append((1 - s1[i]) * (1 - s2[i]))

S = np.array(S)
S = S[:, np.newaxis]

S00 = np.array(S00).reshape((len(s1), 1))
S01 = np.array(S01).reshape((len(s1), 1))
S10 = np.array(S10).reshape((len(s1), 1))
S11 = np.array(S11).reshape((len(s1), 1))

n00 = 0
n01 = 0
n10 = 0
n11 = 0

for i in range(len(s1)):
    if s1[i] == 0 and s2[i] == 0:
        P00 += 1
        n00 += 1

    elif s1[i] == 0 and s2[i] == 1:
        P01 += 1
        n01 += 1

    elif s1[i] == 1 and s2[i] == 1:
        P11 += 1
        n11 += 1

    else:
        P10 += 1
        n10 += 1

P00 /= len(s1)
P01 /= len(s1)
P10 /= len(s1)
P11 /= len(s1)

print(P00)  # Non-white Female
print(P01)  # White Female
print(P10)  # Non-White Male
print(P11)  # White Male

with open('adult.test') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    testY = []
    testS1 = []
    testS2 = []

    i = 0
    for row in csv_reader:
        if i == 0:
            i += 1
            continue

        if row[9] == "Male":
            testS1.append(1)
        else:
            testS1.append(0)

        if row[8] == "White":
            testS2.append(1)
        else:
            testS2.append(0)

        if row[14] == '>50K.':
            testY.append(1)
        else:
            testY.append(0)

train_data = pd
with open('AdultTrain2Sensitive.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    X = []
    i = 0
    columns = ''
    counter = 0

    for row in csv_reader:
        if counter > 9999:
            break

        if i == 0:
            i += 1
            columns = row
            continue

        new_row = []
        for item in row:
            new_row.append(float(item))

        new_row.append(1)  # intercept
        X.append(new_row)
        counter += 1

with open('AdultTest2Sensitive.csv') as csv_file:
    csv_reader = csv.reader(csv_file)

    testX = []
    i = 0
    columns = ''
    for row in csv_reader:
        if i == 0:
            i += 1
            columns = row
            continue

        new_row = []
        for item in row:
            new_row.append(float(item))

        new_row.append(1)  # intercept

        testX.append(new_row)

sc = StandardScaler()

X = np.array(X)
sc.fit(X)
X = sc.transform(X)

testX = sc.transform(testX)

testY = np.array(testY)
m, d = X.shape

print(S)

num_iterations = 5000
step_size = 0.0001
lam_list = [0,
            100,
            3000,
            30000,
            60000,
            120000,
            180000,
            300000,
            500000,
            900000,
            1200000,
            3000000,
            5000000,
            9000000,
            30000000,
            40000000,
            ]

number_of_sensitive_attributes = max(S)[0] + 1
n = X.shape[0]

for lam in [300000]:

    print("Lam: ", lam)
    theta = np.zeros((d, 1))

    # Computing the gradient of regularizer

    for iter_num in range(num_iterations):
        logits = np.dot(X, theta)
        probs = sigmoid(logits)
        grad_probs = grad_sigmoid(logits)

        g1 = np.dot(X.T, (probs - Y_))

        P_Y1 = sum(probs) / n
        P_Y0 = 1 - P_Y1

        D = np.diag(grad_probs.flatten())
        grad_Y1 = np.dot(D, X).sum(axis=0).reshape(d, 1)
        grad_Y1 /= n
        grad_Y0 = - grad_Y1

        regularizer_grad = np.zeros(theta.shape)

        for j in range(number_of_sensitive_attributes):
            indicator_function = (S == j) * 1

            number_of_s = sum(indicator_function)[0]
            P_S = number_of_s / n
            P_Y1S = np.dot(indicator_function.T, probs)[0][0] / number_of_s
            P_Y0S = 1 - P_Y1S

            q_1j = P_Y1S * sqrt(P_S) / sqrt(P_Y1)
            q_0j = P_Y0S * sqrt(P_S) / sqrt(P_Y0)

            # Computing the gradient with respect to theta

            conditional_grad_probs = np.multiply(indicator_function, grad_probs)

            D = np.diag(conditional_grad_probs.flatten())
            grad_Y1S = np.dot(D, X).sum(axis=0).reshape(d, 1)
            grad_Y1S /= number_of_s
            grad_Y0S = - grad_Y1S

            # Gradient of q_ij with respect to theta:
            grad_q1j = sqrt(P_S) / P_Y1 * (sqrt(P_Y1) * grad_Y1S - P_Y1S / (2 * sqrt(P_Y1)) * grad_Y1)
            grad_q0j = sqrt(P_S) / P_Y0 * (sqrt(P_Y0) * grad_Y0S - P_Y0S / (2 * sqrt(P_Y0)) * grad_Y0)

            regularizer_grad += 2 * q_1j * grad_q1j + 2 * q_0j * grad_q0j

        theta -= step_size * (g1 + lam * regularizer_grad)

    # ------------------------------------------------------------------------
    # Training
    testOut = np.dot(X, theta)
    testProbs = sigmoid(testOut)

    preds = testProbs >= 0.5
    acc = (preds == Y_).mean()
    print(acc)

    num_111 = 0
    num_011 = 0
    num_110 = 0
    num_010 = 0
    num_101 = 0
    num_001 = 0
    num_100 = 0
    num_000 = 0

    # Compute Fairness:
    for i in range(len(s1)):
        if s1[i] == 1 and s2[i] == 1 and preds[i]:
            num_111 += 1

        elif s1[i] == 1 and s2[i] == 1 and not preds[i]:
            num_011 += 1

        elif s1[i] == 1 and s2[i] == 0 and preds[i]:
            num_110 += 1

        elif s1[i] == 1 and s2[i] == 0 and not preds[i]:
            num_010 += 1

        elif s1[i] == 0 and s2[i] == 1 and preds[i]:
            num_101 += 1

        elif s1[i] == 0 and s2[i] == 1 and not preds[i]:
            num_001 += 1

        elif s1[i] == 0 and s2[i] == 0 and preds[i]:
            num_100 += 1

        else:
            num_000 += 1

    print(num_000, num_001, num_010, num_011, num_100, num_101, num_110, num_111)
    print(num_000 / n00, num_001 / n01, num_010 / n10, num_011 / n11, num_100 / n00, num_101 / n01, num_110 / n10,
          num_111 / n11)

    yhat = []

    for item in preds:
        if item[0]:
            yhat.append(1)
        else:
            yhat.append(0)

    S = []

    for i in range(len(s1)):
        if s1[i] == 0 and s2[i] == 0:
            S.append(0)

        elif s1[i] == 0 and s2[i] == 1:
            S.append(1)

        elif s1[i] == 1 and s2[i] == 0:
            S.append(2)

        else:
            S.append(3)

    num = normalized_mutual_info_score(S, yhat)
    print("NMI: ", num)
    # Renyi

    R = np.zeros((2, 4))
    pred1 = 0
    pred0 = 0

    P0S00 = 0
    P0S01 = 0
    P0S10 = 0
    P0S11 = 0
    P1S00 = 0
    P1S01 = 0
    P1S10 = 0
    P1S11 = 0

    for i in range(len(S)):
        if yhat[i] == 1:
            pred1 += 1
            if s1[i] == 0 and s2[i] == 0:
                P1S00 += 1
            elif s1[i] == 0 and s2[i] == 1:
                P1S01 += 1
            elif s1[i] == 1 and s2[i] == 0:
                P1S10 += 1
            else:
                P1S11 += 1
        else:
            pred0 += 1
            if s1[i] == 0 and s2[i] == 0:
                P0S00 += 1
            elif s1[i] == 0 and s2[i] == 1:
                P0S01 += 1
            elif s1[i] == 1 and s2[i] == 0:
                P0S10 += 1
            else:
                P0S11 += 1

    pred1 /= (len(S))
    pred0 /= (len(S))
    P0S00 /= (len(S))
    P0S01 /= (len(S))
    P0S10 /= (len(S))
    P0S11 /= (len(S))
    P1S00 /= (len(S))
    P1S01 /= (len(S))
    P1S10 /= (len(S))
    P1S11 /= (len(S))

    R[0][0] = P0S00 / math.sqrt(pred0 * P00)
    R[0][1] = P0S01 / math.sqrt(pred0 * P01)
    R[0][2] = P0S10 / math.sqrt(pred0 * P10)
    R[0][3] = P0S11 / math.sqrt(pred0 * P11)
    R[1][0] = P1S00 / math.sqrt(pred1 * P00)
    R[1][1] = P1S01 / math.sqrt(pred1 * P01)
    R[1][2] = P1S10 / math.sqrt(pred1 * P10)
    R[1][3] = P1S11 / math.sqrt(pred1 * P11)

    u, Vals, vh = np.linalg.svd(R)
    print(Vals)
    print("Test Phase:")

    # Test Phase

    Testn00 = 0
    Testn01 = 0
    Testn10 = 0
    Testn11 = 0

    TestP00 = 0
    TestP01 = 0
    TestP10 = 0
    TestP11 = 0

    for i in range(len(testS1)):
        if testS1[i] == 0 and testS2[i] == 0:
            TestP00 += 1
            Testn00 += 1

        elif testS1[i] == 0 and testS2[i] == 1:
            TestP01 += 1
            Testn01 += 1

        elif testS1[i] == 1 and testS2[i] == 1:
            TestP11 += 1
            Testn11 += 1

        else:
            TestP10 += 1
            Testn10 += 1

    TestP00 /= len(testS1)
    TestP01 /= len(testS1)
    TestP10 /= len(testS1)
    TestP11 /= len(testS1)

    testOut = np.dot(testX, theta)
    testProbs = sigmoid(testOut)

    preds = testProbs >= 0.5
    acc = (preds == testY).mean()
    print("Test Acc: ", acc)

    num_111 = 0
    num_011 = 0
    num_110 = 0
    num_010 = 0
    num_101 = 0
    num_001 = 0
    num_100 = 0
    num_000 = 0

    # Compute Fairness:
    for i in range(len(testS1)):
        if testS1[i] == 1 and testS2[i] == 1 and preds[i]:
            num_111 += 1

        elif testS1[i] == 1 and testS2[i] == 1 and not preds[i]:
            num_011 += 1

        elif testS1[i] == 1 and testS2[i] == 0 and preds[i]:
            num_110 += 1

        elif testS1[i] == 1 and testS2[i] == 0 and not preds[i]:
            num_010 += 1

        elif testS1[i] == 0 and testS2[i] == 1 and preds[i]:
            num_101 += 1

        elif testS1[i] == 0 and testS2[i] == 1 and not preds[i]:
            num_001 += 1

        elif testS1[i] == 0 and testS2[i] == 0 and preds[i]:
            num_100 += 1

        else:
            num_000 += 1

    print(num_000, num_001, num_010, num_011, num_100, num_101, num_110, num_111)
    print(num_000 / n00, num_001 / n01, num_010 / n10, num_011 / n11, num_100 / n00, num_101 / n01, num_110 / n10,
          num_111 / n11)

    yhat = []

    for item in preds:
        if item[0]:
            yhat.append(1)
        else:
            yhat.append(0)

    S = []

    for i in range(len(testS1)):
        if testS1[i] == 0 and testS2[i] == 0:
            S.append(0)

        elif testS1[i] == 0 and testS2[i] == 1:
            S.append(1)

        elif testS1[i] == 1 and testS2[i] == 0:
            S.append(2)

        else:
            S.append(3)

    num = normalized_mutual_info_score(S, yhat)
    print("NMI: ", num)
    # Renyi

    R = np.zeros((2, 4))
    pred1 = 0
    pred0 = 0

    P0S00 = 0
    P0S01 = 0
    P0S10 = 0
    P0S11 = 0
    P1S00 = 0
    P1S01 = 0
    P1S10 = 0
    P1S11 = 0

    for i in range(len(S)):
        if yhat[i] == 1:
            pred1 += 1
            if testS1[i] == 0 and testS2[i] == 0:
                P1S00 += 1
            elif testS1[i] == 0 and testS2[i] == 1:
                P1S01 += 1
            elif testS1[i] == 1 and testS2[i] == 0:
                P1S10 += 1
            else:
                P1S11 += 1
        else:
            pred0 += 1
            if testS1[i] == 0 and testS2[i] == 0:
                P0S00 += 1
            elif testS1[i] == 0 and testS2[i] == 1:
                P0S01 += 1
            elif testS1[i] == 1 and testS2[i] == 0:
                P0S10 += 1
            else:
                P0S11 += 1

    pred1 /= (len(S))
    pred0 /= (len(S))
    P0S00 /= (len(S))
    P0S01 /= (len(S))
    P0S10 /= (len(S))
    P0S11 /= (len(S))
    P1S00 /= (len(S))
    P1S01 /= (len(S))
    P1S10 /= (len(S))
    P1S11 /= (len(S))

    R[0][0] = P0S00 / math.sqrt(pred0 * TestP00)
    R[0][1] = P0S01 / math.sqrt(pred0 * TestP01)
    R[0][2] = P0S10 / math.sqrt(pred0 * TestP10)
    R[0][3] = P0S11 / math.sqrt(pred0 * TestP11)
    R[1][0] = P1S00 / math.sqrt(pred1 * TestP00)
    R[1][1] = P1S01 / math.sqrt(pred1 * TestP01)
    R[1][2] = P1S10 / math.sqrt(pred1 * TestP10)
    R[1][3] = P1S11 / math.sqrt(pred1 * TestP11)

    u, Vals, vh = np.linalg.svd(R)
    print(Vals)
    print("----------------------------------")
