import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_error(X_train, y_train, w, b):
    m = X_train.shape[0]

    cost = 0.0

    for i in range(m):
        z = np.dot(X_train[i], w) + b
        f = sigmoid(z)
        cost += y_train[i] * np.log(f) + (1 - y_train[i]) * np.log(1 - f)
    
    return -cost / m

def calculate_gradient(X_train, y_train, w, b):
    m = X_train.shape[0]
    dj_dw = np.zeros(w.shape[0])
    dj_db = 0

    for i in range(m):
        z = np.dot(X_train[i], w) + b
        f = sigmoid(z)
        diff = f - y_train[i]

        for j in range(w.shape[0]):
            # jth parameter in ith data row
            dj_dw[j] += diff * X_train[i, j]
        dj_db += diff
    
    return dj_dw / m, dj_db / m

def gradient_descent(x_train, y_train, w, b, alpha):
    dj_dw, dj_db = calculate_gradient(x_train, y_train, w, b)
    return w - alpha * dj_dw, b - alpha * dj_db


def main():
    X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
    y_train = np.array([0, 0, 0, 1, 1, 1])

    
    itns = 10000
    alpha = 1e-1
    epsilon = 1e-6
    error_history = []

    w, b = gradient_descent(X_train, y_train, np.zeros(X_train.shape[1]), 0, alpha)
    error_history.append(logistic_error(X_train, y_train, w, b))

    for _ in range(itns):
        w, b = gradient_descent(X_train, y_train, w, b, alpha)
        sse = logistic_error(X_train, y_train, w, b)
        sse_diff = abs(sse - error_history[-1])

        if sse_diff < epsilon:
            break
        error_history.append(sse)

    print(f"logistic regression eqn.: y=1/(1+e^-({w}x+{b}))")
    plt.title("Learning Curve")
    plt.xlabel("Iterations")
    plt.ylabel("J(w,b)")
    plt.plot(np.arange(len(error_history)) + 1, error_history)
    plt.show()

if __name__ == "__main__":
    main()