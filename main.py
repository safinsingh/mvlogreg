import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def inv_sigmoid(sig):
    return -np.log(1/sig - 1)

def logistic_error(X_train, y_train, w, b, lambda_):
    m = X_train.shape[0]
    z = np.dot(X_train, w) + b
    f = sigmoid(z)
    cost = -np.mean(y_train * np.log(f) + (1 - y_train) * np.log(1 - f))
    regularization_cost = (lambda_ / (2 * m)) *  np.sum(w ** 2)

    return cost + regularization_cost

def calculate_gradient(X_train, y_train, w, b, lambda_):
    m = X_train.shape[0]
    z = np.dot(X_train, w) + b
    f = sigmoid(z)
    diff = f - y_train

    dj_dw = (np.dot(X_train.T, diff) + lambda_ * w) / m
    dj_db = np.mean(diff)
    
    return dj_dw, dj_db

def gradient_descent(X_train, y_train, w, b, alpha, lambda_, num_iters, epsilon):
    error_history = []

    for _ in range(num_iters):
        dj_dw, dj_db = calculate_gradient(X_train, y_train, w, b, lambda_)

        w -= alpha * dj_dw
        b -= alpha * dj_db
        
        error = logistic_error(X_train, y_train, w, b, lambda_)
        if error_history and abs(error - error_history[-1]) < epsilon:
            break
        error_history.append(error)
    
    return w, b, error_history


def main():
    X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
    y_train = np.array([0, 0, 0, 1, 1, 1])

    num_iters = 10000
    alpha = 1e-1
    epsilon = 1e-6
    threshold = 0.5
    lambda_ = 0.7

    w = np.zeros(X_train.shape[1])
    b = 0
    w, b, error_history = gradient_descent(X_train, y_train, w, b, alpha, lambda_, num_iters, epsilon)

    print(f"logistic regression eqn.: y=1/(1+e^-({w}x+{b}))")
    plt.title("Learning Curve")
    plt.xlabel("Iterations")
    plt.ylabel("J(w,b)")
    plt.plot(np.arange(len(error_history)) + 1, error_history)
    plt.show()

    # Plot threshold line
    z = inv_sigmoid(threshold)
    plt.title("Logistic Regression")
    plt.xlabel("x_1")
    plt.ylabel("x_2")
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
    # For two features:
    #   z = w_1 * x_1 + w_2 * x_2 + b
    #   0 = w_1 * x_1 + w_2 * x_2 + b - z
    #   -(w_2 * x_2) = w_1 * x_1 + b - z
    #   x_2 = -(w_1 * x_1 + b - z)/w_2
    #   x_2 = (z - b - w_1 * x_1)/w_2
    def calculate_x2(x1):
        return (z - b - w[0] * x1) / w[1]
    x_tmp = np.array([-10, 10])
    y_tmp = calculate_x2(x_tmp)
    plt.plot(x_tmp, y_tmp, label=f"{threshold}={w}x+b")
    plt.show()

if __name__ == "__main__":
    main()