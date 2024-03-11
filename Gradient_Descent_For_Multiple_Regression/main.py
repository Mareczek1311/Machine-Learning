import copy, math
import numpy as np

x_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

print(f"X Shape: {x_train.shape}, X type: {type(x_train)}")
print(x_train)

print("")

print(f"Y Shape: {y_train.shape}, Y type: {type(y_train)}")
print(y_train)

print("")

w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

def compute_cost(x, y, w, b):
    sum = 0.0

    m = x.shape[0]

    for i in range(m):
        prediction = np.dot(x[i], w) + b

        sum = sum + (prediction - y[i]) ** 2

    sum = sum /(2*m)

    return sum   
    

def compute_gradient(x, y, w, b):
    m, n = x.shape #for example we have 3 and 4

    dt_w = np.zeros((n,))
    dt_b = 0.

    for i in range(m):
        error = (np.dot(x[i], w) + b) - y[i]

        for j in range(n):
            dt_w[j] = dt_w[j] + error*x[i,j]
        
        dt_b = dt_b + error

    dt_w = dt_w / m
    dt_b = dt_b / m

    return dt_w, dt_b


def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in

    for i in range(num_iters):
        dt_w, dt_b = compute_gradient(X, y, w, b)

        w = w - alpha*dt_w
        b = b - alpha*dt_b
        
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(X, y, w, b))

    return w, b


initial_w = np.zeros_like(w_init)
initial_b = 0.

iterations = 1000
alpha = 5.0e-7

w_final, b_final = gradient_descent(x_train, y_train, initial_w, initial_b,
                                                    compute_cost, compute_gradient, 
                                                    alpha, iterations)

print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")

m,_ = x_train.shape
for i in range(m):
    print(f"prediction: {np.dot(x_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")