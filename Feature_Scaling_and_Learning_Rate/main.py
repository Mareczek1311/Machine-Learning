
import numpy as np
import copy, math

def compute_cost(x, y, w, b):
    sum = 0.0

    m = x.shape[0]

    for i in range(m):
        prediction = np.dot(x[i], w) + b

        sum = sum + (prediction - y[i]) ** 2

    sum = sum /(2*m)

    return sum   
    

def compute_gradient(x, y, w, b):
    m, n = x.shape #for example we have 3 and n elements from file

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


def gradient_descent(X, y, alpha, num_iters): 
    J_history = []

    m, n = X.shape

    w = np.zeros(n)
    b = 0

    for i in range(num_iters):
        dt_w, dt_b = compute_gradient(X, y, w, b)

        w = w - alpha*dt_w
        b = b - alpha*dt_b
        
        # Save cost J at each iteration
        J_history.append( compute_cost(X, y, w, b))
        print(J_history[-1])

    return w, b

def zscore_normalize_feature(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X-mu) / sigma

    return (X_norm, mu, sigma)



def load_house_data():
    data = np.loadtxt("./data/houses.txt", delimiter=',', skiprows=1)
    X = data[:,:4]
    y = data[:,4]
    return X, y

X_train, y_train = load_house_data()
X_features = ['size(sqft)','bedrooms','floors','age']

#with aplha 9.9e-7 gradient descent does not converge
#_, _ = gradient_descent(X_train, y_train, 9.9e-7, 10)

#lets try with alpha = 9e-7
#_,_ = gradient_descent(X_train, y_train, 9e-7, 10)

#bit smaller value for alpha
#_,_ = gradient_descent(X_train, y_train, 1e-7, 10)

X_norm, X_mu, X_sigma = zscore_normalize_feature(X_train)

w_norm, b_norm = gradient_descent(X_norm, y_train, 1.0e-1, 1000 )

# First, normalize out example.
x_house = np.array([1200, 3, 1, 40])
x_house_norm = (x_house - X_mu) / X_sigma
print(x_house_norm)
x_house_predict = np.dot(x_house_norm, w_norm) + b_norm
print(f" predicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = ${x_house_predict*1000:0.0f}")

