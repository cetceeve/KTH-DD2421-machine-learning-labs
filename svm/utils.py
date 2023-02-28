import numpy as np
import matplotlib.pyplot as plt
import random
from constants import *

# Dataset (given code)
np.random.seed(100) # get the same random data every time you run the program

def get_dataset():
    classA = np.concatenate((
            np.random.randn(20, 2) * 0.4 + [-2.5, 2.5],
            np.random.randn(20, 2) * 0.4 + [0.0, -2.5],
            np.random.randn(20, 2) * 0.4 + [-1.0, -0.5]))

    classB = np.concatenate(
        (np.random.randn(60, 2) * 0.4 + [-0.5, 1.5],
        np.random.randn(0, 2) * 0.3 + [4.0, 3.0]))
      #classB = np.random.randn(20, 2) * 0.4 + [-0.25, -0.5]

    inputs = np.concatenate((classA, classB))
    targets = np.concatenate((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))

    N = inputs.shape[0]

    permute = list(range(N))
    random.shuffle(permute)
    inputs = inputs[permute,:]
    targets = targets[permute]
    
    return (classA, classB, inputs, targets, N)
            
            
#1 Kernel functions
def linear_kernel(x, y):
    return np.dot(np.transpose(x), y)

def poly_kernel(x, y):
    return np.power(np.dot(np.transpose(x), y) + 1, poly_kernel_P)

def rbf_kernel(x, y):
    return np.exp(-(np.power(np.linalg.norm(np.subtract(x, y)), 2)) / (2 * np.power(rbf_kernel_sigma, 2)))

#4.2 Bounds 
def bound(N):
    return [(0, C) for b in range(N)]


# Calculations

#2.2
def pre_compute_P(N, targets, inputs, kernel):
    P = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            P[i][j] = targets[i] * targets[j] * kernel(inputs[i], inputs[j])
    return P

#5
def extract_non_zeros(a, inputs, targets, N):
    threshold = 10e-5
    return [(a[i], inputs[i], targets[i]) for i in range(N) if a[i] > threshold]

#6
def calculate_b(non_zeros, kernel):
    sum = 0
    for val in non_zeros:
        sum += val[0] * val[2] * kernel(non_zeros[0][1], val[1])
    return sum - non_zeros[0][2]

#7
def indicator(x, y, non_zeros, kernel):
    sum = 0
    for val in non_zeros:
        sum += val[0] * val[2] * kernel([x, y], val[1])
    return sum - calculate_b(non_zeros, kernel)
    
# Plotting (given code)
def plot(classA, classB, non_zeros, kernel):
    plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
    plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')
    plt.axis('equal')  # Force same scale on both axes

#  Plotting the Decision Boundary
    xgrid = np.linspace(lower_bound_x, upper_bound_x)
    ygrid = np.linspace(lower_bound_y, upper_bound_y)
    grid = np.array([[indicator(x, y, non_zeros, kernel) for x in xgrid] for y in ygrid])
    plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(0.5, 2, 0.5))
    # plt.savefig('svmplot.pdf')    # Save a copy in a file
    plt.show()  # Show the plot on the screen