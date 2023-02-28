import random, math
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Seeding
np.random.seed(100)
CLASSA = np.concatenate((np.random.randn(10, 2) * 0.2 + [1.5, 0.5], np.random.randn(10, 2) * 0.2 + [1.5, 0.5]))
CLASSB = np.random.randn(20, 2) * 0.5 + [-1.0 , -0.5]
# CLASSA = np.concatenate((
#             np.random.randn(10, 2) * 0.45 + [2.0, 0.5],
#             np.random.randn(10, 2) * 0.45 + [-2.0, 0.5],
#             np.random.randn(0, 2) * 0.45 + [-3.0, -1.0]))

# CLASSB = np.concatenate(
#     (np.random.randn(20, 2) * 0.9 + [-0.0, -1.5],
#     np.random.randn(0, 2) * 0.3 + [4.0, 3.0]))
RBF_KERNEL_SIGMA = 2
POLY_KERNEL_P = 3

# TODO: generate test data
def generate_test_data():
    INPUTS = np.concatenate((CLASSA , CLASSB))
    TARGETS = np.concatenate((np.ones(CLASSA.shape[0]) , -np.ones(CLASSB.shape[0])))

    N = INPUTS.shape[0] # Number of rows (samples)

    permute=list(range(N))
    random.shuffle(permute)
    INPUTS = INPUTS[permute , :]
    TARGETS = TARGETS[permute]
    return (N, INPUTS, TARGETS)

# TODO: Create functions for the differnet kernels
# input: two data points output: scalar
def linear_kernel(x,y):
    return np.dot(np.transpose(x), y)

def ploynomial_kernel(x,y):
    return math.pow(np.dot(np.transpose(x), y) + 1, POLY_KERNEL_P)

def rbf_kernel(x, y):
    return math.exp(-1 * (math.pow(np.linalg.norm(np.subtract(x, y)), 2))/2*math.pow(RBF_KERNEL_SIGMA, 2))


# CONSTANTS
N, INPUTS, TARGETS = generate_test_data()
KERNEL = linear_kernel
C = 25

# TODO: implement objective function
def high_dim_repr():
    P = np.zeros((N, N))
    for i in range(N):
       for j in range(N):
            P[i][j] = TARGETS[i] * TARGETS[j] * KERNEL(INPUTS[i], INPUTS[j])
    return P

P = high_dim_repr()

def objective(alpha):
    return 0.5 * np.dot(alpha, np.dot(alpha, P)) - np.sum(alpha)

# TODO: implement the zerofun
def nonzero(alpha):
    return np.dot(alpha, TARGETS)

# TODO: call minimize
# ret = minimize (objective, start, bounds = B, constraints = XC)
# alpha = ret['x']
# vector alpha which minimizes the functino objective within the bounds B and the constraints XC
start = np.zeros(N)
bounds = [(0, C) for b in range(N)]
constraints={'type':'eq', 'fun':nonzero}
ret = minimize(objective, start, bounds=bounds, constraints=constraints)
alphas = ret['x']

# TODO: extract non-zero alpha values (with threshold 10^(-5))
# sage non-zero allpha with points x_i and values t_i in list
non_zeros = [(a, x, t) for a,x,t in zip(alphas, INPUTS, TARGETS) if a > 10e-5]

# TODO: calculate b values
# def calculate_b(non_zeros):
#     SV = non_zeros[0][1]
#     SV_target = non_zeros[0][2]
#     return np.sum([a*t*KERNEL(SV, x) for a,x,t in non_zeros]) - SV_target

# b = calculate_b(non_zeros)
# # TODO: implement the indicator function (classifier)
# def indicator(sv, non_zeros, b):
#     return np.sum([a*t*KERNEL(sv, x) for a,x,t in non_zeros]) - b

def calculate_b(non_zeros):
    sum = 0
    for val in non_zeros:
        sum += val[0] * val[2] * KERNEL(non_zeros[0][1], val[1])
    return sum - non_zeros[0][2]
b = calculate_b(non_zeros)
#7
def indicator(v, non_zeros, b):
    sum = 0
    for val in non_zeros:
        sum += val[0] * val[2] * KERNEL(v, val[1])
    return sum - b

plt.plot([p[0] for p in CLASSA], [p[1] for p in CLASSB], 'b.')
plt.plot([p[0] for p in CLASSB], [p[1] for p in CLASSB], 'r.')
plt.axis('equal')
#  Plotting the Decision Boundary
xgrid = np.linspace(-5, 5)
ygrid = np.linspace(-4, 4)
grid = np.array([[indicator([x, y], non_zeros, b) for x in xgrid] for y in ygrid])
plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(0.5, 2, 0.5))

plt.savefig('svmplot.pdf')
plt.show()