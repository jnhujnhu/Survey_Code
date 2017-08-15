import numpy as np
import matplotlib.pyplot as plt
from ctypes import *
lib = cdll.LoadLibrary('./svmlib.so')

# Definition in C
dim = lib.get_dim
dim.restype = c_size_t

SVM_new = lib.SVM_new
SVM_new.restype = c_void_p

Data_free = lib.Data_free
Data_free.argtypes = [c_void_p]

SVM_free = lib.SVM_free
SVM_free.argtypes = [c_void_p]

get = lib.get_data
get.restype = c_void_p

Obj_Func = lib.obj_func
Obj_Func.argtypes = [c_void_p, c_void_p, c_double, c_double]
Obj_Func.restype = c_double

SGD = lib.SGD
SGD.argtypes = [c_void_p, c_int, c_void_p]
SGD.restype = c_void_p

SVRG = lib.SVRG
SVRG.argtypes = [c_void_p, c_int, c_void_p]
SVRG.restype = c_void_p

Free = lib.Free
Free.argtypes = [c_void_p]

# Only show 2D cost function
if dim() is not 2:
    print("Dimension is not 2.")
    exit(0)
svm = SVM_new()
data = get()

def obj_func(x_grid, y_grid):
    global svm
    global data
    f = []
    for x in x_grid:
        f_r = []
        for y in y_grid:
            f_r.append(Obj_Func(svm, data, c_double(x), c_double(y)))
        f.append(f_r)
    return f

x_grid = np.linspace(-20, 50, 251)
y_grid = np.linspace(-47, 23, 251)
f_grid = obj_func(x_grid,y_grid)
X, Y = np.meshgrid(x_grid, y_grid)
contours = plt.contour(X, Y, f_grid, 20)
plt.clabel(contours)

# Plot SGD
step_sgd = cast(SGD(svm, c_int(1600), data), POINTER(c_double))
for i in range(2, 3000, 2):
    ax = plt.axes()
    ax.annotate('', xy=(step_sgd[i], step_sgd[i+1]), xytext=(step_sgd[i-2], step_sgd[i-1]),
            arrowprops={'arrowstyle': '->', 'color':'blue', 'lw':1})

SVM_free(svm)
svm = SVM_new()

# Plot SVRG
step_svrg = cast(SVRG(svm, c_int(2), data), POINTER(c_double))
for i in range(2, 3000, 2):
    ax = plt.axes()
    ax.annotate('', xy=(step_svrg[i], step_svrg[i+1]), xytext=(step_svrg[i-2], step_svrg[i-1]),
            arrowprops={'arrowstyle': '->', 'color':'red', 'lw':1})
plt.show()

# Free space
SVM_free(svm)
Data_free(data)
Free(step_sgd)
Free(step_svrg)
