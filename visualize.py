import numpy as np
import matplotlib.pyplot as plt
from ctypes import *
lib = cdll.LoadLibrary('./svmlib.so')

# Definition in C
dim = lib.get_dim
dim.restype = c_size_t

Ridge_new = lib.RLS_new
Ridge_new.restype = c_void_p

SVM_new = lib.SVM_new
SVM_new.restype = c_void_p

Data_free = lib.Data_free
Data_free.argtypes = [c_void_p]

Model_free = lib.Model_free
Model_free.argtypes = [c_void_p]

get = lib.get_data
get.restype = c_void_p

Obj_Func = lib.obj_func
Obj_Func.argtypes = [c_void_p, c_void_p, c_double, c_double]
Obj_Func.restype = c_double

GD = lib.GD
GD.argtypes = [c_void_p, c_int, c_void_p]
GD.restype = c_void_p

SGD = lib.SGD
SGD.argtypes = [c_void_p, c_int, c_void_p]
SGD.restype = c_void_p

KGD = lib.KGD
KGD.argtypes = [c_void_p, c_int, c_void_p]
KGD.restype = c_void_p

SVRG = lib.SVRG
SVRG.argtypes = [c_void_p, c_int, c_void_p]
SVRG.restype = c_void_p

Free = lib.Free
Free.argtypes = [c_void_p]

# Only show 2D cost function
if dim() is not 2:
    print("Dimension is not 2.")
    exit(0)
ridge = Ridge_new()
data = get()

def obj_func(x_grid, y_grid):
    global ridge
    global data
    f = []
    for y in y_grid:
        f_r = []
        for x in x_grid:
            f_r.append(Obj_Func(ridge, data, c_double(x), c_double(y)))
        f.append(f_r)
    return f

# Plot SGD
step_no = 1600
# step_sgd = cast(SGD(ridge, c_int(step_no), data), POINTER(c_double))
# step_sgd[-2] = 20;
# step_sgd[-1] = -10;
# for i in range(0, step_no * 2, 2):
#     ax = plt.axes()
#     ax.annotate('', xy=(step_sgd[i], step_sgd[i+1]), xytext=(step_sgd[i-2], step_sgd[i-1]),
#             arrowprops={'arrowstyle': '->', 'color':'blue', 'lw':1})
#
# Model_free(ridge)
# ridge = Ridge_new()

# Plot SVRG
step_svrg = cast(SVRG(ridge, c_int(10), data), POINTER(c_double))
step_svrg[-2] = 20;
step_svrg[-1] = -10;
for i in range(0, 3000, 2):
    ax = plt.axes()
    ax.annotate('', xy=(step_svrg[i], step_svrg[i+1]), xytext=(step_svrg[i-2], step_svrg[i-1]),
            arrowprops={'arrowstyle': '->', 'color':'red', 'lw':1})

x_grid = np.linspace(int(step_svrg[step_no * 2 - 2] - 30), int(step_svrg[step_no * 2 - 2] + 30), 251)
y_grid = np.linspace(int(step_svrg[step_no * 2 - 1] - 30), int(step_svrg[step_no * 2 - 1] + 30), 251)
f_grid = obj_func(x_grid, y_grid)
# np.reshape(f_grid, (251, 251))
X, Y = np.meshgrid(x_grid, y_grid)
contours = plt.contour(X, Y, f_grid, 40)
plt.clabel(contours)
plt.show()

# Free space
# Model_free(ridge)
# Data_free(data)
# Free(step_sgd)
# Free(step_svrg)
