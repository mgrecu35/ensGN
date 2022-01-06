import numpy as np
import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


def lorenz(x, y, z, s=10, r=28, b=2.667):
    '''
    Given:
       x, y, z: a point of interest in three dimensional space
       s, r, b: parameters defining the lorenz attractor
    Returns:
       x_dot, y_dot, z_dot: values of the lorenz attractor's partial
           derivatives at the point x, y, z
    '''
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot


dt = 0.01
num_steps = 10000

# Need one more for the initial values
xs = np.empty(num_steps + 1)
ys = np.empty(num_steps + 1)
zs = np.empty(num_steps + 1)

# Set initial values
xs[0], ys[0], zs[0] = (0., 1., 1.05)

# Step through "time", calculating the partial derivatives at the current point
# and using them to estimate the next point
for i in range(num_steps):
    x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
    xs[i + 1] = xs[i] + (x_dot * dt)
    ys[i + 1] = ys[i] + (y_dot * dt)
    zs[i + 1] = zs[i] + (z_dot * dt)

x_state=np.array([xs[-1],ys[-1],zs[-1]])

def run_lorenz(x0,num_steps):
    xs=x0[0]
    ys=x0[1]
    zs=x0[2]
    xL=[]
    for i in range(num_steps):
        x_dot, y_dot, z_dot = lorenz(xs, ys, zs)
        xs = xs + (x_dot * dt)
        ys = ys + (y_dot * dt)
        zs = zs + (z_dot * dt)
        xL.append([xs,ys,zs])
    return np.array(xL)
numsteps=125
xL_truth=run_lorenz(x_state,numsteps)
x_0=x_state+np.random.randn(3)*4
yobs_true=xL_truth[25::25].flatten()+np.random.randn(12)*4
xL_Ens=[]
yL_Ens=[]
for i in range(100):
    x_0_p=x_0+np.random.randn(3)*4
    xL2=run_lorenz(x_0_p,numsteps)
    xL_Ens.append(x_0_p)
    yobs=xL2[25::25].flatten()
    yL_Ens.append(yobs)

covXY=np.cov(np.array(xL_Ens).T,np.array(yL_Ens).T)
kgain=covXY[0:3,3:]@np.linalg.pinv(covXY[3:,3:]+np.eye(12)*16)
x_est=np.array(xL_Ens).mean(axis=0)+kgain@(yobs_true-np.array(yL_Ens).mean(axis=0))
