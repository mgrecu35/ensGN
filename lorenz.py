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
numsteps=65

xL_truth=run_lorenz(x_state,numsteps)
x_0=x_state+np.random.randn(3)*2
yobs_true=xL_truth[25::2].flatten()+np.random.randn(60)*2

xL_Ens=[]

R_1=np.eye(39)/9
for i in range(200):
    x_0_p=x_0+np.random.randn(3)*3
    xL_Ens.append(x_0_p)

x_sol=x_0

xEns_L=[]
for i in range(200):
    xEns_L.append(x_sol.copy()+np.random.randn(3)*1)
    
for it in range(4):
    yL_Ens=[]
    x_0_p=x_sol.copy()
    xL2=run_lorenz(x_0_p,numsteps)
    yobs=xL2[25::2].flatten()
    fobs=np.sum((yobs_true-yobs)**2)/4+np.sum((x_sol-x_0)**2)
    print(fobs,x_sol,x_state,np.mean((x_sol-x_state)**2)**0.5)
    yobsL=[]
    for i in range(200):
        x_0_p=xEns_L[i]
        xL2=run_lorenz(x_0_p,numsteps)
        yobs1=xL2[25::2].flatten()
        yobsL.append(yobs1)

    covXY=np.cov(np.array(xEns_L).T,np.array(yobsL).T)
    kgain=covXY[0:3,3:]@np.linalg.inv(covXY[3:,3:]+np.eye(60)*4*4.)

    for i in range(100):
        xEns_L[i]=xEns_L[i]+kgain@(yobs_true-yobsL[i])#-A_1_B_1@(x_sol-x_0)

    x_sol=np.array(xEns_L).mean(axis=0)
    print(np.array(xEns_L).std(axis=0))
    #
    #print(np.mean((xL_m-x_state)**2),it,np.array(xL_Ens).std(axis=0))
    
    
    #for i in range(200):
    #    xL_Ens[i]=xL_Ens[i]+0.5*(kgain@(yobs_true-yL_Ens[i])-A_1_B_1@(xL_Ens[i]-x_0))

    #xL_m=np.array(xL_Ens).mean(axis=0)
    #print(np.mean((xL_m-x_state)**2),it,np.array(xL_Ens).std(axis=0))


stop
for it in range(3):
    yL_Ens=[]
    x_0_p=x_sol.copy()
    xL2=run_lorenz(x_0_p,numsteps)
    yobs=xL2[25::5].flatten()
    jacbL=[]
    fobs=np.sum((yobs_true-yobs)**2)/9+np.sum((x_sol-x_0)**2)/9
    print(fobs,x_sol,x_state)
    jacb2L=[]
    for i in range(100):
        x_0_p=x_sol.copy()
        x_0_p[i]+=0.1
        xL2=run_lorenz(x_0_p,numsteps)
        yobs1=xL2[25::5].flatten()
        fobs1=np.sum((yobs_true-yobs1)**2)/9+np.sum((x_sol-x_0)**2)/9

        jacbL.append((fobs1-fobs)/0.1)
        jacb2L.append((yobs1-yobs)/0.1)

    for i in range(-3):
        x_sol[i]-=0.01*jacbL[i]
    #stop
    #covXY=np.cov(np.array(xL_Ens).T,np.array(yL_Ens).T)
    #kgain=covXY[0:3,3:]@np.linalg.pinv(covXY[3:,3:]+np.eye(60)*4)
    #ht=np.linalg.pinv(covXY[:3,:3])@covXY[0:3,3:]
    #kgain=(np.eye(3)*4)@ht@np.linalg.pinv(np.eye(60)*4+4*ht.T@np.eye(3)@ht)
    ht=np.array(jacb2L)
    lam=0.0
    A=3*ht@R_1@ht.T+np.eye(3)*1/4.+lam*np.eye(3)
    A_1=np.linalg.pinv(A)
    A_1_B_1=np.linalg.inv(A)@np.eye(3)/4
    kgain=A_1@ht@R_1*3
    #print(np.mean((x_sol-x_state)**2))
    x_sol=x_sol+kgain@(yobs_true-yobs)#-A_1_B_1@(x_sol-x_0)
    #xL_m=np.array(xL_Ens).mean(axis=0)0=
    #
    #print(np.mean((xL_m-x_state)**2),it,np.array(xL_Ens).std(axis=0))
    
    
    #for i in range(200):
    #    xL_Ens[i]=xL_Ens[i]+0.5*(kgain@(yobs_true-yL_Ens[i])-A_1_B_1@(xL_Ens[i]-x_0))

    #xL_m=np.array(xL_Ens).mean(axis=0)
    #print(np.mean((xL_m-x_state)**2),it,np.array(xL_Ens).std(axis=0))
