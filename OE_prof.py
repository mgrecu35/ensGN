from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib
import combAlg as sdsu
from bisectm import *
#
import numpy as np
fh=Dataset("simulatedObs_SAM.nc")

zka_obs=fh["zKa_obs"][:]
zka_true=fh["zKa_true"][:]
attka=fh["attKa"][:]
pRate=fh["pRate"][:]
#d={"zKa":zKaL,"zKu":zKuL,"zKuC":zKucL,"pRate":pRateDL,"nodes":[-80,0,26]}
import pickle
d=pickle.load(open("OK_MCS_profiles.pklz","rb"))
zKa=d['zKa']
zKa=np.array(zKa)
zKa[zKa<0]=0
zka_obs_p=zka_obs.copy()[:,::-1]
zka_obs_p[zka_obs_p<0]=0
plt.plot(zka_obs_p.mean(axis=0)[37:37+106],np.arange(106)-80)
plt.plot(zKa.mean(axis=0),np.arange(106)-80)
    
plt.ylim(26,-80)
#stop
from sklearn import preprocessing
scaler_obs  = preprocessing.StandardScaler()
scaler_ture  = preprocessing.StandardScaler()
zka_obs0=zka_obs_p.copy()
#zka_obs0[zka_obs0<0]=0
zka_obs1 = scaler_obs.fit_transform(zka_obs0[:,37:37+80])
zkas=scaler_obs.transform(zKa[:,:80])
#zka_true_sc = scaler_obs.fit_transform(attka[:,0:100])

from sklearn.decomposition import pca
zpca_obs = pca.PCA()
zpca_obs.fit(zka_obs1)

#zpca_true = pca.PCA(n_components=7)
#zpca_true.fit(zka_true_sc)

pca_obs=zpca_obs.transform(zka_obs1)
pca_dpr=zpca_obs.transform(zkas[:,:])


pRateDPR=np.array(d['pRate'])[:,-1]


#pca_true=zpca_true.transform(zka_true_sc)
from sklearn.model_selection import train_test_split
X_train, X_test, \
    y_train, y_test \
    = train_test_split(pca_obs[:,0:7], pRate[:,0], \
                       test_size=0.33, random_state=42)
from sklearn import neighbors
n_neighbors=20
knn = neighbors.KNeighborsRegressor(n_neighbors, weights='distance')
y_ = knn.fit(X_train, y_train).predict(X_test)
ydpr=knn.predict(pca_dpr[:,0:7])
stop

covZZ=np.cov(zka_true.T)
invCovZZ=np.linalg.inv(covZZ[:100,:100])
z_mean=zka_true.mean(axis=0)[0:100].data
attka_mean=attka.mean(axis=0)[0:100].data
covAtt=np.cov(attka.T)
invCovAtt=np.linalg.pinv(covAtt)[0:100,0:100]
coeffsAtt=[]
for k in range(100):
    a=np.nonzero(attka[:,k]>0)
    if len(a[0]>10):
        coeffs=np.polyfit(np.log(attka[a[0],k]),zka_true[a[0],k],1)
        coeffsAtt.append(coeffs)
        #print(coeffs)

from numba import njit
coeffsAtt=np.array(coeffsAtt)

def fobj(att,z,z_obs,invCovZZ,invCovAtt,z_mean,\
         attka_mean,coeffsAtt,dr,n,\
         wobs,wz,watt,wzatt):
    f=0
    piaKa=0
    for k in range(n-1,-1,-1):
        piaKa+=np.exp(att[k])*dr
        zsim=z[k]-piaKa
        f+=wobs*(zsim-z_obs[k])**2
        piaKa+=np.exp(att[k])*dr
        
    #print(f)
    t1=np.dot(z-z_mean,invCovZZ)
    f+=wobs*np.dot(t1,z-z_mean)
    #print(f)
    datt=np.exp(att)-attka_mean
    t2=np.dot(datt,invCovAtt)
    f+=watt*np.dot(t2,datt)
    #print(f)
    for k in range(n):
        #print(coeffsAtt[k,0]*att[k]+coeffsAtt[k,,att[k],z[k])
        f+=wzatt*(coeffsAtt[k,0]*att[k]+coeffsAtt[k,1]-z[k])**2
    #print(f)
    return f

def wrapped_fobj(x):
    att=x[0:100]
    z=x[100:]
    n=100
    wobs=1.0
    wz=2.0
    watt=1.0
    wzatt=0.1
    #f=fobj(att,z,z_obs,invCovZZ,invCovAtt,z_mean,attka_mean,coeffsAtt,dr,n,\
    #       wobs,wz,watt,wzatt)

    f2 = sdsu.fobj_f90t(att,z,z_obs,invCovZZ,invCovAtt,z_mean,\
                       attka_mean,\
                       coeffsAtt,dr,wobs,wz,watt,wzatt)
    fb=1.0
    #attb1,zb1 = sdsu.fobj_f90_b(att,z,z_obs,invCovZZ,invCovAtt,z_mean,\
    #                          attka_mean,coeffsAtt,dr,\
    #                          wobs,wz,watt,wzatt,f2,fb)
    #attb2,zb2 = sdsu.fobj_f90t_b(att,z,z_obs,invCovZZ,invCovAtt,z_mean,\
    #                         attka_mean,coeffsAtt,dr,\
    #                         wobs,wz,watt,wzatt,0,fb)
    #print(attb2)
    #print(zb2)
    return f2

def wrapped_jac_fobj(x):
    att=x[0:100]
    z=x[100:]
    n=100
    wobs=1.0
    wz=2.0
    watt=1.0
    wzatt=0.1

    fb=1.0
    
    attb2,zb2 = sdsu.fobj_f90t_b(att,z,z_obs,invCovZZ,invCovAtt,z_mean,\
                              attka_mean,coeffsAtt,dr,\
                              wobs,wz,watt,wzatt,0,fb)
    jacb=[]
    jacb.extend(attb2)
    jacb.extend(zb2)
    return np.array(jacb)


dr=0.125
z_obs=zka_obs[17,:100]
x=[]
x.extend(np.log(attka[17,:100]+1e-5))
x.extend(zka_true[17,:100])
x1=[]
x1.extend(np.log(attka_mean+1e-5))
x1.extend(z_mean)

x=np.array(x)
x1=np.array(x1)

f1=wrapped_fobj(x)
f2=wrapped_fobj(x1)

import dfols
import dfbgn
import pybobyqa
#soln = dfbgn.solve(wrapped_fobj, x1)
#stop
import scipy

bnds=[]
for i in range(100):
    bnds.append((-3,2))
for i in range(100):
    bnds.append((0,50))

xL=[]
for i in range(17,18):
    x1=[]
    x1.extend(np.log(attka_mean+1e-5))
    x1.extend(z_mean)
    z_obs=zka_obs[i,:100]
    soln=scipy.optimize.minimize(wrapped_fobj, x1, \
                                 jac=wrapped_jac_fobj,\
                                 method='SLSQP',bounds=bnds,\
                                 options={'maxiter':300})
    print(i,zka_true[i,0],soln['x'][100])
    xL.append(soln['x'])
