import numpy as np
from netCDF4 import Dataset

#xL=xr.DataArray(xL,dims=['dim_0','dim_11'])
#d=xr.Dataset({"zKa_obs":zKa_obs,"zKa_true":zKa_true,"zKa_ms":zKa_ms,\
#              "pRate":pRate,"attKa":attKa,"tb35":tb35, "piaKa":piaKa,"xL":xL})
#d.to_netcdf
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib

fh=Dataset("simulatedObs_SAM2.nc")

#xL=np.array(xL)

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

plt.figure(figsize=(12, 12))

zKa=fh["zKa_obs"][:]
tbL=fh["tb35"][:]
xL=fh["xL"][:]
jacobL=fh["jacob"][:]
piaKa=fh["piaKa"][:]
n_samples = 1500
random_state = 170
zKa0=zKa
zKa0[zKa0<0]=0
nc=3
pRate=fh["pRate"][:]

from sklearn.model_selection import train_test_split

ind=range(zKa.shape[0])

ind_train, ind_test, \
    y_train, y_test \
    = train_test_split(ind, pRate[:,0], \
                       test_size=0.15, random_state=42)



kmean = KMeans(n_clusters=nc, random_state=random_state).fit(zKa0[ind_train,:])
tbL=np.array(tbL)
covL=[]
jacobL=np.array(jacobL[ind_train,:])

def pinverse(pRate):
    u,s,vt=np.linalg.svd(pRate,full_matrices=False)
    #print(u.shape)
    #print(s.shape)
    #print(vt.shape)
    nc=10
    u1=u[:,:nc]
    vt1=vt[:nc,:]
    inv1=np.diag(1/s[:nc])
    pinv=vt1.T@inv1@u1.T
    bapp=u1@np.diag(s[:nc])@vt1
    return pinv,bapp

h=125/2.+np.arange(150)*125

xL=np.array(xL)
xL_train=xL[ind_train,:]
tbL_train=tbL[ind_train]
sfcRainClass=np.zeros((nc),float)

kGainL=[]
pRateClassL=[]
a_invL=[]
piaKa_train=piaKa[ind_train]
yobs_train=[]
for i in ind_train:
    y1=list(zKa0[i,3:43])
    y1.append(tbL[i])
    yobs_train.append(y1)

yobs_train=np.array(yobs_train)
nobs=yobs_train.shape[-1]
tbClass=np.zeros((nc,nobs),float)
for i in range(nc):
    a=np.nonzero(kmean.labels_==i)
    cov1=np.cov(yobs_train[a[0],:].T,xL_train[a[0],:].T)   
    vard=np.array([v for v in np.diag(cov1[1:,1:])])
    p1,bapp=pinverse(cov1[nobs:,nobs:])
    ht=p1@cov1[nobs:,:nobs]
    #B=cov1[1:,1:]
    #plt.plot(cov1[0,1:]/(vard+1),h/1000)
    sfcRainClass[i]=y_train[a].mean()
    tbClass[i,:]=yobs_train[a].mean(axis=0)
    covyy_inv=np.linalg.pinv(4*np.eye(nobs)+cov1[0:nobs,0:nobs])
    kGainL.append(cov1[nobs:-1,:nobs]@\
                  np.linalg.pinv(4*np.eye(nobs)
                                 +cov1[0:nobs,0:nobs]))
    pRateClassL.append(pRate[ind_test,:].mean(axis=0))





from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(30, weights='distance')
knn_tb = KNeighborsRegressor(30, weights='distance')
knn.fit(zKa0[ind_train,:],y_train)
knn_tb.fit(zKa0[ind_train,:],tbL[ind_train])
yp=knn.predict(zKa0[ind_test,:])
tbp=knn_tb.predict(zKa0[ind_test,:])
labels=kmean.predict(zKa0[ind_test])

yL=[]
yobs_test=[]
for i in ind_test:
    y1=list(zKa0[i,3:43])
    y1.append(tbL[i])
    yobs_test.append(y1)

yobs_test=np.array(yobs_test)

for i,l in enumerate(labels):
    y1=sfcRainClass[l]+kGainL[l][0,:]@(yobs_test[i]-tbClass[l])
    yL.append(y1)
