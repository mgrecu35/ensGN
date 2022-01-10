import numpy as np
from netCDF4 import Dataset

#xL=xr.DataArray(xL,dims=['dim_0','dim_11'])
#d=xr.Dataset({"zKa_obs":zKa_obs,"zKa_true":zKa_true,"zKa_ms":zKa_ms,\
#              "pRate":pRate,"attKa":attKa,"tb35":tb35, "piaKa":piaKa,"xL":xL})
#d.to_netcdf
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib

fh=Dataset("simulatedObs_SAM.nc")

#xL=np.array(xL)

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

plt.figure(figsize=(12, 12))

zKa=fh["zKa_obs"][:]
zKa_true=fh["zKa_true"][:]
attKa=fh["attKa"][:]
tbL=fh["tb35"][:]
xL=fh["xL"][:]
#jacobL=fh["jacob"][:]
piaKa=fh["piaKa"][:]
n_samples = 1500
random_state = 170
zKa0=zKa
zKa0[zKa0<0]=0
nc=3
pRate=fh["pRate"][:]

a=np.nonzero(zKa_true[:,50]>0)
pZ_snow=np.polyfit(np.log(pRate[a[0],50]),zKa_true[a[0],50],1)
att_snow=np.polyfit(np.log(pRate[a[0],50]),np.log(attKa[a[0],50]),1)
a=np.nonzero(zKa_true[:,0]>0)
pZ_rain=np.polyfit(np.log(pRate[a[0],0]),zKa_true[a[0],0],1)
att_rain=np.polyfit(np.log(pRate[a[0],0]),np.log(attKa[a[0],0]),1)
h=125/2.+np.arange(150)*125
plt.plot(zKa0.mean(axis=0),h)
plt.plot(zKa0.mean(axis=0)[:40],h[:40],'*')
fint=np.interp(range(150),[0,35,40,150],[1,1,0,0])
zL=[]
dr=0.125
for i,pRate1 in enumerate(pRate[:,:]):
    snowRate=(1-fint)*pRate1
    zSnow=pZ_snow[0]*np.log(snowRate+1e-3)+pZ_snow[1]
    attSnow=np.exp(att_snow[0]*np.log(snowRate+1e-3)+att_snow[1])
    rainRate=fint*pRate1
    zRain=pZ_rain[0]*np.log(rainRate+1e-3)+pZ_rain[1]
    zTot=10*np.log10(10**(0.1*zSnow)+10**(0.1*zRain))
    attRain=np.exp(att_rain[0]*np.log(rainRate+1e-3)+att_rain[1])
    attTot=(attSnow+attRain)*2*dr
    pia=attTot[::-1].cumsum()[::-1]
    #zTot[zTot<0]=0
    zTot-=pia
    zL.append(zTot)

plt.figure()
plt.plot(zKa_true.mean(axis=0)[:120],h[:120])
plt.plot(np.array(zL).mean(axis=0)[:120],h[:120])


from sklearn.model_selection import train_test_split

ind=range(zKa.shape[0])
ind_train, ind_test, \
    y_train, y_test \
    = train_test_split(ind, pRate[:,0], \
                       test_size=0.15, random_state=42)

stop
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
    nc=20
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
    y1=list(zKa0[i,0:43])
    y1.append(tbL[i])
    yobs_train.append(y1)

yobs_train=np.array(yobs_train)
nobs=yobs_train.shape[-1]
tbClass=np.zeros((nc,nobs),float)
a1=list(range(43))
a1.append(150)
jacobr=jacobL[:,:,a1]
R_1=np.eye(nobs)*4
for i in range(nc):
    a=np.nonzero(kmean.labels_==i)
    l1=100
    cov1=np.cov(yobs_train[a[0],:].T,xL_train[a[0],:-1].T)   
    vard=np.array([v for v in np.diag(cov1[1:,1:])])
    p1,bapp=pinverse(cov1[nobs:,nobs:]+l1*np.eye(150))
    ht=np.linalg.pinv(cov1[nobs:,nobs:]+l1*np.eye(150))@cov1[nobs:,:nobs]
    kgain=np.linalg.pinv(ht@R_1@ht.T+np.linalg.pinv(cov1[nobs:,nobs:]))@ht@R_1
    bapp1=cov1[nobs:,nobs:]

    Ak=kgain@ht.T
    AkL=[]
    for i1 in a[0]:
        kgain1=bapp@jacobr[i1,:,:]@np.linalg.pinv(np.eye(nobs)*4+jacobr[i1,:,:].T@bapp@jacobr[i1,:,:])
        Ak1=kgain1@jacobr[i1,:,:].T
        AkL.append(Ak1)
    stop
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
    y1=list(zKa0[i,0:43])
    y1.append(tbL[i])
    yobs_test.append(y1)

yobs_test=np.array(yobs_test)

for i,l in enumerate(labels):
    y1=sfcRainClass[l]+kGainL[l][0,:]@(yobs_test[i]-tbClass[l])
    yL.append(y1)
