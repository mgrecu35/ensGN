from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib
import combAlg as sdsu
from bisectm import *
#import dfbgn
#import dfols
sdsu.mainfortpy()
sdsu.initp2()

import glob
import numpy as np
fs=sorted(glob.glob("../../../NAO/SAM*nc"))
nt=0
R=287
zKuL=[]
h=125/2.+np.arange(150)*125

graupCoeff=np.polyfit(np.log10(sdsu.tablep2.gwc[:272]),\
                      sdsu.tablep2.zkag[:272],1)


cfadZ=np.zeros((150,40),float)
cfadZms=np.zeros((150,40),float)
piaKaL=[]
zsfcKaL=[]
zsfcKa_mL=[]
zBBL=[]
zKaL=[]
pRateL=[]
wL=[]
zKa_tL=[]
attKaL=[]
zKa_msL=[]
for f in fs[:]:
    fh=Dataset(f)
    z=fh['z'][:]
    qr=fh["plw"][:]
    qg=fh["pli"][:]
    a=np.nonzero(qr[0,:,:]>0.0005)
    print(a[0].shape[0],qr[0,:,:].max())
    T=fh["ta"][0,:,:,:]
    prs=fh["pa"][0,:,:,:]
    w=fh['wa'][0,:,:,:]
    rho=prs/(R*T)
    rwc=qr*rho*1e3
    gwc=qg*rho*1e3
    qv=fh['QV'][0,:,:,:]
    wv=qv*rho
    #plt.figure(figsize=(8,11))
    
    for i1,i2 in zip(a[0],a[1]):
        zKa_1D=[]
        att_1D=[]
        pwc_1D=[]
        prate_1D=[]
       
        piaKa=0
        dr=0.125
        dn1=np.zeros((150),float)
        rwc1=np.interp(h,z[:45],rwc[:45,i1,i2])
        swc1=np.interp(h,z[:45],gwc[:45,i1,i2])
        temp=np.interp(h,z[:45],T[:45,i1,i2])
        press=np.interp(h,z[:45],prs[:45,i1,i2])
        wv1=np.zeros((150),float)
        
        zka_m ,zka_t, attka, dzka_m,piaka,dpiaka, \
            kext,salb,asym,pRate\
            =sdsu.reflectivity(rwc1,swc1,wv1,dn1,temp,press,dr)
        #stop
        dr=0.125
        noms=0
        alt=400.
        freq=35.5
        nonorm=0
        theta=0.35
        zms = sdsu.multiscatterf(kext[::-1],salb[::-1],asym[::-1],\
                                 zka_t[::-1],dr,noms,alt,\
                                 theta,freq,nonorm)

        zms=zms[::-1]
        piaKaL.append(piaka)
        zKaL.append(zka_m)
        zKa_tL.append(zka_t)
        attKaL.append(attka)
        zKa_msL.append(zms)
        pRateL.append(pRate)
        for k in range(149,-1,-1):
            if zka_m[k]>0 and zka_m[k]<40 and swc1[k]+rwc1[k]>0.01:
                i0=int(zka_m[k])
                cfadZ[k,i0]+=1
            if zms[k]>0 and zms[k]<40 and swc1[k]+rwc1[k]>0.01:
                i0=int(zms[k])
                cfadZms[k,i0]+=1
        
        #plt.scatter(zKa_m,h,s=1)

        #zKuL.append(zKu)
                
    #stop
    #plt.figure()
    #qr=np.ma.array(qr,mask=qr<1e-4)
    #plt.pcolormesh(qr[0,:,100:900],norm=matplotlib.colors.LogNorm(),cmap='jet')
    #plt.colorbar()
    #nt+=a[0].shape[0]
    #stop
plt.pcolormesh(cfadZ,cmap='jet',norm=matplotlib.colors.LogNorm())
plt.figure()
plt.pcolormesh(cfadZms,cmap='jet',norm=matplotlib.colors.LogNorm())
attKaL=np.array(attKaL)
zKa_tL=np.array(zKa_tL)
zKaL=np.array(zKaL)
#zKaL[zKaL<0]=0
pRateL=np.array(pRateL)

import xarray as xr

zKa_obs=xr.DataArray(zKaL)
zKa_true=xr.DataArray(zKa_tL)
zKa_ms=xr.DataArray(zKa_msL)
pRate=xr.DataArray(pRateL)
attKa=xr.DataArray(attKaL)

d=xr.Dataset({"zKa_obs":zKa_obs,"zKa_true":zKa_true,"zKa_ms":zKa_ms,\
              "pRate":pRate,"attKa":attKa})
d.to_netcdf("simulatedObs_SAM.nc")

stop
#zka_obs=fh["zka_m"][:]
#zka_true=fh["zka_t"][:]
#attka=fh["attka"][:]

#from minisom import MiniSom

n1=40
n2=1
nz=150
som = MiniSom(n1,n2,nz,sigma=2.5,learning_rate=0.5, random_seed=0)
som.random_weights_init(zKaL)
som.train_random(zKaL,500) # training with 100 iterations
nt=zKaL.shape[0]
winL=np.zeros((nt),int)
it=0
zKaClass=np.zeros((n1,nz),float)
pRateClass=np.zeros((n1),float)

for it,z1 in enumerate(zKaL):
    win=som.winner(z1)
    winL[it]=win[0]

for i in range(n1):
    a=np.nonzero(winL==i)
    zKaClass[i]=zKaL[a[0],:].mean(axis=0)
    pRateClass[i]=pRateL[a[0],0].mean()


                  
from sklearn.cluster import KMeans

wL=np.array(wL)

from sklearn.model_selection import train_test_split

X_train, X_test, \
    y_train, y_test = train_test_split(zKaL, pRateL, \
                                       test_size=0.33, random_state=42)

from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(30, weights='distance')
knn.fit(X_train[:,25:100],y_train[:,0])
yp_=knn.predict(X_test[:,25:100])
nc=40
kmeans = KMeans(n_clusters=nc, random_state=10).fit(X_train)


for i in range(nc):
    a=np.nonzero(kmeans.labels_==i)
    print(len(a[0]))
                  
pLabels=kmeans.predict(X_test)
ypL=[]
for i,zka in enumerate(X_test):
    a=np.nonzero(kmeans.labels_==pLabels[i])
    covYY=np.cov(X_train[a[0],25:125].T)
    covXY=np.cov(y_train[a[0],0],X_train[a[0],25:125].T)
    y_mean=X_train[a[0],25:125].mean(axis=0)
    x_mean=y_train[a[0],0].mean(axis=0)
    kgain=np.dot(covXY[0,1:],np.linalg.pinv(covYY+np.eye(100)*9))
    y_p=x_mean-np.dot(kgain,(y_mean-X_test[i,25:125]))
    ypL.append([y_p,y_test[i,0],x_mean])
#    stop
