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
h1=0.+np.arange(76)*250

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
#stop1
umu=np.cos(0.0/180*np.pi)
fisot=2.7
tbL=[]
tb_L=[]
sfcRainL=[]
xL=[]
jacobL=[]
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
        temp1=np.interp(h1,z[:45],T[:45,i1,i2])
        press=np.interp(h,z[:45],prs[:45,i1,i2])
        wv1=np.zeros((150),float)
        
        zka_m ,zka_t, attka, piaka, \
            kext,salb,asym,kext_,salb_,asym_,pRate\
            =sdsu.reflectivity_2(rwc1,swc1,wv1,dn1,temp,press,dr)

        #stop
        if zka_m[40:50].min()<15:
            continue
        dr=0.125
        noms=0
        alt=400.
        freq=35.5
        nonorm=0
        theta=0.35
        zms = sdsu.multiscatterf(kext[::-1],salb[::-1],asym[::-1],\
                                 zka_t[::-1],dr,noms,alt,\
                                 theta,freq,nonorm)
        kext1=np.zeros((75),float)
        salb1=np.zeros((75),float)
        asym1=np.zeros((75),float)
        for k in range(75):
            kext1[k]=kext[2*k:2*k+2].mean()
            salb1[k]=salb[2*k:2*k+2].mean()
            asym1[k]=asym[2*k:2*k+2].mean()

        kext1_=np.zeros((75),float)
        salb1_=np.zeros((75),float)
        asym1_=np.zeros((75),float)
        for k in range(75):
            kext1_[k]=kext_[2*k:2*k+2].mean()
            salb1_[k]=salb_[2*k:2*k+2].mean()
            asym1_[k]=asym_[2*k:2*k+2].mean()
        emis=0.85+0.1*np.random.rand()
        ebar=emis
        lambert=1
        salb1[salb1>0.99]=0.99
        tb = sdsu.radtran(umu,temp1[0],temp1,h1/1000.,kext1,salb1,asym1,\
                          fisot,emis,ebar,lambert)
        #tb_ = sdsu.radtran(umu,temp1[0],temp1,h1/1000.,kext1_,salb1_,asym1_,\
        #                  fisot,emis,ebar,lambert)
        #stop
        jacob1=[]
        for k in range(150):
            if rwc1[k]>0.01 or swc1[k]>0.01:
                rwc11=rwc1.copy()
                swc11=swc1.copy()
                if rwc1[k]>0.01:
                    rwc11[k]=rwc1[k]*(1+0.1)
                if swc1[k]>0.01:
                    swc11[k]=swc1[k]*(1+0.1)
                zkag_m ,zkag_t, attkag, piakag, \
                    kextg,salbg,asymg,kext_,salb_,asym_,pRate1\
                    =sdsu.reflectivity_2(rwc11,swc11,wv1,dn1,temp,press,dr)
                kext1=np.zeros((75),float)
                salb1=np.zeros((75),float)
                asym1=np.zeros((75),float)
                for k1 in range(75):
                    kext1[k1]=kextg[2*k1:2*k1+2].mean()
                    salb1[k1]=salbg[2*k1:2*k1+2].mean()
                    asym1[k1]=asymg[2*k1:2*k1+2].mean()
                tbg_ = sdsu.radtran(umu,temp1[0],temp1,h1/1000.,\
                                   kext1,salb1,asym1,\
                                   fisot,emis,ebar,lambert)
                g=(tbg_-tb)/(pRate1[k]-pRate[k])
                if g==g:
                    jacob1.append(g)
                else:
                    jacob1.append(0)
                #stop
            else:
                jacob1.append(0.)
        if tb!=tb:
            print('FNY')
            stop
            continue
        tbL.append(tb)
        #tb_L.append(tb_)
        jacobL.append(jacob1)
        #stop
        zms=zms[::-1]
        piaKaL.append(piaka)
        zKaL.append(zka_m)
        zKa_tL.append(zka_t)
        attKaL.append(attka)
        zKa_msL.append(zms)
        pRateL.append(pRate)
        sfcRainL.append(pRate[0])
        x1=list(pRate)
        x1.append(emis)
        xL.append(x1)
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
tb35=xr.DataArray(tbL)
piaKa=xr.DataArray(piaKaL)
xL=xr.DataArray(xL,dims=['dim_0','dim_11'])
jacob=xr.DataArray(jacobL,dims=['dim_0','dim_1'])
d=xr.Dataset({"zKa_obs":zKa_obs,"zKa_true":zKa_true,"zKa_ms":zKa_ms,\
              "pRate":pRate,"attKa":attKa,"tb35":tb35, "piaKa":piaKa,"xL":xL,\
              "jacob":jacob})
d.to_netcdf("simulatedObs_SAM.nc")

xL=np.array(xL)

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

plt.figure(figsize=(12, 12))

n_samples = 1500
random_state = 170
zKa0=zKaL.copy()
zKa0[zKa0<0]=0
nc=40
kmean = KMeans(n_clusters=nc, random_state=random_state).fit(zKa0)
tbL=np.array(tbL)
covL=[]
jacobL=np.array(jacobL)
def pinverse(pRate):
    u,s,vt=np.linalg.svd(pRate,full_matrices=False)
    print(u.shape)
    print(s.shape)
    print(vt.shape)
    nc=10
    u1=u[:,:nc]
    vt1=vt[:nc,:]
    inv1=np.diag(1/s[:nc])
    pinv=vt1.T@inv1@u1.T
    bapp=u1@np.diag(s[:nc])@vt1
    return pinv,bapp

xL=np.array(xL)
for i in range(nc):
    a=np.nonzero(kmean.labels_==i)
    cov1=np.cov(tbL[a],xL[a[0],:].T)
   
    vard=np.array([v for v in np.diag(cov1[1:,1:])])
    p1,bapp=pinverse(cov1[1:,1:])
    ht=p1@cov1[0,1:]
    B=cov1[1:,1:]
    kgain=bapp@ht/(4+ht.T@bapp@ht)
    ht2=jacobL[a[0],:].mean(axis=0)
    kgain2=bapp[:-1,:-1]@ht2/(4+ht2.T@bapp[:-1,:-1]@ht2)
    #plt.plot(cov1[0,1:]/(vard+1),h/1000)
    if len(a[0])>60:
        plt.figure()
        plt.plot(cov1[0,1:-1]/(4+cov1[0,0]),h/1000)
        plt.plot(kgain[:-1],h/1000)
        plt.plot(kgain2[:],h/1000)
        print((1+ht.T@bapp@ht)/(4+cov1[0,0]))
        plt.ylim(0,10)

stop
#stop
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
