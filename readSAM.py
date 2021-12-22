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
piaKaL=[]
zsfcKaL=[]
zsfcKa_mL=[]
zBBL=[]
zKaL=[]
pRateL=[]
wL=[]
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
        
        zka_m ,attka, dzka_m,piaka,dpiaka = \
            sdsu.reflectivity(rwc1,swc1,wv1,dn1,temp,press,dr)
        for k in range(45):
            prate1=0
            if rwc[k,i1,i2]>1e-2:
                ibin=bisectm(sdsu.tablep2.rwc[:289],289,rwc[k,i1,i2])
                zKar=sdsu.tablep2.zkar[ibin]
                attKar=sdsu.tablep2.attkar[ibin]
                prate1+=(sdsu.tablep2.rainrate[ibin])
            else:
                zKar=-0
                attKar=0
            if gwc[k,i1,i2]>1e-2:
                if T[k,i1,i2]<273.15:
                    n1=np.exp(-0.122*(T[k,i1,i2]-273.15))
                else:
                    n1=1
                if n1>50:
                    n1=50
                #n1=1
                ibin=bisectm(sdsu.tablep2.gwc[:253],253,gwc[k,i1,i2]/n1)
                zKag=sdsu.tablep2.zkag[ibin]+10*np.log10(n1)
                if gwc[k,i1,i2]/n1 < sdsu.tablep2.gwc[0]:
                    zKag=np.log10(gwc[k,i1,i2]/n1)*graupCoeff[0]+\
                        graupCoeff[1]+10*np.log10(n1)
                prate1+=(sdsu.tablep2.snowrate[ibin]*n1)
                attKag=sdsu.tablep2.attkag[ibin]*n1
            else:
                zKag=-0
                attKag=0
            prate_1D.append(prate1)
            pwc_1D.append(rwc[k,i1,i2]+gwc[k,i1,i2])
            zKa=10*np.log10(10**(0.1*zKar)+10**(0.1*zKag))
            #print(zKar,zKag,k)
            zKa_1D.append(zKa)
            att_1D.append(attKar+attKag)
        zKa_int=np.interp(h,z[:45],zKa_1D)
        attKa_int=np.interp(h,z[:45],att_1D)
        pwc_int=np.interp(h,z[:45],pwc_1D)
        prate_int=np.interp(h,z[:45],prate_1D)
        wL.append(np.interp(h,z[:45],w[:45,i1,i2]))
        zKa_m=zKa_int.copy()

        for k in range(149,-1,-1):
            piaKa+=attKa_int[k]*dr
            zKa_m[k]-=piaKa
            piaKa+=attKa_int[k]*dr
            if zKa_m[k]>0 and zKa_m[k]<40 and pwc_int[k]>0.01:
                i0=int(zKa_m[k])
                cfadZ[k,i0]+=1
        
        piaKaL.append(piaKa)
        zsfcKaL.append(zKa_int[0])
        zsfcKa_mL.append(zKa_m[0])
        zBBL.append(zKa_int[38])
        zKaL.append(zka_m)
        pRateL.append(prate_int)
        
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
stop
zKaL=np.array(zKaL)
zKaL[zKaL<0]=0
pRateL=np.array(pRateL)

from minisom import MiniSom
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
