n1=4900
n2=5100
#n1=1700
#n2=1900
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
x2L=[]
for ip in range(100):
    xL=[]
    x=0
    dx=0
    for i in range(30):
        if np.random.randn()>0:
            dx=dx+0.05
        else:
            dx=dx-0.05
        x=0.9*x+0.1*np.random.randn()
        xL.append(x+dx)
    plt.plot(xL)
    x2L.append(xL)

x2L=np.array(x2L)
#plt.plot(x2L.mean(axis=0))
#stop
fname='2A.GPM.DPR.V8-20180723.20180625-S180356-E193630.024566.V06A.HDF5'
fname='/home/grecu/XPlainable/data/2A.GPM.DPR.V8-20180723.20180625-S041042-E054316.024557.V06A.HDF5'
fhC=Dataset('/home/grecu/XPlainable/data/2B.GPM.DPRGMI.CORRA2018.20180625-S041042-E054316.024557.V06A.HDF5')
fh=Dataset(fname,'r')
zKu=fh['NS/PRE/zFactorMeasured'][n1:n2,:,]
zKuC=fh['NS/SLV/zFactorCorrected'][n1:n2,:,:]
pRateCMB=fhC['MS/precipTotRate'][n1:n2,:,:]
zKa=fh['MS/PRE/zFactorMeasured'][n1:n2,:,]
bzd=fh['NS/VER/binZeroDeg'][n1:n2,:,]
pType=(fh['NS/CSF/typePrecip'][n1:n2,:,]/1e7).astype(int)
stormTop=(fh['NS/PRE/binStormTop'][n1:n2,:,]/1e7).astype(int)
bcf=fh['NS/PRE/binClutterFreeBottom'][n1:n2,:,]
h0=fh['NS/VER/heightZeroDeg'][n1:n2,:]
s0=fh['NS/PRE/sigmaZeroMeasured'][n1:n2,:]
s0Ka=fh['MS/PRE/sigmaZeroMeasured'][n1:n2,:]
pia=fh['NS/SRT/pathAtten'][n1:n2,:]
fhKu=Dataset('/home/grecu/XPlainable/data/2A.GPM.Ku.V8-20180723.20180625-S041042-E054316.024557.V06A.HDF5')
srtPIA=fhKu['NS/SRT/pathAtten'][n1:n2,:]
relPIA=fhKu['NS/SRT/reliabFlag'][n1:n2,:]

relFlag=fh['NS/SRT/reliabFlag'][n1:n2,:]
sfcRain=fh['NS/SLV/precipRateNearSurface'][n1:n2,:]
pRate=fh['NS/SLV/precipRate'][n1:n2,:]
zKum=np.ma.array(zKu,mask=zKu<0)
zKuC=np.ma.array(zKuC,mask=zKuC<0)
zKam=np.ma.array(zKa,mask=zKa<0)
zKuC.data[zKuC.data<0]=0


plt.subplot(211)
plt.title('Ku-band')
plt.pcolormesh(range(100),175*0.125-np.arange(176)*0.125, zKum[75:175,34,:].T,cmap='jet',vmax=50)
plt.ylabel('Height [km]')
plt.ylim(0,15)
frame1 = plt.gca()
frame1.axes.xaxis.set_ticklabels([])
plt.colorbar()
plt.subplot(212)
plt.title('Ka-band')
plt.pcolormesh(range(100),175*0.125-np.arange(176)*0.125, zKam[75:175,22,:].T,cmap='jet',vmax=40)
plt.ylabel('Height [km]')
plt.xlabel('Scan #')
plt.ylim(0,15)
plt.colorbar()
plt.savefig('OK_MCS.png')

hCoeff=np.array([ 0.06605835, -2.08407732])
hCoeff=np.array([ 0.06737808, -3.34419843])
hCoeff=np.array([ 0.07946967, -2.32328256])

import matplotlib

nx,ny,nz=zKum.shape
hailD=np.zeros((nx,ny),int)
hailD2=np.zeros((nx,ny),int)
zKuL=[]
zKucL=[]
indL=[]
zKaL=[]
bzdL=[]
ihaiL=[]
pRateL=[]
pRateDL=[]
srtPIAL=[]
pRateCFL=[]
h0L=[]
bcfL=[]
for i in range(nx):
    for j in range(12,37):
        n1=min(bzd[i,j],bcf[i,j])
        ind=np.nonzero(zKum[i,j,:n1]>40)
        if bcf[i,j]-bzd[i,j]<25:
            continue
        if pType[i,j]==2:
            nzb1=bzd[i,j]
            zKuL.append(zKum[i,j,nzb1-80:nzb1+26])
            zKucL.append(zKuC[i,j,nzb1-80:nzb1+26])
            zKaL.append(zKam[i,j-12,nzb1-80:nzb1+26])
            nzb=int(bzd[i,j]/2)
            #pRateL.append(pRateCMB[i,j-12,nzb-30:nzb+10])
            pRateDL.append(pRate[i,j,nzb1-80:nzb1+26])
            #pRateCFL.append(pRate[i,j,nzb1+25])
            bzdL.append(bzd[i,j])
            srtPIAL.append(srtPIA[i,j])
            indL.append([i,j])
            h0L.append(h0[i,j])
            bcfL.append(bcf[i,j])
            if len(ind[0])>16:
                #print(ind)
                #stop
                ihaiL.append(1)
                if n1-ind[0][0]>8:
                    hailD[i,j]=1
            else:
                ihaiL.append(0)
                if len(ind[0])<4:
                    hailD2[i,j]=1
                    #else:
            #    if len(ind[0])>4:
            #        hailD2[i,j]=1
        
d={"zKa":zKaL,"zKu":zKuL,"zKuC":zKucL,"pRate":pRateDL,"nodes":[-80,0,26]}
import pickle
pickle.dump(d,open("OK_MCS_profiles.pklz","wb"))
stop
#import combAlg as cmb
#cmb.mainfortpy()
#cmb.initp2()

#plt.colorbar()
#plt.subplot(212)
#plt.pcolormesh(zKam[:,23,::-1].T,cmap='jet',vmax=35)
#plt.xlim(125,175)
#plt.ylim(10,100)
#plt.xlim(145,160)
#plt.colorbar()
#plt.figure()
pRateL=np.array(pRateL)
a=np.nonzero(hailD2>0)
if 1==2:
    for i1,j1 in zip(a[0][:],a[1][:]):
        plt.subplot(121)
        plt.plot(zKum[i1,j1,:],range(176))
        plt.ylim(160,60)
        plt.xlim(0,55)
        plt.subplot(122)
        if abs(j1-24)<12:
            plt.plot(zKam[i1,j1-12,:],range(176))
            plt.ylim(160,60)
    #plt.show()
from minisom import MiniSom
zKuL=np.array(zKuL)
zKaL=np.array(zKaL)
n1=10
n2=1
nz=86
zKuL[zKuL<0]=0
zKaL[zKaL<0]=0
som = MiniSom(n1,n2,nz,sigma=2.5,learning_rate=0.5, random_seed=0)
#np.random.seed(seed=10)
som.random_weights_init(zKuL)
som.train_random(zKuL,500) # training with 100 iterations
nt=zKuL.shape[0]
winL=np.zeros((nt),int)
it=0
for z1 in zKuL:
    win=som.winner(z1)
    winL[it]=win[0]
    it+=1

import pytmatrix.refractive
import pytmatrix.refractive as refr

wl=[pytmatrix.refractive.wl_Ku,pytmatrix.refractive.wl_Ka,\
    pytmatrix.refractive.wl_W]

ifreq=0
refr_ind_w=pytmatrix.refractive.m_w_0C[wl[ifreq]]
rhow=1000.0
rhos=500
refr_ind_s=refr.mi(wl[0],rhos/rhow)

f=open('zKuProfiles.txt','w')
fKa=open('zKaProfiles.txt','w')
from dmCoeff import *
from hb import *
dr=0.125
plt.figure()
pia12=[]
zcL=[]
zku2=[]
dmL=[]
ifreq=0
import bhmief as bh
nw_g=0.04
nw_r=0.02
pRateL=[]
zc2L=[]
nfz=51
nEns=30
for i in range(0,1):
    aw=np.nonzero(winL==i)
    for iw in [229]:#aw[0][:]:
        alpha=10**attMCoeffs[1]
        beta=attMCoeffs[0]
        srt_piaKu=srtPIAL[iw]
        
        zc,eps,pia=hb(zKuL[iw,:],alpha,beta,dr,srt_piaKu)
    
        #zG,zKuREns,pRateEns=profiling(zKuL[iw,:],srtPIAL[iw],dmgCoeff,dmrCoeff,attGCoeffs,attRCoeffs,\
        #                              hb,nEns,refr_ind_s,refr_ind_w,rhos,rhow,\
        #                              wl,ifreq,nz,dr,nfz,bh,nw_g,nw_r,f_mu,mu)
        
        #zc=zKuL[iw,:]
        #dmg=10**(rrateCoeff[0]*zc[:]/10+rrateCoeff[1])
        #dmr=10**(grateCoeff[0]*zc[:]/10+grateCoeff[1])
        #plt.plot(zG,range(nfz))
        #zKuREns=np.array(zKuREns)
        #plt.plot(zKuREns.mean(axis=0),range(nfz,nz))
        #plt.plot(dmg,range(nz))

        if zc[-1]>70:
            plt.figure()
            plt.plot(zc,range(nz))
            plt.ylim(nz-1,0)
            plt.plot(zKuL[iw,:],range(nz))
            plt.plot(zKucL[iw],range(nz))
            plt.show()
            stop
        #stop

stop
plt.figure()

zcL=np.array(zcL)
zc2L=np.array(zc2L)
zku2=np.array(zku2)
plt.plot(zcL.mean(axis=0),range(nz))
plt.plot(zc2L.mean(axis=0),range(nz))
plt.plot(zku2.mean(axis=0),range(nz))
#plt.plot(zKuL[iw,:],range(nz))
plt.ylim(nz-1,0)
dmL=np.array(pRateL)
plt.figure()
plt.plot(dmL.mean(axis=0),range(nz))
plt.ylim(nz-1,0)
stop
for i in range(1,2):
    aw=np.nonzero(winL==i)
    plt.figure()
    plt.subplot(131)
    #plt.plot(zKuL[aw[0],:101].mean(axis=0),range(nz))
    SL=[]
    piaL=[]
    print(aw[0])
    for iw in aw[0]:
        zString=''
        for k in range(86):
            zString+='%6.2f '%zKuL[iw,k]
        f.write(zString+'\n')
        zString=''
        for k in range(86):
            zString+='%6.2f '%zKaL[iw,k]
        fKa.write(zString+'\n')
            
        #print(zKuL[iw,57:59],zKuL[iw,64:67])
        dfr1=zKuL[iw,59]-zKaL[iw,59]
        #S=(10**(0.1*zKuL[iw,:])/a)**(1/b)
    dmg=10**(dmgCoeff[0]*zKuL[aw[0],:101]+dmgCoeff[1])
    plt.plot(dmg[:,:101].mean(axis=0),range(nz))
    #plt.plot(zKuL[aw[0],60:61].mean(axis=0),range(60,61),'*')
    plt.ylim(nz-1,0)
    #plt.title(ihaiL[aw[0]].sum()/len(aw[0]))
    plt.subplot(132)
    plt.title('class# %i'%(i+1))
    plt.plot(zKuL[aw[0],:90].mean(axis=0)-zKaL[aw[0],:90].mean(axis=0),range(nz))
    plt.ylim(nz-1,0)
    plt.subplot(133)
    plt.title('# profiles %i'%(len(aw[0])))
   # plt.plot(pRateDL[aw[0],:].mean(axis=0),np.arange(nz))
    
        
    plt.ylim(nz,0)
f.close()
fKa.close()
stop
plt.figure()
plt.plot(range(125,175),10-s0[125:175,23])
plt.plot(range(125,175),0-s0Ka[125:175,11])
plt.plot(range(125,175),10-s0[125:175,23])
plt.plot(range(125,175),(s0[125:175,22]-s0Ka[125:175,11]-2)/6.)
plt.xlim(145,160)


plt.figure()
plt.subplot(211)
plt.pcolormesh(zKum[155,:,::-1].T,cmap='jet',vmax=50)
plt.colorbar()
plt.subplot(212)
hCoeff=np.array([ 0.06605835, -2.38407732])
plt.pcolormesh((10**(hCoeff[0]*zKum[160,:,::-1]+hCoeff[1])).T,\
               cmap='jet',vmax=10)
plt.colorbar()
#stop
#stop
pRate=fh['NS/SLV/precipRate'][n1:n2,:]
zKu[zKu<0]=0
zmL2=[]
pRateL=[]
piaL=[]
relFlagL=[]
sfcRainL=[]

hCoeff=np.array([ 0.06605835, -2.38407732])

stop
for i1 in range(zKu.shape[0]):
    for j1 in range(20,28):
        if bzd[i1,j1]>stormTop[i1,j1]+4 and bcf[i1,j1]-bzd[i1,j1]>20 and\
           pType[i1,j1]==2:
            if bzd[i1,j1]-60>0 and bzd[i1,j1]+20<176:
                zmL2.append(zKu[i1,j1,bzd[i1,j1]-60:bzd[i1,j1]+20])
                pRateL.append(pRate[i1,j1,bzd[i1,j1]-60:bzd[i1,j1]+30])
                piaL.append(pia[i1,j1])
                relFlagL.append(relFlag[i1,j1])
                sfcRainL.append(sfcRain[i1,j1])

import pickle
som=pickle.load(open("miniSOM_Land.pklz","rb"))

nx=len(zmL2)
iclassL=[]
for it in range(nx):
    win=som.winner(zmL2[it])
    iclass=win[0]*3+win[1]+1
    iclassL.append(iclass)

plt.figure()
plt.plot(np.array(pRateL).mean(axis=0),-60+np.arange(90))
plt.ylim(30,-60)
plt.xlim(0,40)

plt.figure()

for i in range(nx):
    if iclassL[i]==9:
        plt.plot(zmL2[i],-60+np.arange(80))
plt.ylim(20,-60)

from sklearn.cluster import KMeans

plt.figure(figsize=(12, 12))

iclassL=np.array(iclassL)
a=np.nonzero(iclassL==9)
# Incorrect number of clusters
zmL2=np.array(zmL2)

kmeans = KMeans(n_clusters=16, random_state=10).fit(zmL2[a[0],:])
plt.figure()
zmAvg=[]
piaL=np.array(piaL)
sfcRainL=np.array(sfcRainL)
for i in range(16):
    a1=np.nonzero(kmeans.labels_==i)
    #plt.figure()
    zm1=zmL2[a[0][a1],:].mean(axis=0)
    if zm1.max()>47:
        plt.plot(zm1,-60+np.arange(80))
    zmAvg.append(zmL2[a[0][a1],:].mean(axis=0))
    plt.ylim(20,-60)
    #plt.title("PIA=%6.2f %6.2f"%(piaL[a[0][a1]].mean(),sfcRainL[a[0][a1]].mean()))
plt.xlabel('dBZ')
plt.ylabel('Relative range')
plt.savefig('deepConvProfs.png')
pickle.dump({"zmAvg":zmAvg},open("zmAvg.pklz","wb"))
