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

def computeZ(pRate,fint,pZ_snow,pZ_rain,att_snow,att_rain):
    zL=[]
    for i, pRate1 in enumerate(pRate[:,:]):
        snowRate=(1-fint)*pRate1
        zSnow=pZ_snow[0]*np.log(snowRate+1e-3)+pZ_snow[1]
        attSnow=np.exp(att_snow[0]*np.log(snowRate+1e-3)+att_snow[1])
        rainRate=fint*pRate1
        zRain=pZ_rain[0]*np.log(rainRate+1e-3)+pZ_rain[1]
        zTot=10*np.log10(10**(0.1*zSnow)+10**(0.1*zRain))
        attRain=np.exp(att_rain[0]*np.log(rainRate+1e-3)+att_rain[1])
        attTot=(attSnow+attRain)*2*dr
        pia=attTot[::-1].cumsum()[::-1]
        zTot-=pia
        zL.append(zTot)
    return np.array(zL)

plt.figure()
plt.plot(zKa_true.mean(axis=0)[:120],h[:120])
plt.plot(np.array(zL).mean(axis=0)[:120],h[:120])

zL=np.array(zL)

from sklearn.model_selection import train_test_split

ind=range(zKa.shape[0])
ind_train, ind_test, \
    y_train, y_test \
    = train_test_split(ind, pRate[:,0], \
                       test_size=0.15, random_state=42)

covT=np.cov(zL[ind_train,:120].T,pRate[ind_train,:120].T)
covYY_inv=np.linalg.pinv(covT[:120,:120]+np.eye(120)*9)
kGain=covT[120:,:120]@covYY_inv
pRatem=pRate[ind_train,:120].mean(axis=0)
zm=zL[ind_train,:120].mean(axis=0)
x1,x2=[],[]
from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=100)
zthr=10
zL0=zL.copy()
zL0[zL0<zthr]=0
neigh.fit(zL0[ind_train,:120])
x1,x2,x3=[],[],[]
ind_train=np.array(ind_train)
rng = np.random.default_rng()
import combAlg as calg
import scipy
import scipy.optimize
from scipy.optimize import minimize as minimize

B_inv=np.linalg.pinv(covT[120:,120:])
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

B_inv=pinverse(covT[120:,120:])[0]
def f_obj(x):
    pRateEns1=x
    z_att_f = calg.computez(pRateEns1,fint[:120],\
                            pZ_snow,pZ_rain,att_snow,att_rain,dr)
    f_val=1/2*(z_att_f[a0]-zobs[a0]).T@(z_att_f[a0]-zobs[a0])/4+\
        1/2*(x-pRatem).T@B_inv@(x-pRatem)
    #print(float(f_val))
    return float(f_val)

def g_fobj(x):
    pRateEns1=x
    pRateD=pRateEns1*0+1.0
    z_att_f, z_attd = calg.computez_d(pRateEns1,pRateD,fint[:120],\
                            pZ_snow,pZ_rain,att_snow,att_rain,dr)
    
    

    return z_attd,z_att_f
bnds=[(0, 100) for i in range(120)]

for i in ind_test:
    a0=np.nonzero(zL[i,:120]>10)
    zobs=zL[i,:120]
    
    nEns=130
    pRateEns = rng.multivariate_normal(pRatem, covT[120:,120:], (nEns))
    #for it in range(2):
    pRateEns[pRateEns<0]=0
    alpha=0.02
    x=pRatem.copy()
    fL=[f_obj(x)]
    for it in range(300):
        z_attd,z_att_f=g_fobj(x)
        g_val=(z_attd[a0[0],:]).T@(z_att_f[a0]-zobs[a0])/4.0+\
            10*(x-pRatem).T@B_inv
        x-=alpha*g_val
        x[x<0]=0
        x[x>100]=100
    fL.append(f_obj(x))
    fL.append(f_obj(pRate[i,:120]))
    print(fL)
    
    #stop
    #z_att,z_attd = calg.computez_d(pRateEns[1,:],pRateD,fint[:120],\
    #                               pZ_snow,pZ_rain,att_snow,att_rain,dr)
    
    #z_att_f = calg.computez(pRateEns[1,:],fint[:120],\
    #                        pZ_snow,pZ_rain,att_snow,att_rain,dr)
        
    #stop
    #covT_ens=np.cov(zEns.T,pRateEns.T)
    #covYY_inv=np.linalg.pinv(covT_ens[:120,:120]+np.eye(120)*4*2)
    #kGainE=covT_ens[120:,:120]@covYY_inv
    #print(pRateEns.mean(axis=0)[:3],pRateEns.std(axis=0)[:3])
    #for ie in range(nEns):
    #    pRateEns[ie,:]=pRateEns[ie,:]+\
    #        0.5*kGainE[:,a0[0]]@(zL[i,a0[0]]-zEns[ie,a0[0]])
   

    x1.append(x[0])
    x2.append(pRate[i,0])
   # x3.append(pRatem[0])
stop
