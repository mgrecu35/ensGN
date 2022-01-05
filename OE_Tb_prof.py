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
tb=fh["tb35"][:]
#d={"zKa":zKaL,"zKu":zKuL,"zKuC":zKucL,"pRate":pRateDL,"nodes":[-80,0,26]}


from sklearn import preprocessing
scaler_obs  = preprocessing.StandardScaler()
scaler_true  = preprocessing.StandardScaler()
zka_obs[zka_obs<0]=0
zka_obs = scaler_obs.fit_transform(zka_obs[:,10:90])

#zka_true_sc = scaler_obs.fit_transform(attka[:,0:100])

import sklearn.decomposition as pca
zpca_obs = pca.PCA()
zpca_obs.fit(zka_obs)

#zpca_true = pca.PCA(n_components=7)
#zpca_true.fit(zka_true_sc)

pca_obs=zpca_obs.transform(zka_obs)
pca_obs[:,7]=(tb-253)/20.


#pca_true=zpca_true.transform(zka_true_sc)
from sklearn.model_selection import train_test_split
X_train, X_test, \
    y_train, y_test \
    = train_test_split(pca_obs[:,0:7], pRate[:,0], \
                       test_size=0.33, random_state=42)
