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
