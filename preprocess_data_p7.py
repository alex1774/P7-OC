import pandas as pd
import numpy as np





def col_to_bool(dataf):
    liste_col=[]
    for i in range(0, len(dataf.columns)):
        if (dataf.iloc[:,i].dtypes == 'int64') & (len(dataf.iloc[:,i].value_counts())==2):
            liste_col.append(i)
    return(liste_col)

def visu_col_to_bool (dataf, liste):
    for i in liste:
        print(i, ' ',dataf.columns[i])
        display(dataf[dataf.columns[i]].value_counts())
        print()

def transform_data_to_predict(dataf):
    dataf.iloc[:,col_to_bool(dataf)]=dataf.iloc[:,col_to_bool(dataf)].astype('bool',copy=False)
    dataf['AVG_NOTE']= np.mean(dataf.loc[:,['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']],axis=1)
    return dataf
