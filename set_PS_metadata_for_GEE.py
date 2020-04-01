# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 07:40:36 2020

@author: 22631228
"""
import os
import pandas as pd

fs = r"D:\#DATA\imagery\process_dandaragan\images\ps\shifted_rasters"

rasters = os.listdir(fs)

l=[]
for f in rasters:
    #print(f)
    date = f.split("_")[0]
    print(date)

    year = date[0:4]
    month = date[4:6]
    day = date[6:8]
    
    timestamp1 = year + month + day 
    t = pd.to_datetime(timestamp1,format='%Y%m%d')  
    doy = str(t.dayofyear)
    doy = doy.zfill(3)
    
    hour = '12'
    timestamp2 = year + str(doy) + hour

    gee_timestamp = int(pd.to_datetime(timestamp2,format='%Y%j%H').value // 10**6)
    id_no = f.split(".")[0]
    cloud_cover = 0
    SENSOR = "PS"
    spatial_coverage = "Null"
    pair = (id_no, date, gee_timestamp, cloud_cover, SENSOR, spatial_coverage, year, month, day, timestamp2)
    l.append(pair)
    
dfout = pd.DataFrame(l, columns = ['id_no', 'date', "system:time_start", "cloud_cover", "SENSOR", "spatial_coverage","year","month","day","timestamp2"])




#%%

print(dfout.head())

dfout.to_csv(r"D:\#DATA\imagery\process_dandaragan\images\ps\metadata_ps2.csv", index=False)



#%%


fs = r"D:\#DATA\imagery\sentinel2\processed\tifs_4bands_dandaragan\S2_processed"

rasters = os.listdir(fs)

l=[]
for f in rasters:
    #print(f)
    date = f.split("_")[0]
    print(date)

    year = date[0:4]
    month = date[4:6]
    day = date[6:8]
    
    timestamp1 = year + month + day 
    t = pd.to_datetime(timestamp1,format='%Y%m%d')  
    doy = t.dayofyear

    hour = '12'
    timestamp2 = year + str(doy) + hour

    gee_timestamp = int(pd.to_datetime(timestamp2,format='%Y%j%H').value // 10**6)
    id_no = f.split(".")[0]  
    pair = (id_no, date, gee_timestamp)
    l.append(pair)
    
dfout = pd.DataFrame(l, columns = ['id_no', 'date', "system:time_start"])


print(dfout.head())

dfout.to_csv(r"D:\#DATA\imagery\process_dandaragan\images\s2\metadata_s2.csv", index=False)








