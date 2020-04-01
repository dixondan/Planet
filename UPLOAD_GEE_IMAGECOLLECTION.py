# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 16:58:15 2020

@author: 22631228
"""

import pandas as pd
import subprocess
import os
import time
df = pd.read_csv(r"D:\#DATA\imagery\process_dandaragan\images\ps\metadata_ps2.csv")
# for ps
rows = zip(df.id_no, df['system:time_start'])
l=[]
for f in rows:
    #time.sleep(30)
    print(f)
    tif = f[0] + '.tif'
    ID = f[0]
    t = f[1]
    try:
        cmd = ['earthengine','upload','image','--asset_id=users/files/Dandaragan_PS2/{}'.format(ID),'--time_start={}'.format(t),'gs://gee-bucekt1/{}'.format(tif)]
        subprocess.call(cmd)
    except:
        cmd = ['earthengine','upload','image','--asset_id=users/files/Dandaragan_PS2/{}'.format(ID),'--time_start={}'.format(t),'gs://gee-bucekt1/{}'.format(tif)]
        l.append(cmd)

    
    
    
    
    
    
    
    
# for s2
import pandas as pd
import subprocess
import os
import time
df = pd.read_csv(r"D:\#DATA\imagery\process_dandaragan\images\s2\metadata_s2.csv")

rows = zip(df.id_no, df['system:time_start'])
l=[]
for f in rows:
    time.sleep(30)
    print(f)
    tif = str(f[0]) + '.tif'
    ID = f[0]
    t = f[1]
    try:
        cmd = ['earthengine','upload','image','--asset_id=users/files/Dandaragan_S2_pre/{}'.format(ID),'--time_start={}'.format(t),'gs://gee-bucket1/{}'.format(tif)]
        subprocess.call(cmd)
    except:
        cmd = ['earthengine','upload','image','--asset_id=users/files/Dandaragan_S2_pre/{}'.format(ID),'--time_start={}'.format(t),'gs://gee-bucket1/{}'.format(tif)]
        l.append(cmd)
       














