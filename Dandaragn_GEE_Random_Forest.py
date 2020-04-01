# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 08:49:40 2020

@author: 22631228
"""


import geopandas as gpd
import os
import pandas as pd
import rasterio
import fiona
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from osgeo import gdal, ogr
from collections import OrderedDict
#import time
import itertools
import rasterio.mask
import seaborn as sns
from osgeo import gdal
import numpy as np
from shapely.geometry import Point
from shapely.geometry import Polygon
import scipy
from sklearn.metrics import mean_squared_error
from math import sqrt



'''Creating this random forest model will require a multi-step approach'''

'''
    Inputs: 
    x 1. GNDVI extracted params -> smoothed at 9, 15, 25, 31, 37, 45, 51
    x 2. 4-Band + GNDVI, NDVI and BNDVI 7-day mean and 7-day max
    x 4. Add January and April dates to training and testing
    
    Some other random smoothing params with BNDVI and NDVI
    # 5. 7-day mean NDVI on DOI
    # 6. 7-day mean BNDVI on DOI
    # 7. December mean, max NDVI, GNDVI
    # 8. January mean, max NDVI, GNDVI
    
    3. Post: Remove noisy crown edges
    possible options -> rescale y values to 0-1
'''


''' x values'''
''' FIRST START WITH SMOOTHED WINDOWS'''
''' FIRST START WITH SMOOTHED WINDOWS'''
''' FIRST START WITH SMOOTHED WINDOWS'''






# get the general budmin, budmax, and flower vals from each row
def getGenVals(row, LocOI):
    #prebud range is a range from jan 1 to jan 1 - 130?
    prebud_range = row[: Jan1_loc]
    VAR1_budstart = prebud_range.min()
    budstart_loc = np.where(row == VAR1_budstart)[0][0]
    
    #bud range starts after bud start and stops and bud max
    bud_range = row[budstart_loc: (budstart_loc + 150)]
    VAR2_budmax = bud_range.max()
    budmax_loc = np.where(row == VAR2_budmax)[0][0]

    #flower range starts after budmax
    flower_range = row[budmax_loc: (budmax_loc + 150)]
    VAR3_flowermax = flower_range.min()
    # LocOI values
    # Difference from VAR1_budstart to VAR2_budmax
    VAR4_budDiff = VAR1_budstart - VAR2_budmax
    # Difference from VAR2_budmax to VAR3_flowermax
    VAR5_flowerDiff = VAR2_budmax - VAR3_flowermax
    
    ''' get relationship to LocOI value'''
    VAR6_val = row[LocOI]
    # Difference from VAR2_budmax to LocOI_val 
    VAR7_budDiff = VAR2_budmax - VAR6_val
    # Difference from VAR3_flowermax to LocOI_val
    VAR8_flowerDiff = VAR3_flowermax - VAR6_val
    
    '''return vals'''
    return VAR1_budstart, VAR2_budmax, VAR3_flowermax, VAR4_budDiff, VAR5_flowerDiff, VAR6_val, VAR7_budDiff, VAR8_flowerDiff
    
smooth_folder = r"D:\#DATA\ee_2020\Planet\Dandaragan\GEE_Processed\smooth"



''' this loop will cycle through each date of interest and each GNDVI smothed window, extract the values, 
        and create a new DF:
'''


l = []
datesOI = ['2019-01-15', '2019-03-13', '2019-04-30']
for DOI_date in datesOI:
    
    # writing a program to see which smoothing window is best 
    for window in ['9','15','21','25','31','39','45','51']:
           
        f = r"D:\#DATA\ee_2020\Planet\Dandaragan\GEE_Processed\smooth\smooth{}.csv".format(window)
        print(os.path.exists(f))
        s = pd.read_csv(f)
        s = s[['ID1','mean','uid']]
        
        #sys.exit()
        '''Step1: convert day to doy milliseconds and reshape array'''
        
        s['year'] = s['uid'].str.split("-").str[0]
        s['j'] = s['uid'].str.split("-").str[-1].str.zfill(3)
        s['date'] = pd.to_datetime(s['year'] + s['j'], format='%Y%j')
    
        s['ms'] = s.date.astype('int64')
        
        s = s.sort_values(by='ms')
        ss = s[['ID1','mean','ms']]
        
        df = pd.pivot_table(ss, index = 'ID1', columns = 'ms', values = 'mean').reset_index()
        df['mindex'] = range(0, len(df))
        
        #sys.exit()
    
        #getID = s[s.ID1 == ss.ID1.unique()[0]].reset_index() # finding the first ID1 and getting location of m13 and a12
        getID = s[s.ID1 == '0_ID'].reset_index() # finding the first ID1 and getting location of m13 and a12
        
        #print(getID)
        Jan1_loc = getID[getID.date == '2019-01-01'].index[0]
    
        arr = np.array(df[df.columns[1:]]) # (4533, 320)
    
        DOI = getID[getID.date == DOI_date].index[0]
   
        genVals = np.apply_along_axis(getGenVals, 1, arr, LocOI = DOI)
        df2 = pd.DataFrame(genVals, columns = ['VAR1_budstart','VAR2_budmax','VAR3_flowermax',
                                     'VAR4_budDiff', 'VAR5_flowerDiff', 'VAR6_val', 'VAR7_budDiff', 'VAR8_flowerDiff'])
        df2['mindex'] = range(0, len(df))
        df_wID1 = pd.merge(df2, df, on='mindex')
        df_wID1 = df_wID1[['ID1','mindex','VAR1_budstart','VAR2_budmax','VAR3_flowermax',
                                     'VAR4_budDiff', 'VAR5_flowerDiff', 'VAR6_val', 'VAR7_budDiff', 'VAR8_flowerDiff']]
    
        df_wID1['DOI'] = DOI_date
        df_wID1['smooth'] = window
        
        l.append(df_wID1)
        
        print(window, DOI_date)
        
df_smooth = pd.concat(l) 
df_smooth.to_csv(r"D:\#DATA\ee_2020\Planet\Dandaragan\GEE_Processed2\df_smoothed_windowparams_jan-mar-apr.csv", index = False)






'''adding raw interp values'''
''' getting 7-day smoothed daily values AND mean dec, jan, feb, march, apr values for that pixel
        dec, jan, feb, mar static mean GNDVI, NDVI, b1, b2, b3, b4'''

raw_folder = r"D:\#DATA\ee_2020\Planet\Dandaragan\GEE_Processed\raw_interp_vals"
''' this code will get mean and max values for 7 day and 5 day windows'''

l=[]
for band in ['b1','b2','b3','b4','NDVI','BNDVI','GNDVI']:
#for band in ['GNDVI']:
    path = raw_folder
    f = os.path.join(path, band + ".csv")
    print(os.path.exists(f))
    df = pd.read_csv(f)
    df = df[['ID1','mean','uid']]
    #sys.exit()
    
    df['year'] = df['uid'].str.split("-").str[0]
    df['j'] = df['uid'].str.split("-").str[-1].str.zfill(3)
    df['date'] = pd.to_datetime(df['year'] + df['j'], format='%Y%j')
    df['ms'] = df.date.astype('int64')
    df = df.sort_values(by='ms')
    l2 = []
    for i, grp in df.groupby('ID1'):
        print(i)
        pass
        #sys.exit()
        tf = grp[grp['mean'].isna() == True]
        if len(tf) == 0:
            #pass
            #sys.exit()
            grp['7day_mean'] = grp.rolling(window=7,center=True)['mean'].mean()
            grp['7day_max'] = grp.rolling(window=7,center=True)['mean'].max()
            l2.append(grp)                
    df2 = pd.concat(l2)            
    df2.to_csv(r"D:\#DATA\ee_2020\Planet\Dandaragan\GEE_Processed2\df_raw_mean_max\{}.csv".format(band), index = False)


# datesOI = ['2019-01-15', '2019-03-13', '2019-04-30']
# for DOI_date in datesOI:


'''2. 4-Band + NDVI and BNDVI raw linear interp values on DOI'''
path = r'D:\#DATA\ee_2020\Planet\Dandaragan\GEE_Processed2\df_raw_mean_max'
l=[]
datesOI = ['2019-01-15', '2019-03-13', '2019-04-30']
for DOI_date in datesOI:
    for band in ['b1','b2','b3','b4','NDVI','BNDVI']:
        f = os.path.join(path, band + ".csv")
        print(os.path.exists(f))
        df = pd.read_csv(f)
        df = df[['ID1','mean','uid']]
        df['year'] = df['uid'].str.split("-").str[0]
        df['j'] = df['uid'].str.split("-").str[-1].str.zfill(3)
        df['date'] = pd.to_datetime(df['year'] + df['j'], format='%Y%j')
        df['ms'] = df.date.astype('int64')
        df = df[df.date == DOI_date]
        df = df[['ID1','mean','ms']]
        df['date'] = DOI_date
        df.columns = ['ID1','raw','ms','date']
        df['band'] = band
        l.append(df)

df_raw = pd.concat(l)
df_raw.to_csv(r"D:\#DATA\ee_2020\Planet\Dandaragan\GEE_Processed2\df_raw_vals_jan-mar-apr.csv", index = False)




''' potential forest masking'''
forest = gpd.read_file(r"D:\#DATA\EE_processing\forestmask_intersect.gpkg.shp")
forest['area'] = forest.geometry.area * 1000000000
forest = forest[forest['area'] > 2.7]



'''building my x variable set'''

import geopandas as gpd
import os
import pandas as pd
import rasterio
import fiona
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from osgeo import gdal, ogr
from collections import OrderedDict
#import time
import itertools
import rasterio.mask
import seaborn as sns
from osgeo import gdal
import numpy as np
from shapely.geometry import Point
from shapely.geometry import Polygon
import scipy
from sklearn.metrics import mean_squared_error
from math import sqrt
from functools import reduce




'''1. EXTRACTED VALS GNDVI'''
dfmain = pd.read_csv(r"D:\#DATA\ee_2020\Planet\Dandaragan\GEE_Processed2\df_smoothed_windowparams_jan-mar-apr.csv")
l=[]
for window in list(dfmain.smooth.unique()):
    print(window)
    window = int(window)
    dfmain2 = dfmain[dfmain.smooth == window]
    
    dfmain2 = pd.pivot_table(dfmain2, index = ['ID1', 'DOI','smooth'],values=['VAR1_budstart', 'VAR2_budmax', 'VAR3_flowermax',
                                                           'VAR4_budDiff', 'VAR5_flowerDiff', 'VAR6_val', 'VAR7_budDiff',
                                                           'VAR8_flowerDiff']).reset_index()
    dfmain2 = dfmain2[['ID1','DOI','VAR1_budstart', 'VAR2_budmax',
       'VAR3_flowermax', 'VAR4_budDiff', 'VAR5_flowerDiff', 'VAR6_val',
       'VAR7_budDiff', 'VAR8_flowerDiff']]
    
    
    cols = dfmain2.columns[2:]
    newcols = ['ID1','DOI'] + [n+"w_{}".format(window) for n in cols]
    dfmain2.columns = newcols
    dfmain2['ID1Date'] = dfmain2['ID1'] + "__" + dfmain2['DOI']
    dfmain2.drop("ID1", inplace = True, axis = 1)
    dfmain2.drop("DOI", inplace = True, axis = 1)
    l.append(dfmain2)

df_sgf = reduce(lambda  left,right: pd.merge(left,right,on=['ID1Date'],how='outer'), l)


'''2. SMOOTHED RAW VALUES ON DOI (Jan, Mar, Apr)''' 
df_raw = pd.read_csv(r"D:\#DATA\ee_2020\Planet\Dandaragan\GEE_Processed2\df_raw_vals_jan-mar-apr.csv")
df_raw = pd.pivot_table(df_raw, index = ['ID1', 'date'],columns='band', values = 'raw').reset_index()
df_raw['ID1Date'] =  df_raw['ID1'] + "__" + df_raw['date']
df_raw.drop(['ID1','date'], axis = 1, inplace = True)

'''3. SMOOTHED VALUES W/ 7Day Min and 7Day Max'''
l=[]
datesOI = ['2019-01-15', '2019-03-13', '2019-04-30']
for band in ['b1','b2','b3','b4','BNDVI','GNDVI']:
    df = pd.read_csv(r"D:\#DATA\ee_2020\Planet\Dandaragan\GEE_Processed2\df_raw_mean_max\{}.csv".format(band))
    df = df[df.date.isin(datesOI)]
    #sys.exit()
    df['ID1Date'] =  df['ID1'] + "__" + df['date']
    df = df[['ID1Date','7day_mean', '7day_max']]
    df.columns = ['ID1Date','7day_mean_{}'.format(band), '7day_max_{}'.format(band)]
    l.append(df)
    print( band)

df_smooth = reduce(lambda  left,right: pd.merge(left,right,on=['ID1Date'],how='outer'), l)





''' merge all three dfs together'''
''' merge all three dfs together'''
''' merge all three dfs together'''
''' be sure to get all non int and floats out!!!'''

three_dfs = [df_smooth, df_raw, df_sgf]
dff = reduce(lambda  left,right: pd.merge(left,right,on=['ID1Date'],how='outer'), three_dfs)
dff = dff.dropna()
dff.to_csv(r"D:\#DATA\ee_2020\Planet\Dandaragan\GEE_Processed2\Xvalues.csv", index=False)
























'''building my x variable set'''

import geopandas as gpd
import os
import pandas as pd
import rasterio
import fiona
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from osgeo import gdal, ogr
from collections import OrderedDict
#import time
import itertools
import rasterio.mask
import seaborn as sns
from osgeo import gdal
import numpy as np
from shapely.geometry import Point
from shapely.geometry import Polygon
import scipy
from sklearn.metrics import mean_squared_error
from math import sqrt
from functools import reduce


''' y values'''
dfy = pd.read_csv(r"D:\#DATA\EE_processing\fcount_50s.csv") # from 



datesOI = ['2019-01-15', '2019-03-13', '2019-04-30']
l=[]
for date in datesOI:
    dfyt = dfy.copy()
    if date == '2019-03-13':
        dfyt['date'] = date
        dfyt['ID1Date'] =  dfyt['ID1'] + "__" + dfyt['date']
    else:
        dfyt['fcount'] = 0
        dfyt['fratio'] = 0
        dfyt['date'] = date
        dfyt['ID1Date'] =  dfyt['ID1'] + "__" + dfyt['date']
    l.append(dfyt)
dfy = pd.concat(l)
dfy = dfy[['ID1Date','fcount']]

'''x values'''
dfx = pd.read_csv(r"D:\#DATA\ee_2020\Planet\Dandaragan\GEE_Processed2\Xvalues.csv")





''' running the test'''

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


dff = pd.merge(dfx, dfy, on='ID1Date', how='inner')

dff['ID1'] = dff['ID1Date'].str.split('__').str[0]
dff['date'] = dff['ID1Date'].str.split('__').str[1]


dff = dff[dff.date == '2019-03-13']


y = dff['fcount']

X = dff.drop('fcount', axis = 1)
X = X.drop('ID1Date', axis = 1)
X = X.drop('date', axis = 1)
X = X.drop('ID1', axis = 1)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size= 0.5, random_state = 1)




mod = RandomForestRegressor(n_estimators = 500, max_features =  , max_depth = )
mod.fit(x_train, y_train)
y_pred = mod.predict(x_test)

imps = pd.DataFrame(zip(X.columns, mod.feature_importances_), columns = ['var', 'imp'])
imps = imps.sort_values(by = 'imp', ascending = False)





plt.plot(y_test, y_pred,'o',  label = 'testing pixels')
plt.ylim(0, 0.3)
plt.xlim(0, 0.3)
#plt.show()

y_trainpred = mod.predict(x_train)
plt.plot(y_trainpred, y_train,'o', label = 'training pixels')
plt.ylim(0, 0.3)
plt.xlim(0, 0.3)
plt.legend()
plt.show()






X2 = X.copy()
X2['y_pred'] = mod.predict(X)
shp = gpd.read_file(r"D:\#DATA\ee_2020\Planet\template6m_32750.shp")
shpm = pd.merge(shp, dff, on='ID1')
shpm = gpd.GeoDataFrame(shpm, geometry = shpm.geometry, crs = shp.crs)
shpm = shpm[['ID1','fratio','y_pred','geometry']]
shpm['diff'] = shpm['fratio'] - shpm['y_pred']
shpm.to_file(r"D:\#DATA\ee_2020\Planet\rfresults1_allpixels.shp")



















from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state = 42)
from pprint import pprint

from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)




rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model





dff = pd.merge(dfx, dfy, on='ID1Date', how='inner')
dff['ID1'] = dff['ID1Date'].str.split('__').str[0]
dff['date'] = dff['ID1Date'].str.split('__').str[1]
#dff = dff[dff.date == '2019-03-13']
y = dff['fcount']
X = dff.drop('fcount', axis = 1)
X = X.drop('ID1Date', axis = 1)
X = X.drop('date', axis = 1)
X = X.drop('ID1', axis = 1)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size= 0.5, random_state = 1)


rf_random.fit(x_train, y_train)
rf_random.best_params_










best =    {'n_estimators': 800,
         'min_samples_split': 2,
         'min_samples_leaf': 2,
         'max_features': 'sqrt',
         'max_depth': 50,
         'bootstrap': False}





dff = pd.merge(dfx, dfy, on='ID1Date', how='inner')
dff['ID1'] = dff['ID1Date'].str.split('__').str[0]
dff['date'] = dff['ID1Date'].str.split('__').str[1]
dff = dff[dff.date == '2019-03-13']
y = dff['fcount']









essentials = ['VAR7_budDiffw_9', 'VAR8_flowerDiffw_9', 'VAR2_budmaxw_51',
       'VAR7_budDiffw_21', 'VAR2_budmaxw_45', 'VAR2_budmaxw_39',
       'VAR1_budstartw_39', 'VAR7_budDiffw_51', 'VAR1_budstartw_45',
       'VAR2_budmaxw_9', 'VAR7_budDiffw_45', 'VAR5_flowerDiffw_45']





X = dff.drop('fcount', axis = 1)
X = X.drop('ID1Date', axis = 1)
X = X.drop('date', axis = 1)
X = X.drop('ID1', axis = 1)
X = X[essentials]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size= 0.4, random_state = 1)
mod = RandomForestRegressor(n_estimators = 800,
                             min_samples_split = 2,
                             min_samples_leaf = 2,
                             max_features = 'sqrt',
                             max_depth = 50,
                             bootstrap = False )

mod.fit(x_train, y_train)
y_pred = mod.predict(x_test)

imps = pd.DataFrame(zip(X.columns, mod.feature_importances_), columns = ['var', 'imp'])
imps = imps.sort_values(by = 'imp', ascending = False)


plt.plot(y_test, y_pred,'o',  label = 'testing pixels')
plt.ylim(0, 25000)
plt.xlim(0, 25000)


y_trainpred = mod.predict(x_train)
plt.plot(y_trainpred, y_train,'o', label = 'training pixels')
plt.legend()
plt.show()














