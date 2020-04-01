# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 17:08:26 2020

@author: 22631228
"""




'''Processing Drone + Planet + S2

    Data uploading:
        - Have to batch upload to image collection in GEE

    Pre-processing:
        1. Read in and organize the time series data:
            - Resampling and shifting to get 5 datasets
                3m PS
                6m PS
                10m PS (have to align)
                10m S2 (have to align)
                10m PS + S2 (have to align)
                - 3m Planet are registered to drone image
                - 3m Planet is resampled to 6m and 10m
                - S2 is registered to resampled Planet 10m
            - Export time series to pd.Series
        2. Smoothing
            - Interpolate w/ Pandas
            - Fit w/ Scipy
            - Following window sizes:
                21
                41
                61
                81
   
    Processing:
        1. Extract metrics from each pixel
            - DOY GNDVI value
            - DOY 7-day smoothed 4 bands
            - Peak flower
            - Peak bud
            - Start bud
            - Peak flower value - DOY GNDVI value
        2. Extract flower pixel counts for two drone images and each pixel resolution
       
   
    Compare results across pixel sizes

'''



import ee
import numpy as np
import pandas as pd
import sys
import os

ee.Initialize()

AOI_coords = [
            [115.75528237584325,-30.892264163971],
            [115.7551107144663,-30.8922457504062],
            [115.75526091817113,-30.89816552907025],
            [115.75605485203954,-30.898137910423955],
            [115.75791094067785,-30.89811949798866], 
            [115.75783583882543,-30.896876650420314],
            [115.75693461659642,-30.896858237742475],
            [115.75690243008825,-30.895836328570454],
            [115.75755688908788,-30.89490647381079],
            [115.75756761792394,-30.893810890774144],
            [115.75727793935033,-30.893645171240212],
            [115.75723502400609,-30.892521953506435],
            [115.75681659939977,-30.892439092660002],
            [115.75658056500646,-30.892273370752072],
            [115.75528237584325,-30.892264163971]
          ]

AOI = ee.Geometry.Polygon(AOI_coords)


# then add GNDVI to each reso
def addGNDVI(image):
  gndvi = image.normalizedDifference(['b4', 'b2']).rename('GNDVI')
  return image.addBands(gndvi)

def resample_reso_6m(image):
    crs = 'EPSG:32750'
    scale = 6
    # proj = image.projection().getInfo()
    # crs = proj['crs']
    mode = 'bilinear'
    image_resampled = image.resample('bilinear').reproject(crs=crs, scale=scale)
    image_resampledAOI = image_resampled.clip(AOI)
    return image_resampledAOI



PS = ee.ImageCollection("users/files/Dandaragan_PS")

#for reso in ['3','6','10']:
        
    # first resample and reproject if 6 or 10 m resos
    
    
    
PS_adj = PS.map(resample_reso_6m)
    
    
image = PS_adj.first()
    


    
    task = ee.batch.Export.image(image, '6m34reduiee')
    task.start()














var resample_6m = function(image){
    var scale = 6;
    var mode = 'bilinear';
    var crs = image.projection();
    var resampled_image = image.resample(mode).reproject({
        crs: crs,
        scale: scale
    });
    return resampled_image;
}













    
    