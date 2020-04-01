# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 12:43:48 2020

@author: 22631228
"""

# Download 4 band planet scope scenes for Ba

import os
import json
import requests
from requests.auth import HTTPBasicAuth
import time
from os import environ
from tqdm import tqdm
import zipfile

# Some initial variables
# Set API key (this should to be an environment variable)
api_key = "f6092e3140d849fe82dff89dffc64634"

item_type = "PSScene4Band"

asset_type = "analytic_sr"

start_date = "2019-06-01T00:00:00.000Z"

end_date = "2019-09-05T00:00:00.000Z"

# Filter scenes by cloud cover, date, and study site
study_area = {
    "type": "Polygon",
    "coordinates": [
          [
            [
              116.17584943771361,
              -31.995010634179685
            ],
            [
              116.18241548538207,
              -31.995010634179685
            ],
            [
              116.18241548538207,
              -31.98753089855013
            ],
            [
              116.17584943771361,
              -31.98753089855013
            ],
            [
              116.17584943771361,
              -31.995010634179685
            ]
          ]
        ]
}

# get images that overlap with our AOI
geometry_filter = {
  "type": "GeometryFilter",
  "field_name": "geometry",
  "config": study_area
}

# get images acquired within a date range
date_range_filter = {
  "type": "DateRangeFilter",
  "field_name": "acquired",
  "config": {
    "gte": start_date,
    "lte": end_date
  }
}

# only get images which have <50% cloud coverage
cloud_cover_filter = {
  "type": "RangeFilter",
  "field_name": "cloud_cover",
  "config": {
    "lte": 0.5
  }
}

# combine our geo, date, cloud filters
combined_filter = {
  "type": "AndFilter",
  "config": [geometry_filter, date_range_filter, cloud_cover_filter]
}

# API request object
search_request = {
  "interval": "day",
  "item_types": [item_type],
  "filter": combined_filter
}

# fire off the POST request
search_result = \
  requests.post(
    'https://api.planet.com/data/v1/quick-search',
    auth=HTTPBasicAuth(api_key, ''),
    json=search_request)

#print(json.dumps(search_result.json(), indent=1))

image_ids = [feature['id'] for feature in search_result.json()['features']]
print(image_ids)
#image_ids = image_ids[0:4] # this is just for testing

# study area for clipping server side
aoi_json = '''{
    "type": "Polygon",
    "coordinates": [
          [
            [
              116.17584943771361,
              -31.995010634179685
            ],
            [
              116.18241548538207,
              -31.995010634179685
            ],
            [
              116.18241548538207,
              -31.98753089855013
            ],
            [
              116.17584943771361,
              -31.98753089855013
            ],
            [
              116.17584943771361,
              -31.995010634179685
            ]
          ]
        ]
}'''

# iterate over each scene in study area, within date range, and with < 50% cloud cover











#r = requests.get(asset['_links']['_self'], auth=(api_key, ''))
QUOTA_URL = 'https://api.planet.com/auth/v1/experimental/public/my/subscriptions'

r = requests.get(QUOTA_URL, auth=(api_key, ''))
l = r.json()
l



# clip scene and download
loss = []
for i in image_ids:
    print(i)

    # Sent Scene ID
    scene_id = i
    txtfile = r"D:\#DATA\EE_processing\downloaded_scenes.txt"
    # create / open file containing scenes already downloaded
    f = open(txtfile, "r")
    downloaded_scenes = f.read()

    if i not in downloaded_scenes:

        # Construct clip API payload
        clip_payload = {
            'aoi': json.loads(aoi_json),
            'targets': [
                {
                    'item_id': scene_id,
                    'item_type': item_type,
                    'asset_type': asset_type
                }
            ]
        }
    
        try:
            # Request clip of scene (This will take some time to complete)
            request = requests.post('https://api.planet.com/compute/ops/clips/v1', auth=(api_key, ''), json=clip_payload)
            clip_url = request.json()['_links']['_self']
    
            # Poll API to monitor clip status. Once finished, download and upzip the scene
            clip_succeeded = False
            while not clip_succeeded:
    
                # Poll API
                check_state_request = requests.get(clip_url, auth=(api_key, ''))
    
                # If clipping process succeeded , we are done
                if check_state_request.json()['state'] == 'succeeded':
                    clip_download_url = check_state_request.json()['_links']['results'][0]
                    clip_succeeded = True
                    print("Clip of scene succeeded and is ready to download")
    
                    # Still activating. Wait 1 second and check again.
                else:
                    print("...Still waiting for clipping to complete...")
                    time.sleep(15)
    
            # Download clip
            response = requests.get(clip_download_url, stream=True)
            with open( scene_id + '.zip', "wb") as handle:
                for data in tqdm(response.iter_content()):
                    handle.write(data)
    
            # Unzip file
            zipped_item = zipfile.ZipFile(scene_id + '.zip')
            zipped_item.extractall(scene_id)
    
            del zipped_item
            # Delete zip file
            os.remove(scene_id + '.zip')
    
            # add scene id to list of downloaded scenes to avoid duplicating downloads
            f.close()
            f = open(txtfile, "a")
            f.write(i+'\n')
            f.close()
            print('finished downloading scene')
        except:
            loss.append(i)
            print(loss, 'loss')
    



# r = requests.get(asset['_links']['_self'], auth=(api_key, ''))
# r.ok # checking to see if the asset link is all good 

# QUOTA_URL = 'https://api.planet.com/auth/v1/experimental/public/my/subscriptions'

# r = requests.get(QUOTA_URL, auth=(api_key, ''))
# l = r.json()


# #2645

