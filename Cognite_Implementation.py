# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 15:59:37 2019

@author: jarl
"""

from cognite.client import CogniteClient

import numpy as np
import scipy as sci
import scipy.interpolate as interpolate

import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

import time

from utils import *

import mpmath as mp


CogniteDataFusionAPI_key = "MDcxM2M1NjQtMGRjYS00YjQ3LTkxMGYtNzhhNzIwY2Q1YzE5"

client = CogniteClient(api_key=CogniteDataFusionAPI_key, client_name="COGNITE_CLIENT_NAME", project="publicdata")

# if you need to get the files from the publicdata project, uncomment this section:
'''
for file_instance in client.files.list():
    client.files.download('.\\Retrived_docs\\' + file_instance.name + '.'+ file_instance.mime_type, file_instance.id)
'''



AL = client.assets.list()
Asset_names = []
Asset_ids = {}
root_id = client.assets.list()[0].root_id
for asset_item in client.assets.list():
    Asset_names.append(asset_item.name)
    Asset_ids.update({asset_item.name : asset_item.id})

Asset_ids_all = {}
Parent_nodes = {}
TimeSeries_all = {}


fetch_timer = datetime.now()
ALL_ASSETS = client.assets.retrieve_subtree(id=root_id)
fetch_timer -= datetime.now()



ALL_TIME_SERIES = ALL_ASSETS.time_series()

for asset_item in ALL_ASSETS[1:]:
    Asset_ids_all.update({asset_item.name : asset_item.id})
    
    temp_timeseries = {}
    fetch_timer = datetime.now()
    for item in asset_item.time_series():
        temp_timeseries.update({item.name: item.id})
    TimeSeries_all.update({asset_item.name: temp_timeseries})
    fetch_timer -= datetime.now()
    
    
    if asset_item.parent_id != None:
        new_key = True
        ###if asset_item.parent() is not None:
        ##key = 1
        for key in Parent_nodes.keys():
            if key == asset_item.parent_id and asset_item.id != None :
                new_key = False
                temp_array = Parent_nodes[key]
                temp_array.append(asset_item.id)
                Parent_nodes.update({key: temp_array })
                break
        if new_key and asset_item.id != None:
            Parent_nodes.update({ asset_item.parent_id : [asset_item.id] })



TimeSeries_ids_all = {}
for time_ser_item in ALL_TIME_SERIES:
    TimeSeries_ids_all.update({time_ser_item.name : time_ser_item.id})
       

"""
Case analysis: 
Check the stability of the independent (assumingly) transfer function of the
timeseries of "VAL_23-TT-92602:X.Value".

Other analyses may be performed by selecting one from ALL_TIME_SERIES

"""
SeriesName = 'VAL_23-TT-92602:X.Value'


granularity = '1s'
aggregates = ['average']

start_series_time = datetime(2019, 8, 20)
end_series_time = datetime.now()


time_start = str( int( ( end_series_time - start_series_time ).total_seconds() ) ) + "s-ago"  #str( int(time_since_dawn.days) ) +"d-ago" # str( int( ( start_series_time - end_series_time ).days  ) ) # str( int(time_since_dawn.days) ) +"d-ago"
time_end = "now"
    
print("Fetching data....")
fetch_timer = time.time()

TimeSeriesData = client.datapoints.retrieve_dataframe(external_id=SeriesName, start=time_start, end=time_end, aggregates=aggregates, granularity=granularity)

print("Time to get data: " + str( time.time() - fetch_timer) + " seconds")


"""
Implementation of time series analysis
1st: prepearing the obtained data, such that t = secoinds, t0 = 0.
"""


mp.dps = 30
max_size = 6000
window_Mean = 20  # average weigth window

# apply the selected complex numbers to test on the transfer function i.e. omega*i
omega = np.logspace(32, 34, num=20)

# Here, the numerical approximate begins, by taking creating a time series, of t and f(t) respectively
T =  np.array( ( TimeSeriesData.index[-max_size:] - TimeSeriesData.index[-max_size] ).total_seconds() ) 
F =  np.array( TimeSeriesData[TimeSeriesData.columns[0]].rolling(window=window_Mean, win_type=None).mean()[-max_size:]  )

# Convert the float points to mpmath points..
for i in range (max_size):
    T[i] = mp.mpf(T[i])
    F[i] = mp.mpf(F[i])



"""
2nd: Computing the laplace integrand with the specialized simpsonsaitkens method
"""

# Now we can calculate the actual laplace transforms:
SolutionVector  = [complex(0,0)]*len(omega)

fetch_timer = time.time()
for i in range(len(omega)):
    SolutionVector[i] = simpson_nonuniform_AitkensExtrap(T, [ dLaplace_dt(F[j], T[j], complex(0, omega[i]) ) for j in range(len(T))  ], maxAitkensIterations=int(max_size/30) )
    SolutionVector[i]  = complex( float(SolutionVector[i].real), float(SolutionVector[i].imag ))

# Convert array type to allow for compact operator notation, e.g. vec.Real, vec.mean, etc.
SolutionVector = np.array(SolutionVector)

print("Time to compute Laplace tranforms: " + str( time.time() - fetch_timer) + " seconds")

plt.plot(T, F, '-k')
plt.title(SeriesName + " in the time range: " + str(TimeSeriesData.index[-max_size] ) + " to " + str(TimeSeriesData.index[-1] ))
plt.xlabel('Time (seconds)')
plt.ylabel('Sensor value')
plt.ylim([float( F.min() ), float( F.max() ) ])
plt.show()

plt.plot( SolutionVector.real, SolutionVector.imag, '-b')
plt.title("Nyquist plot of " + SeriesName + " for " + str(TimeSeriesData.index[-1] ) )
plt.xlabel('\Re(L[f(t)](\omega i))')
plt.ylabel('\Im(L[f(t)](\omega i))')
plt.show()


if  SolutionVector.real.mean() +  SolutionVector.real.std() * 2 > - 1.0:
    print(SeriesName + " is stable in the region " + str(TimeSeriesData.index[-max_size] ) + \
          " to " + str(TimeSeriesData.index[-1] ) + " as the mean+2std > -1.0 (95% statistical significance)")

