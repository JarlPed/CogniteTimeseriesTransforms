# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 15:59:37 2019

@author: jarl
"""

from cognite.client import CogniteClient

import cognite
import re
import numpy as np
import scipy as sci
import scipy.interpolate as interpolate

import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt

import time
import json
#import math
from influxdb import InfluxDBClient


# register matplotlib onverter to avoid warnings
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


CogniteDataFusionAPI_key = "YOUR KEY GOES HERES"

client = CogniteClient(api_key=CogniteDataFusionAPI_key, client_name="COGNITE_CLIENT_NAME", project="publicdata")

# to get the files from the publicdata project:
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



#ALL_TIME_SERIES = ALL_ASSETS.time_series()

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


'''
Parent_node_names = {}
for parent_node_id in Parent_nodes.keys():
    for item in ALL_ASSETS:
        if item.id == parent_node_id:
            temp_Parent_name = item.name
    temp_array_children = []
    for children_id in Parent_nodes[parent_node_id]:
        for item in ALL_ASSETS:
            if item.id == children_id:
                temp_array_children.append(item.name)
    Parent_node_names.update({temp_Parent_name: temp_array_children})

'''

    




# attempt to find any data that does not return error code 400:
#suc_id = {}
#for key in Asset_ids_all.keys():
#    try:
#        temp = client.datapoints.retrieve_latest(id=Asset_ids_all[key])
#        suc_id.update({key: Asset_ids_all[key]})
        
#    except:
#        continue

#TimeSeries_ids_all = {}
#for time_ser_item in ALL_TIME_SERIES:
#    TimeSeries_ids_all.update({time_ser_item.name : time_ser_item.id})
       



#scrubber_file_name = 'PH-ME-P-0152-001'
   
#for file_instance in client.files.list():
#    if file_instance.name == scrubber_file_name:
#        Scrubber_Diagram_metadata = file_instance.to_pandas()



#scrubber_level_working_setpoint = 'VAL_23-LIC-92521:Control Module:YR'
#scrubber_level_measured_value  = 'VAL_23-LIC-92521:Z.X.Value'
#scrubber_level_output  = 'VAL_23-LIC-92521:Z.Y.Value'
#all_ts_names = [scrubber_level_working_setpoint, scrubber_level_measured_value, scrubber_level_output]


time_since_dawn = datetime.now() - datetime(1970, 1,1)
start = '52w-ago'  # str(int( time_since_dawn.days / 7 ) ) + "w-ago"
end = 'now'
aggregates = ['average'] # average; interpolation; min, max, sum, count, step interpolation; continous variance; discrete variance; total variation
granularity = '1d'

#DataFrameExample = client.datapoints.retrieve_dataframe(id=3720706503095541, start=start, end=end, aggregates=aggregates, granularity=granularity)
#DataDictExample = client.datapoints.retrieve_dataframe_dict(external_id=all_ts_names, start=start, end=end, aggregates=aggregates, granularity=granularity)
#SingleDictExample = client.datapoints.retrieve_dataframe_dict(external_id='VAL_23-LIC-92521:Control Module:YR', start=start, end=end, aggregates=aggregates, granularity=granularity)

# For these functions, mid point rule has to be applied considering a variable dt_array values

dLaplace_dt = lambda f_t, t, s : f_t * np.exp(-s * t) 
LaplaceTrans = lambda f_t, t, dt, s : dLaplace_dt(f_t, t, s)  *dt[i]

DiscreteLaplaceTrans_midPoint = lambda f_t_array, t_array, dt_array, s : sum( np.array( [ LaplaceTrans(f_t=f_t_array[i], t=t_array[i], dt=dt_array[i], s=s) for i in range(len(t_array)) ])  )
DiscreteLaplaceTrans_Trapz = lambda f_t_array, t_array, dt_array, s : 0.5 * ( LaplaceTrans(f_t=f_t_array[0], t=t_array[0], dt=dt_array[0], s=s) \
                                                                           + LaplaceTrans(f_t=f_t_array[-1], t=t_array[-1], dt=dt_array[-1], s=s) \
                                                                           + 2* sum( np.array( [ LaplaceTrans(f_t=f_t_array[i], t=t_array[i], dt=dt_array[i], s=s) for i in range(1, len(t_array)) -1 ])  ) )

DiscreteUnilateralZTransform = lambda f_array, z : sum(np.array([ f_array[i]*z**(-i) for i in range(len(f_array)) ]) )
    
def simpson_nonuniform(x, f):
    """
    copy pasta from: https://en.wikipedia.org/wiki/Simpson%27s_rule#Composite_Simpson's_rule_for_irregularly_spaced_data
    Simpson rule for irregularly spaced data.

        Parameters
        ----------
        x : list or np.array of floats
                Sampling points for the function values
        f : list or np.array of floats
                Function values at the sampling points

        Returns
        -------
        float : approximation for the integral
    """
    # For Quality assurance, lets add a assert:
    assert(len(x) == len(f) )
    N = len(x) - 1
    h = np.diff(x)

    result = 0.0
    for i in range(1, N, 2):
        hph = h[i] + h[i - 1]
        result += f[i] * ( h[i]**3 + h[i - 1]**3
                           + 3. * h[i] * h[i - 1] * hph )\
                     / ( 6 * h[i] * h[i - 1] )
        result += f[i - 1] * ( 2. * h[i - 1]**3 - h[i]**3
                              + 3. * h[i] * h[i - 1]**2)\
                     / ( 6 * h[i - 1] * hph)
        result += f[i + 1] * ( 2. * h[i]**3 - h[i - 1]**3
                              + 3. * h[i - 1] * h[i]**2)\
                     / ( 6 * h[i] * hph )

    if (N + 1) % 2 == 0:
        result += f[N] * ( 2 * h[N - 1]**2
                          + 3. * h[N - 2] * h[N - 1])\
                     / ( 6 * ( h[N - 2] + h[N - 1] ) )
        result += f[N - 1] * ( h[N - 1]**2
                           + 3*h[N - 1]* h[N - 2] )\
                     / ( 6 * h[N - 2] )
        result -= f[N - 2] * h[N - 1]**3\
                     / ( 6 * h[N - 2] * ( h[N - 2] + h[N - 1] ) )
    return result    


def Arg (complexNum):
    if complexNum.imag != 0:
        return 2*np.arctan( ( complexNum.real**2 + complexNum.imag**2 - complexNum.real) / complexNum.imag )
    elif complexNum.real == 0 and complexNum.imag == 0 :
        return np.nan
    elif complexNum.real > 0:
        return 0.0
    elif complexNum.real < 0:
        return np.pi
    



def pushFiniteArray (array, size_n, push_elem):
    if len(array) < size_n -0.01:
        return array.append(push_elem)
    else:
        return array[1:].append(push_elem)


granularity = '1s'
max_size = 100
array_stack = []


start_series_time = datetime(2019, 8, 15)
end_series_time = datetime.now()


time_start = str( int( ( end_series_time - start_series_time ).total_seconds() ) ) + "s-ago"  #str( int(time_since_dawn.days) ) +"d-ago" # str( int( ( start_series_time - end_series_time ).days  ) ) # str( int(time_since_dawn.days) ) +"d-ago"
time_end = "now"
    
print("Fetching data....")
fetch_timer = time.time()
#new_data = client.datapoints.retrieve_dataframe(external_id='VAL_23-LIC-92521:Control Module:YR', start=start, end=end, aggregates=aggregates, granularity=granularity)
new_data = client.datapoints.retrieve_dataframe(external_id='VAL_23-TT-92602:X.Value', start=time_start, end=time_end, aggregates=aggregates, granularity=granularity)

print("Time to get data: " + str( time.time() - fetch_timer) + " seconds")


Smoothening_factor = 1

fetch_timer = time.time()
#interpolateFunction = interpolate.UnivariateSpline( ( new_data.index - datetime(1970, 1,1)).total_seconds() , new_data[ new_data.columns[0]].values, s=Smoothening_factor  )
print("Time to spline interpolate: " + str( time.time() - fetch_timer) + " seconds")
#new_data[ new_data.columns[0] + " Spline"] = interpolateFunction(( new_data.index - datetime(1970, 1,1)).total_seconds())

window_Mean = 40
new_data[new_data.columns[0] + " Rolling_mean (w=" + str(window_Mean) +")" ] = new_data[ new_data.columns[0] ].rolling_mean(window=window_Mean)

new_data[new_data.columns[0] +" Noize (w=" + str(window_Mean) + ")" ] = new_data[new_data.columns[0]] - new_data[new_data.columns[0] + " Rolling_mean (w=" + str(window_Mean) +")"]


'''
new_data['dt (seconds)'] = new_data.index.to_series().diff().dt.seconds
new_data['time passing (seconds)'] = new_data['dt (seconds)'].cumsum()
values_new_data = new_data[new_data.columns[0]].values
'''

'''
# No need for mid point now, as Simpson_nonuniform is used
mid_point = [0.0]*(len(values_new_data))
for i in range(len(values_new_data)-1):
    mid_point[i+1] =  ( values_new_data[i] + values_new_data[i+1] ) /2.0


new_data[new_data.columns[0] + " Midpoint"] = mid_point
'''




'''
omega = np.linspace(-1e5, 1e5, 100)
Laplace_Continous = [0]*len(omega)
for i in range(len(omega)):
    Laplace_Continous[i] = DiscreteLaplaceTrans(new_data[new_data.columns[0] + " Midpoint"][1:].values, new_data['time passing (seconds)'][1:].values, new_data['dt (seconds)'][1:].values, complex(0, omega[i]) )
plt.plot([Laplace_Continous[i].real for i in range(len(Laplace_Continous)) ], [Laplace_Continous[i].imag for i in range(len(Laplace_Continous)) ] )
plt.title("Discrete Laplace analysys analysis of L(" + str( omega[0] ) +"< \omega < "+ str(omega[-1]) + ")"   )
plt.xlabel('\Re(\mathcal{L}(\omega \cdot i))')
plt.ylabel('\Im(\mathcal{L}(\omgea \cdot i))')
plt.show()

'''
 



fetch_timer = time.time()
s = np.linspace(0, 4, 20)
omega = np.logspace(0, 4, 20) # here range is 10^0 to 10^4
Laplace_Continous =  [0]*len(omega)  # [0]*len(s) #  [0]*len(omega) 

maxsize_TimeSeries = 100
if maxsize_TimeSeries > len( new_data.index) :
    maxsize_TimeSeries = len( new_data.index)



startTimeSeries = new_data.index[-maxsize_TimeSeries].to_pydatetime()
new_data['t (s)'] = ( new_data.index - startTimeSeries ).seconds


'''
for i in range(len(s)):
    Laplace_Continous[i] = simpson_nonuniform( new_data['t (s)'][startTimeSeries:].values, \
                                                 [ dLaplace_dt(new_data[new_data.columns[0]][startTimeSeries:][j], \
                                                               new_data['t (s)'][startTimeSeries:][j], s[i])  for j in range(maxsize_TimeSeries)]  )


'''

for i in range(len(omega)):
    Laplace_Continous[i] = simpson_nonuniform( new_data['t (s)'][startTimeSeries:].values, \
                                                 [ dLaplace_dt(new_data[new_data.columns[0]][startTimeSeries:][j], \
                                                               new_data['t (s)'][startTimeSeries:][j], complex(0, omega[i]) )  for j in range(maxsize_TimeSeries)]  )




'''
for i in range(len(s)):
    Laplace_Continous[i] = DiscreteLaplaceTrans(new_data[new_data.columns[0] + " Midpoint"][1:].values, new_data['time passing (seconds)'][1:].values, new_data['dt (seconds)'][1:].values, complex(0, s[i]) )
'''
    

    

print("Time to compute Laplace transform data: " + str( time.time() - fetch_timer) + " seconds")



'''
#plt.subplot(311)
plt.plot(s, Laplace_Continous, '-k')
plt.title("Discrete Laplace analysys analysis of L(" + str( s[0] ) +"< s < "+ str(s[-1]) + ")"   )
plt.xlabel('s (1/second)')
plt.ylabel('L(s)')
plt.yscale('log')
plt.grid(b=True)
plt.show()


#plt.subplot(312)
plt.plot(new_data.index.values, new_data[new_data.columns[0]].values, '-k')
plt.title("Time series of " +  new_data.columns[0] + " fetched"  )
plt.xlabel('date (yyyy-mm-dd)')
plt.ylabel('(deg. C)')
plt.grid(b=True)
plt.show()


#plt.subplot(313)
plt.plot(new_data['t (s)'][startTimeSeries:].values , new_data[new_data.columns[0]][startTimeSeries:].values, '-k')
plt.title("Portion of " +  new_data.columns[0] + " analyzed (" + str(maxsize_TimeSeries) + " points)" )
plt.xlabel('t (s)')
plt.ylabel('(deg. C)')
plt.grid(b=True)
plt.show()
'''

max_Amp = max(  [( abs( Laplace_Continous[i] ) ) for i in range(len(Laplace_Continous)) ] )

plt.subplot(211)
plt.plot(omega, [  20 * np.log10 ( abs( Laplace_Continous[i] ) / max_Amp ) for i in range(len(Laplace_Continous)) ], '-k')
plt.title("Bode plots of " + new_data.columns[0] +") for "+ str( omega[0] ) +"< \omega < "+ str(omega[-1]) + ")"   )
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.xscale('log')
plt.grid(b=True)
plt.show()


plt.subplot(212)
plt.plot(omega, [ 180.0/np.pi * Arg(Laplace_Continous[i]) for i in range(len(Laplace_Continous)) ], '-k')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (deg)')
plt.xscale('log')
plt.grid(b=True)
plt.show()





plt.plot( [Laplace_Continous[i].real for i in range(len(Laplace_Continous)) ], [Laplace_Continous[i].imag for i in range(len(Laplace_Continous)) ], '-k')
plt.title("Nyquist of " + new_data.columns[0] +") for "+ str( omega[0] ) +"< \omega < "+ str(omega[-1]) + ")"   )
plt.xlabel('\Re(L(j \omega))')
plt.ylabel('\Im(L(j \omega))')
plt.grid(b=True)
plt.show()







plt.plot(new_data['t (s)'][startTimeSeries:].values , new_data[new_data.columns[0]][startTimeSeries:].values, '-k')
plt.title("Portion of " +  new_data.columns[0] + " analyzed (" + str(maxsize_TimeSeries) + " points)" )
plt.xlabel('t (s)')
plt.ylabel('(deg. C)')
plt.grid(b=True)
plt.show()


plt.plot(new_data.index.values, new_data[new_data.columns[0] +" Noize (w=" + str(window_Mean) + ")" ].values)
plt.title(new_data.columns[0] + " Noize " )
plt.xlabel('date')
plt.ylabel('(deg. C)')
plt.grid(b=True)
plt.show()




'''
plt.plot(new_data['t (s)'][startTimeSeries:].values , new_data[ new_data.columns[0] + " Spline"][startTimeSeries:].values, '-k')
plt.title("Spline portion with Smoothening = " + str( Smoothening_factor ) )
plt.xlabel('t (s)')
plt.ylabel('(deg. C)')
plt.grid(b=True)
plt.show()

'''

'''
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')






#new_data_2 =client.datapoints.retrieve_dataframe(external_id='VAL_23-TT-92539:X.Value', start=start, end=end, aggregates=aggregates, granularity=granularity)



    
    # Would need to find the # of points new data lies above array_stack
    


'''
#influxdb = InfluxDBClient(host='localhost', username='root', password='root')
#influxdb.get_list_database()

#ret_pol = influxdb.create_retention_policy('INF')




















