#! /usr/bin/env python

# Author: Athina Karaoli, April 2021

# import the relevant packages
import sys
import subprocess
from netCDF4 import Dataset
from numpy import *
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import statsmodels.api as sm
import math
import pandas as pd
import matplotlib.colors as mcolors

############## Settings ##################################################3


# Location of data
indir_binned='/home/jvandermolen/data_out/TESO/binned'

# The names of the ADCPs
names=['hd','tx']

# Choose velocity 
pl_vars='EAST_VEL'  
#pl_vars='NORTH_VEL'

# Choose date
yy='2021'; mm = '07'; dd = '23';  

# Crossings
instance_1 = 0; instance_2 = 32;  instance_step=1; 
 
# Error threshold
error_thresh = 0.20   # mask using error velocity 0.2

# Bad_fraction
bad_fraction=0.3      # don't plot columns with a fraction of points with error velocity over error_threshold larger than bad_fraction

# Choose latitude
latitude = [52.9886]


dpi_setting=300
save_figures=0
figtype='.jpg'
figdir='/home/akaraoli/python_scripts/figures/'


# Read the data for the water elevation for specific location and time
data_ferryport_2020 = pd.read_csv("/home/akaraoli/python_scripts/data/2020_ferryport.csv",encoding= 'unicode_escape',header = 1)
data_ferryport_2021 = pd.read_csv("/home/akaraoli/python_scripts/data/2021_ferryport.csv",encoding= 'unicode_escape',header = 1)
data_lighthouse_2020 = pd.read_csv("/home/akaraoli/python_scripts/data/2020_lighthouse.csv",encoding= 'unicode_escape',header = 1)
data_lighthouse_2021 = pd.read_csv("/home/akaraoli/python_scripts/data/2021_lighthouse.csv",encoding= 'unicode_escape',header = 1)


############# Functions ###############################################
  
#--------------------------------------Ferryport----------------------------------------------
def ferry_data(data_ferryport_2020, data_ferryport_2021, CHOOSE_DAY_START, CHOOSE_DAY_END,yy):
  date_fer_2020 = []; date_fer_2021 = []
  elev_fer_2020 = []; elev_fer_2021 = []

  #2020
  if yy == '2020':

    for i in range(len(data_ferryport_2020)):
        string =  str(data_ferryport_2020.iloc[i,1])
        
        # Extract the date and the time
        date = string.split(';')[20] + ' ' + string.split(';')[21]
        date_fer_2020.append(date)
        
        # Extract the elevation
        elev = string.split(';')[23]
        elev_fer_2020.append(elev)
    
    date_fer_2020 = array(date_fer_2020 , dtype=object)
    elev_fer_2020 = array(elev_fer_2020 , dtype=object)
    
    start = where(date_fer_2020 == CHOOSE_DAY_START)[0][0]
    end = where(date_fer_2020 == CHOOSE_DAY_END)[0][0]
    
    date_ferry = []; tide_ferry = []
    for i in range(start,end +1):
        date_ferry.append(date_fer_2020[i])
        tide_ferry.append(int(elev_fer_2020[i]))   
        
    date_ferry = array(date_ferry , dtype=object)
    tide_ferry = array(tide_ferry , dtype=object)
    tide_ferry[where(tide_ferry == 999999999)] = nan
  
  #2021
  if yy == '2021':
  
    for i in range(len(data_ferryport_2021)):
        string =  str(data_ferryport_2021.iloc[i,0])
        
        # Extract the date and the time
        date = string.split(';')[21] + ' ' + string.split(';')[22]
        date_fer_2021.append(date)
        
        # Extract the elevation
        elev = string.split(';')[24]
        elev_fer_2021.append(elev)
        
    date_fer_2021 = array(date_fer_2021 , dtype=object)
    elev_fer_2021 = array(elev_fer_2021 , dtype=object)
      
    start = where(date_fer_2021 == CHOOSE_DAY_START)[0][0]
    end = where(date_fer_2021 == CHOOSE_DAY_END)[0][0]
    
    date_ferry = []; tide_ferry = []
    for i in range(start,end +1):
        date_ferry.append(date_fer_2021[i])
        tide_ferry.append(int(elev_fer_2021[i]))   
        
    date_ferry = array(date_ferry , dtype=object)
    tide_ferry = array(tide_ferry , dtype=object)
    tide_ferry[where(tide_ferry == 999999999)] = nan
  

  return date_ferry, tide_ferry
    
#----------------------------------------Lighthouse----------------------------------------
def lighthouse_data(data_ferryport_2020, data_ferryport_2021, CHOOSE_DAY_START, CHOOSE_DAY_END,yy):
  date_light_2020 = []; date_light_2021 = []
  elev_light_2020 = []; elev_light_2021 = []
  
  #2020
  if yy == '2020':
  
    for i in range(len(data_lighthouse_2020)):
        string =  str(data_lighthouse_2020.iloc[i,0])
        
        # Extract the date and the time
        date = string.split(';')[21] + ' ' + string.split(';')[22]
        date_light_2020.append(date)
        
        # Extract the elevation
        elev = string.split(';')[24]
        elev_light_2020.append(elev)
    
    date_light_2020 = array(date_light_2020 , dtype=object)
    elev_light_2020 = array(elev_light_2020 , dtype=object)
    
    start = where(date_light_2020 == CHOOSE_DAY_START)[0][0]
    end = where(date_light_2020 == CHOOSE_DAY_END)[0][0]
    
    date_lighthouse = []; tide_lighthouse = []
    for i in range(start,end +1):
        date_lighthouse.append(date_light_2020[i])
        tide_lighthouse.append(int(elev_light_2020[i]))   
        
    date_lighthouse = array(date_lighthouse , dtype=object)
    tide_lighthouse = array(tide_lighthouse , dtype=object)
    tide_lighthouse[where(tide_lighthouse == 999999999)] = nan
          
  #2021
  if yy == '2021':
      
    for i in range(len(data_lighthouse_2021)):
        string =  str(data_lighthouse_2021.iloc[i,0])
        
        # Extract the date and the time
        date = string.split(';')[21] + ' ' + string.split(';')[22]
        date_light_2021.append(date)
        
        # Extract the elevation
        elev = string.split(';')[24]
        elev_light_2021.append(elev)
        
    date_light_2021 = array(date_light_2021 , dtype=object)
    elev_light_2021 = array(elev_light_2021 , dtype=object)
  
    start = where(date_light_2021 == CHOOSE_DAY_START)[0][0]
    end = where(date_light_2021 == CHOOSE_DAY_END)[0][0]
    
    date_lighthouse = []; tide_lighthouse = []
    for i in range(start,end +1):
        date_lighthouse.append(date_light_2021[i])
        tide_lighthouse.append(int(elev_light_2021[i]))   
        
    date_lighthouse = array(date_lighthouse , dtype=object)
    tide_lighthouse = array(tide_lighthouse , dtype=object)
    tide_lighthouse[where(tide_lighthouse == 999999999)] = nan
      

  return date_lighthouse, tide_lighthouse
  
#-----------------------------------------------------------------------

def remove_bad_columns(data):

  sd=shape(data)
  out=1*data
  for col in range(sd[0]):
    column=data[col,:]
    nbadpoints=count_nonzero(column==-9999.)
    nwetpoints=(~isnan(column)).sum(0)
    if nbadpoints>nwetpoints*bad_fraction:
      column=NaN
      out[col,:]=column
    out[col,:]=where(out[col,:]==-9999.,NaN,out[col,:])

  return out

#-----------------------------------------------------------------------

def correct_depth(data,lat):

  meandepth_tx=9.0
  depth_ship=5.0
  index=argmax(lat,axis=0)
#  print('index',index)
  column=data[index,:]
#  print(column)
#  botindex=isnan(column[1:]).argmax()
  botindex=nonzero(column==1)[0][0] #where(column==1)
#  print('botindex',botindex)
  depth_below_ship=(botindex+1)*0.5
#  print('depth_below_ship',depth_below_ship)
  out=meandepth_tx-(depth_ship+depth_below_ship)
#  print('out',out)
#  sys.exit()

  return out

#-----------------------------------------------------------------------

def loadvar(indir,name,yy,mm,dd,varname,instance):

  # open file
  infname=indir+'/'+name+yy+mm+dd+'/'+name+yy+mm+dd+'.nc'
  infile=Dataset(infname,'r',format='NETCDF4') 
  
  # read variable
  var=infile.variables[varname]
  dum=var[:] 
  sd=shape(dum)
#  print('sd',sd)
#  print(len(sd))
  if len(sd)==2:
    out=dum[instance,:]
  else:
    out=dum[instance,:,:]

  if hasattr(var,'units'):
    units=var.units
  else:
    units=''
  longname=var.long_name

  # close file
  infile.close()

  return out,units,longname
  
  
#-----------------------------------------------------------------------

def loadvar_binned(indir,yy,mm,dd,varname,instance):

  # open file
  infname=indir+'/binned_'+yy+mm+dd+'.nc'
  infile=Dataset(infname,'r',format='NETCDF4') 
  
  # read variable
  var=infile.variables[varname]
  dum=var[:] 
  sd=shape(dum)
#  print('sd',sd)
#  print(len(sd))
  if len(sd)==2:
    out=dum[instance,:]
  else:
    out=dum[instance,:,:]

  if hasattr(var,'units'):
    units=var.units
  else:
    units=''
  longname=var.long_name

  # close file
  infile.close()

  return out,units,longname
  
# ------------------------------------------------------------------

def find_nearest(array, value):
    array = asarray(array)
    idx = (abs(array - value)).argmin()
    return array[idx]
  
# ----------------------------------------------------------------------
# Compute the mean value of the velocity in a specific latitide
def plot_all_instruments_test2(indir,yy,mm,dd,varname,instance,error_thresh, lat_num) :

  [lat,lat_units,lat_longname]=loadvar_binned(indir,yy,mm,dd,'LATITUDE',instance)
  [dep,dep_units,dep_longname]=loadvar_binned(indir,yy,mm,dd,'BINDEPTH',instance)
  [plv,plv_units,plv_longname]=loadvar_binned(indir,yy,mm,dd,varname,instance) 
  [t,t_units,t_longname]=loadvar_binned(indir,yy,mm,dd,'TIME',instance)
  [d,d_units,d_longname]=loadvar_binned(indir,yy,mm,dd,'DAY',instance)
   
  # Direction of the ferry
  if (lat[0]-lat[20]) > 0:
    ar='right'
  else:
    ar='left' 
    
  # Create depth bins of 0.5m
  sd=shape(dep)
  depstep=dep[0,1]-dep[0,0]
  depax=arange(dep[0,0],dep[0,0]+sd[1]*depstep,depstep)

  ####
  below_bottom=where(abs(plv)>10.,1,0)    
  plvp=where(below_bottom==1,NaN,plv)
  plvT=1*plvp.T
  
  # Choose specific latitude
  latmax=len(lat)
  index = where(lat[0:latmax] == find_nearest(lat[0:latmax], lat_num))[0][0]    
  # find_nearest function: find the measured latitude which is closer to the chosen latitude
  # Note that the ferry does not sail in a straight line and its speed changes
  # along its trip and as a result the two ADCPs do not take the velocity 
  # measurements at exactly the same latitudinal location but it differs by 0.25%.
  
  # Convert seconds into time in the form of yy-mm-dd HH:MM:SS
  tt = datetime(int(yy),int(mm),int(dd),0,0,0) + timedelta(seconds=int(t[index])+3600) # Time of the measuremnet at the chosen location 
  time = datetime(int(yy),int(mm),int(dd),0,0,0) + timedelta(seconds=int(t[0])+3600) # Time of the first measurement 

 
  # Put in an array the profile of the velocity at the chosen location 
  vel = zeros(len(depax))
  for i in range(len(depax)):
    vel[i] = plvT[i,index]
  
  # Remove the nan values
  count = 0; tot_vel =[]
  for i in range(len(depax)):
    bo= ~isnan(vel[i])
    if  bo == True: 
      tot_vel.append(vel[i])
      count= count+ 1
  
  # Compute the mean value 
  mean_val = mean(tot_vel)
 
  tight_layout()
  
  
  return mean_val, tt, time, ar
  
# ---------------------------------------------------------------------------------------------------------------------------
  
def timeseries(data_ferryport_2020, data_ferryport_2021, data_lighthouse_2020, data_lighthouse_2021, pl_vars, latitude, yy, mm, dd):
   
  # Empty lists 
  mean_current = []; time_current = []; time_tot=[]; direction_ship = []
  
  for lat_num in latitude:
    for instance in range(instance_1,instance_2,instance_step):
      print(instance)
      avg_current, t, time, ar = plot_all_instruments_test2(indir_binned,yy,mm,dd,pl_vars,instance,error_thresh, lat_num) 
      mean_current.append(avg_current*100) # cm/s
      time_current.append(t)
      time_tot.append(time)
      direction_ship.append(ar)
       
   
  # Convert time in the form of yy-mm-dd HH:MM:SS back to total seconds:
  time_tot_sec = []
  for k in range(len(time_current)):
    p = (time_current[k] - datetime(int(yy),int(mm),int(dd),0,0,0)).total_seconds()
    time_tot_sec.append(p)
  
  
  # Find the water elevation at that specific location at that specific time
  CHOOSE_DAY_START = dd+'-'+mm+'-'+yy+' 00:00:00'; CHOOSE_DAY_END = dd+'-'+mm+'-'+yy+' 23:50:00'
  X = array([datetime(int(yy), int(mm), int(dd), 5,0,0) + timedelta(minutes=i) for i in range(15*60)])
  date_ferry, tide_ferry = ferry_data(data_ferryport_2020, data_ferryport_2021, CHOOSE_DAY_START, CHOOSE_DAY_END,yy)
  date_lighthouse, tide_lighthouse = lighthouse_data(data_lighthouse_2020, data_lighthouse_2021, CHOOSE_DAY_START, CHOOSE_DAY_END,yy)

  ######### Plot #######
  fignr = 1
  figure(fignr,(32,13))
  subplot(1,1,1)
  x_values = [datetime.strptime(d,"%d-%m-%Y %H:%M:%S") for d in date_ferry]
  plot(x_values, tide_ferry, label = 'Water elevation', linewidth=3)
  xlabel('time [hours]',fontsize=13)

  if pl_vars=='NORTH_VEL':
    ylabel('$\eta$ [cm]; v [cm/s]',fontsize=1)
    plot(time_current, mean_current, 'r',label = 'Depth-averaged northward current velocity', linewidth=3)
  elif pl_vars=='EAST_VEL':
    ylabel('$\eta$ [cm]; u [cm/s]',fontsize=13)
    plot(time_current, mean_current, 'r',label = 'Depth-averaged eastward current velocity', linewidth=3)
      
 
  ax = plt.gca()
  ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
  ax.tick_params(axis='both', labelsize=11)
  grid()
  legend(loc=2, fontsize=8)
  #legend(loc='lower left', bbox_to_anchor= (0.0, 1.01), ncol=2, fontsize=7)  
  #plt.yticks(fontsize=11)
  #plt.xticks(fontsize=11)
  
  if save_figures==1:
    figfilename=pl_vars+'_'+str(yy)+str(mm)+str(dd)+'_tides_timeseries_interval100'.zfill(2)+figtype
    savefig(figdir+'/'+figfilename,dpi=300)
    close(fignr) 
  
  fignr = fignr + 1
   
  show()
  return
  
#-----------------------------------------------------------------------

############ Main ######################################################

# Plot the timesries of the water level and the current speed 
timeseries(data_ferryport_2020, data_ferryport_2021, data_lighthouse_2020, data_lighthouse_2021, pl_vars, latitude, yy, mm, dd)  
show()
