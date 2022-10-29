#! /usr/bin/env python

# python script to plot TESO transects

# Author: Johan van der Molen, modified by Athina Karaoli February 2022

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
import pandas as pd
import matplotlib.dates as mdates
import statsmodels.api as sm
import math
import matplotlib.colors as mcolors

############## Settings ##################################################3

# The names of the ADCPs
names=['hd','tx']

# Choose velocity 
#pl_vars=['EAST_VEL']  
#pl_vars=['NORTH_VEL']
pl_vars=['VERT_VEL']

# Choose date
yy='2022'; mm = '06'; dd = '01';  

# Crossings
instance_1 = 5; instance_2 = 6;  instance_step=1; 
 
# Error threshold
error_thresh = 0.35   # mask using error velocity 0.2

# Bad_fraction
bad_fraction = 0.3      # don't plot columns with a fraction of points with error velocity over error_threshold larger than bad_fraction


# Location of the data
if yy =='2022' and mm == '06' and dd == '01':
  indir = '/home/prdusr/data_out/TESO/daily'
else: 
  indir='/home/jvandermolen/data_out/TESO/daily'

mode = 1  #raw and cleaned-up pl_vars; 
if mode==1:
  nrows=2; ncols=2

#Max value for the colorbar
if pl_vars==['VERT_VEL']:
  max_ax=0.6
else: 
  max_ax=1.5

  
fs1=16; fs2=9 # Dimensions of figures
fs=12  # Fontsize
dpi_setting=300
save_figures=0
figtype='.jpg'
figdir='/home/akaraoli/python_scripts/figures/'
figlabel=['a','b','c','d','e','f','g','h','i','j']


############# Subroutines ###############################################

def plot_all_vars(indir,names,yy,mm,dd,pl_vars,instance,fignr,error_thresh):

  for pl_var in pl_vars:
    fignr=fignr+1
    if mode==1:
      plot_all_instruments_test1(indir,names,yy,mm,dd,pl_var,instance,fignr,error_thresh)  
    tight_layout()

    if save_figures==1 and mode==1:
      figfilename=pl_var+'_'+str(yy)+str(mm)+str(dd)+'_'+str(instance)+'_thresh_0'+str(int(error_thresh*100)).zfill(2)+figtype
      savefig(figdir+'/'+figfilename,dpi=300)
      close(fignr)

  return fignr
  
  
#-----------------------------------------------------------------------

def plot_all_instruments_test1(indir,name,yy,mm,dd,varname,instance,fignr,error_thresh):

  figure(fignr,(fs1,fs2),None,facecolor='w',edgecolor='k')

  # loop over instruments
  nsub=0
  for name in names:
    print(name)
    [lat,lat_units,lat_longname]=loadvar(indir,name,yy,mm,dd,'LATITUDE',instance)
    [dep,dep_units,dep_longname]=loadvar(indir,name,yy,mm,dd,'BINDEPTH',instance)
    [siv,siv_units,siv_longname]=loadvar(indir,name,yy,mm,dd,'ERROR_VEL',instance)
    [plv,plv_units,plv_longname]=loadvar(indir,name,yy,mm,dd,varname,instance)
    [qc,qc_units,qc_longname]=loadvar(indir,name,yy,mm,dd,'General_QC',instance)
    [t,t_units,t_longname]=loadvar(indir,name,yy,mm,dd,'TIME',instance)
    [d,d_units,d_longname]=loadvar(indir,name,yy,mm,dd,'DAY',instance)
    
    # Direction of the ferry
    if (lat[100]-lat[200]) > 0:
      ar='right'
    else:
      ar='left' 
    
    # Create depth bins of 0.5m
    sd=shape(dep)
    depstep=dep[0,1]-dep[0,0]
    depax=arange(dep[0,0],dep[0,0]+sd[1]*depstep,depstep)

    latmax=0
    while lat[latmax]>0:
      latmax=latmax+1
      if latmax>len(lat):
        break

    # Plot of total data (including bad data)
    subplot(nrows,ncols,nsub+1)
    below_bottom=where(abs(plv)>10.,1,0)    
    plvp=where(below_bottom==1,NaN,plv)
    plvT=1*plvp.T
    pcolormesh(lat[0:latmax],-depax,plvT[:,0:latmax],vmin=-max_ax,vmax=max_ax,cmap='coolwarm')
    gca().invert_xaxis()
    colorbar()
    if (nsub+1==3 or nsub+1==4):
      xlabel(lat_longname+' ['+lat_units+']',fontsize=fs)
    if (nsub+1==1 or nsub+1==3):
      ylabel(dep_longname+' ['+dep_units+']',fontsize=fs)
    title(name+' '+plv_longname,fontsize=fs)
    text(52.97,-29.5,'Den Helder')
    text(53.002,-29.5,'Texel')
    text(53.005,-5,figlabel[nsub],fontsize=14)
    
    
    subplot(nrows,ncols,nsub+2)
    below_bottom=where(abs(plv)>10.,1,0)    
    plvp=where(abs(siv)>error_thresh,-9999.,plv)            # don't plot points with large error velocity
    plvp=where(below_bottom==1,NaN,plvp)
    plv_cleaned=remove_bad_columns(plvp)                    # remove columns with a large number of errors
    plvT=1*plv_cleaned.T

    
    # Bad data is removed 
    pcolormesh(lat[0:latmax],-depax,plvT[:,0:latmax],vmin=-max_ax,vmax=max_ax,cmap='coolwarm')
    gca().invert_xaxis()
    cb = colorbar()
    cb.ax.tick_params(labelsize='large')
    if (nsub+2==3 or nsub+2==4):
      xlabel(lat_longname+' ['+lat_units+']',fontsize=fs)
    if (nsub+2==1 or nsub+2==3):
      ylabel(dep_longname+' ['+dep_units+']',fontsize=fs)
    title(name+' '+plv_longname,fontsize=fs)
    text(52.97,-29.5,'Den Helder')
    text(53.002,-29.5,'Texel')
    text(53.005,-5,figlabel[nsub+1],fontsize=14)
    #plt.yticks(fontsize=13)
    #plt.xticks(fontsize=12)
     
    if ar == 'right':
      arrow(52.985, -29.5, -0.005, 0.0,fc='k',ec='k',length_includes_head=True, head_width=0.5, head_length=0.001)
    elif ar == 'left':
      arrow(52.980, -29.5, 0.005, 0.0,fc='k',ec='k',length_includes_head=True, head_width=0.5, head_length=0.001)
    
    nsub=nsub+2

  tight_layout()


  return

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
  
#-----------------------------------------------------------------------

############ Main ######################################################

#################### Plot the profile of the velocity #################################

fignr = 0
for instance in range(instance_1,instance_2,instance_step):
  plot_all_vars(indir,names,yy,mm,dd,pl_vars,instance,fignr,error_thresh)
  fignr=fignr+1
  
show()


