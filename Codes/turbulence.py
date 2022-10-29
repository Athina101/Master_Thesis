#! /usr/bin/env python

# python script to plot TESO transects

# Author: Athina Karaoli, April 2022

# import the relevant packages
import sys
import subprocess
from netCDF4 import Dataset
from numpy import *
from pylab import *
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as cm
from datetime import datetime, timedelta
from scipy import optimize
from IPython.display import display, Math
import difflib
from scipy.interpolate import make_interp_spline
import shapely
from shapely.geometry import LineString


######################## Functions that we need#####################################

# ---------------------------------------------------------------------- 
    
def parabola_func(x, a,c):
    return a*(x)**2 + c

# ----------------------------------------------------------------------
def find_nearest(array, value):
    array = asarray(array)
    idx = (abs(array - value)).argmin()
    return array[idx]
    
#-----------------------------------------------------------------------
def plot_distributions(indir,names,instance_1,instance_2,instance_step,bad_fraction,varname):

  # Choose date
  yy='2021'; mm = '07'; dd = '15'; instance = 28; error_thresh = 0.41
  
  fignr = 1
  figure(fignr,(8,9),None,facecolor='w',edgecolor='k')
  
  nsub=1
  # loop over instruments
  for name in names:
    [lat,lat_units,lat_longname]=loadvar(indir,name,yy,mm,dd,'LATITUDE',instance)
    [dep,dep_units,dep_longname]=loadvar(indir,name,yy,mm,dd,'BINDEPTH',instance)
    [siv,siv_units,siv_longname]=loadvar(indir,name,yy,mm,dd,'ERROR_VEL',instance)
    [plv,plv_units,plv_longname]=loadvar(indir,name,yy,mm,dd,varname,instance)
    [qc,qc_units,qc_longname]=loadvar(indir,name,yy,mm,dd,'General_QC',instance)
    [t,t_units,t_longname]=loadvar(indir,name,yy,mm,dd,'TIME',instance)

    # Create depth bins of 0.5m
    sd=shape(dep)
    depstep=dep[0,1]-dep[0,0]
    depax=arange(dep[0,0],dep[0,0]+sd[1]*depstep,depstep)
    
    # Direction of the ferry 
    if name=='hd':
      if (lat[100]-lat[200]) > 0:
        ar='right'
      else:
        ar='left'
        
    latmax=0
    while lat[latmax]>0:
      latmax=latmax+1
      if latmax>len(lat):
        break
   
    # Save the latitutde and the time data for hd and tx ADCP separately
    if name == 'hd':
      lat_hd =[]; t_hd = []
      for i in range(len(lat[0:latmax])):
        lat_hd.append(lat[i])
        t_hd.append(t[i])
    if name == 'tx':
      lat_tx = []; t_tx = []
      for i in range(len(lat[0:latmax])):
        lat_tx.append(lat[i]) 
        t_tx.append(t[i])
           
    below_bottom=where(abs(plv)>10.,1,0)    
    plvp=where(abs(siv)>error_thresh,-9999.,plv)            # don't plot points with large error velocity
    plvp=where(below_bottom==1,NaN,plvp)
    plv_cleaned=remove_bad_columns(plvp)                    # remove columns with a large number of errors
    plvT=1*plv_cleaned.T
    levs=arange(-1.3,1.4,0.05)
    
    # Save the velocity data for hd and tx ADCP separately
    if name == 'hd':
      vel_hd = zeros([len(depax),latmax])
      limit_hd = latmax # length of latitude data fro hd
      for i in range(len(depax)):
        for j in range(latmax):
          vel_hd[i,j] = plvT[i,j]
    if name == 'tx':
      vel_tx = zeros([len(depax),latmax])
      limit_tx = latmax # length of latitude data for tx
      for i in range(len(depax)):
        for j in range(latmax):
          vel_tx[i,j] = plvT[i,j]
          
  # The length of the data between the hd and tx ADCP differs. We find the common latitudes for hd and tx ADCP.
  lat_common = []; index_hd = []; index_tx = []
  
  if limit_hd > limit_tx:
    latmax2 = limit_tx
    for i in range(latmax2):
      lat_common.append(find_nearest(lat_hd, lat_tx[i]))
      index_hd.append(where(lat_hd == find_nearest(lat_hd, lat_tx[i]))[0][0])    
  else:
    latmax2 = limit_hd 
    for i in range(latmax2):
      lat_common.append(find_nearest(lat_tx, lat_hd[i]))
      index_tx.append(where(lat_tx == find_nearest(lat_tx, lat_hd[i]))[0][0])
  
   
  # For each latitude we find the time of the measurement and the velocity profile for the hd and tx ADCP. 
  time_hd = []; time_tx = []
  vel_hd2 = zeros([len(depax),len(lat_common)])
  vel_tx2 = zeros([len(depax),len(lat_common)])
  if limit_hd > limit_tx:
    for i in range(len(depax)):
      for j in range(len(lat_common)):
        vel_tx2[i,j] = vel_tx[i,j]
        vel_hd2[i,j] = vel_hd[i,index_hd[j]]
        if i == 0:
          time_hd.append(t_hd[index_hd[j]])
          time_tx.append(t_tx[j])
  else:
    for i in range(len(depax)):
      for j in range(len(lat_common)):
        vel_hd2[i,j] = vel_hd[i,j]
        vel_tx2[i,j] = vel_tx[i,index_tx[j]]
        if i == 0:
          time_hd.append(t_hd[j])
          time_tx.append(t_tx[index_tx[j]])

  #remove nan values
  tot_vel_hd =[]; tot_vel_tx =[]
  for i in range(len(depax)):
    for j in range(latmax2):
      v_hd = vel_hd2[i,j];  bo_hd = ~isnan(v_hd)
      v_tx = vel_tx2[i,j]; bo_tx = ~isnan(v_tx)
      if  bo_hd == True: 
        tot_vel_hd.append(v_hd)
      if  bo_tx == True: 
        tot_vel_tx.append(v_tx)
  
  #calculate the mean and the standard deviation for each ADCP      
  mean_val_hd = mean(tot_vel_hd); standev_hd = std(tot_vel_hd)
  mean_val_tx = mean(tot_vel_tx); standev_tx = std(tot_vel_tx)
  avg_mean = (mean_val_hd + mean_val_tx)/2
  
  
  #calculate the mean and the std deviation for the velocity difference of the two ADCPs
  tot_vel =[]
  for i in range(9,len(depax)):
    for j in range(latmax2):
      res = vel_hd2[i,j]-vel_tx2[i,j]
      bo = ~isnan(res)
      if  bo == True: 
        tot_vel.append(res)
        
  mean_val = mean(tot_vel); standev = std(tot_vel)


  #Plot distribution 
  if plv_longname == 'Eastward current speed':
    c = 'royalblue'
  elif plv_longname == 'Northward current speed':
     c = 'orange'
    
  hist(tot_vel, bins = 1000, range=(-1, 1),color=c)
  if plv_longname == 'Eastward current speed':
    xlabel(' u$_{hd}$ - u$_{tx}$ ['+plv_units+']',fontsize=16)
  elif plv_longname == 'Northward current speed':
    xlabel(' v$_{hd}$ - v$_{tx}$ ['+plv_units+']',fontsize=16)
  plt.yticks(fontsize=14)
  plt.xticks(fontsize=13)
   
  # Generate text to write.
  text1 = 'mean =  ' + str(round(mean_val,3))+ ' m/s'
  text2 = 'std =  ' + str(round(standev,3))+ ' m/s'
  text = text1 + '\n' + text2
  annotate(text, xy=(0.08, 1), xytext=(-15, -15), fontsize=15,xycoords='axes fraction', textcoords='offset points',bbox=dict(facecolor='white', alpha=0.8),horizontalalignment='left', verticalalignment='top')
  
  if savefigg == 1:
    figfilename=varname+'_'+str(yy)+str(mm)+str(dd)+'_'+str(instance)+'_hist_diff'.zfill(2)+figtype
    savefig(figdir+'/'+figfilename,dpi=300)
    close(fignr)
  

  tight_layout()

  return


#-----------------------------------------------------------------------

def plot_all_instruments_test3(indir,name,yy,mm,dd,varname,instance,error_thresh):


  # loop over instruments
  for name in names:
#    print(name)
    [lat,lat_units,lat_longname]=loadvar(indir,name,yy,mm,dd,'LATITUDE',instance)
    [dep,dep_units,dep_longname]=loadvar(indir,name,yy,mm,dd,'BINDEPTH',instance)
    [siv,siv_units,siv_longname]=loadvar(indir,name,yy,mm,dd,'ERROR_VEL',instance)
    [plv,plv_units,plv_longname]=loadvar(indir,name,yy,mm,dd,varname,instance)
    [qc,qc_units,qc_longname]=loadvar(indir,name,yy,mm,dd,'General_QC',instance)
    
    # Create depth bins of 0.5m
    sd=shape(dep)
    depstep=dep[0,1]-dep[0,0]
    depax=arange(dep[0,0],dep[0,0]+sd[1]*depstep,depstep)
      
    latmax=0
    while lat[latmax]>0:
      latmax=latmax+1
      if latmax>len(lat):
        break
        
    # Save the latitutde data for hd and tx ADCP separately    
    if name == 'hd':
      lat_hd =[]
      for i in range(len(lat[0:latmax])):
        lat_hd.append(lat[i])
    if name == 'tx':
      lat_tx = []
      for i in range(len(lat[0:latmax])):
        lat_tx.append(lat[i]) 
      
    below_bottom=where(abs(plv)>10.,1,0)    
    plvp=where(abs(siv)>error_thresh,-9999.,plv)            # don't plot points with large error velocity
    plvp=where(below_bottom==1,NaN,plvp)
    plv_cleaned=remove_bad_columns(plvp)                    # remove columns with a large number of errors
    plvT=1*plv_cleaned.T
    levs=arange(-1.3,1.4,0.05)
    
    # Save the velocity data for hd and tx ADCP separately
    if name == 'hd':
      vel_hd = zeros([len(depax),latmax])
      limit_hd = latmax
      for i in range(len(depax)):
        for j in range(latmax):
          vel_hd[i,j] = plvT[i,j]
    if name == 'tx':
      vel_tx = zeros([len(depax),latmax])
      limit_tx = latmax
      for i in range(len(depax)):
        for j in range(latmax):
          vel_tx[i,j] = plvT[i,j]
          
  # The length of the data between the hd and tx ADCP differs. We find the common latitudes for hd and tx ADCP.
  lat_common = []; index_hd = []; index_tx = []
  
  if limit_hd > limit_tx:
    latmax2 = limit_tx
    for i in range(latmax2):
      lat_common.append(find_nearest(lat_hd, lat_tx[i]))
      a = find_nearest(lat_hd, lat_tx[i])
      index_hd.append(where(lat_hd == a)[0][0])    
  else:
    latmax2 = limit_hd 
    for i in range(latmax2):
      lat_common.append(find_nearest(lat_tx, lat_hd[i]))
      a = find_nearest(lat_tx, lat_hd[i])
      index_tx.append(where(lat_tx == a)[0][0])
  
  # For each latitude we find the velocity profile for the hd and tx ADCP.   
  vel_hd2 = zeros([len(depax),len(lat_common)])
  vel_tx2 = zeros([len(depax),len(lat_common)])
  if limit_hd > limit_tx:
    for i in range(len(depax)):
      for j in range(len(lat_common)):
        vel_tx2[i,j] = vel_tx[i,j]
        vel_hd2[i,j] = vel_hd[i,index_hd[j]]
  else:
    for i in range(len(depax)):
      for j in range(len(lat_common)):
        vel_hd2[i,j] = vel_hd[i,j]
        vel_tx2[i,j] = vel_tx[i,index_tx[j]]
        
  # Exclude the data near the harbors
  depth_min = 0; depth_max = 30; lat_min = 52.967;  lat_max = 52.999
  for b in range(len(depax)):
    for c in range(len(lat_common)):
      if depax[b] < depth_min or depax[b] > depth_max:
        vel_hd2[b,:] = nan
        vel_tx2[b,:] = nan
        if lat_common[c] < lat_min or lat_common[c] > lat_max:
          vel_hd2[b,c] = nan
          vel_tx2[b,c] = nan
      if lat_common[c] < lat_min or lat_common[c] > lat_max:
        vel_hd2[b,c] = nan
        vel_tx2[b,c] = nan

  #remove nan values
  tot_vel_hd =[]; tot_vel_tx =[]
  for i in range(len(depax)):
    for j in range(latmax2):
      v_hd = vel_hd2[i,j]; bo_hd = ~isnan(v_hd)
      v_tx = vel_tx2[i,j]; bo_tx = ~isnan(v_tx)
      if  bo_hd == True: 
        tot_vel_hd.append(v_hd)
      if  bo_tx == True: 
        tot_vel_tx.append(v_tx)
        
  #calculate the mean and the std deviation for each ADCP
  mean_val_hd = mean(tot_vel_hd); standev_hd = std(tot_vel_hd)
  mean_val_tx = mean(tot_vel_tx); standev_tx = std(tot_vel_tx)
  avg_mean = (mean_val_hd + mean_val_tx)/2
  
  #calculate the mean and the std deviation for the velocity difference between the ADCPs
  tot_vel =[]
  for i in range(9,len(depax)): # excluding the first 10m to reduce the chance to observe artificial turbulence due to ferry’s motion
    for j in range(latmax2):
      res = vel_hd2[i,j]-vel_tx2[i,j]
      bo = ~isnan(res)
      if  bo == True: 
        tot_vel.append(res)
        
  mean_val = mean(tot_vel); standev = std(tot_vel)
  
  # Horizontally averaged velocity difference (u_hd - u_tx)
  dif_vel_depth_std = []; dif_vel_depth = []; depth = []
  for i in range(0,len(depax)):
    tot_vel =[]
    for j in range(latmax2):
      res = vel_hd2[i,j]-vel_tx2[i,j]; bo = ~isnan(res)
      if  bo == True:
        tot_vel.append(res)
    dif_vel_depth.append(mean(tot_vel))    # mean value of the horizontally averaged velocity difference for each depth layer
    dif_vel_depth_std.append(std(tot_vel)) # standard deviation value of the horizontally averaged velocity difference for each depth layer
    depth.append(-depax[i])
    
  # Horizontally averaged velocity (u_hd + u_tx)/2
  avg_mean_dep_square = []; avg_mean_dep =[]
  for i in range(0,len(depax)):
    tot_vel =[]
    for j in range(latmax2):
      res = (vel_hd2[i,j]+vel_tx2[i,j])/2; bo = ~isnan(res)
      if  bo == True: 
        tot_vel.append(res)
    avg_mean_dep_square.append(mean(tot_vel)**2) # mean squared value of the horizontally averaged velocity for each depth layer
    avg_mean_dep.append(mean(tot_vel))  # mean value of the horizontally averaged velocity for each depth layer

  return mean_val, standev, avg_mean, plv_units, dif_vel_depth_std, depth, dep_longname, avg_mean_dep_square, avg_mean_dep, dif_vel_depth
#-----------------------------------------------------------------------

def mean_values(yy,mm,dd,instance_1,instance_2,error_thresh,pl_var):
  
  means = []; mean_ADCPs =[]; stds =[]; depth_vel_std = []; domain_depth = []; depth_vel_error =[]; mean_ADCPs_per_depth_square = []; depth_vel_diff=[]
  
  depth_vel_std_ar = zeros([32,50]); mean_ADCPs_per_depth= zeros([32,50]); dif_vel_depth_ar = zeros([32,50]); tidal_plot = zeros([32,50])
  # 32 crossings, 50 length of depth array
  
  for instance in range(instance_1,instance_2,instance_step): # loop for 32 transects for one day
    mean_val, standev, avg_mean, plv_units, dif_vel_depth_std, depth, dep_longname, avg_mean_dep_square,avg_mean_dep, dif_vel_depth = plot_all_instruments_test3(indir,names,yy,mm,dd,pl_vars,instance,error_thresh) 
    means.append(mean_val) # mean velocity difference for each transect (u_hd - u_tx)
    mean_ADCPs.append(avg_mean) # mean velocity for each transect (u_hd + u_tx)/2
    stds.append(standev) # standard deviation value of the horizontally averaged velocity difference for each transect
    if instance == 0:
      domain_depth.extend(depth) # list for depth
    for j in range(len(domain_depth)):
      depth_vel_std_ar[instance,j] = dif_vel_depth_std[j]  # standard deviation value of the horizontally averaged velocity difference for each depth layer for each transect 
      dif_vel_depth_ar[instance,j] = dif_vel_depth[j]  # mean value of the horizontally averaged velocity difference for each depth layer for each transect 
      mean_ADCPs_per_depth[instance,j] = avg_mean_dep_square[j] # mean squared value of the horizontally averaged velocity for each depth layer for each transect
      tidal_plot[instance,j] = avg_mean_dep[j] # mean value of the horizontally averaged velocity for each depth layer for each transect
  

  # Compute the daily average values (mean value of the 32 crossings)
  for j in range(len(domain_depth)):
    list1 = []; list2 = []; list3 = []
    for transect in range(32):
      bo1 = ~isnan(depth_vel_std_ar[transect,j])
      bo2 = ~isnan(mean_ADCPs_per_depth[transect,j])
      bo3 = ~isnan(dif_vel_depth_ar[transect,j])
      if  bo1 == True: 
        list1.append(depth_vel_std_ar[transect,j])
      if  bo2 == True: 
        list2.append(mean_ADCPs_per_depth[transect,j]) 
      if  bo3 == True: 
        list3.append(dif_vel_depth_ar[transect,j])  
            
    depth_vel_std.append(mean(list1)) # daily mean value of the standard deviation value of the horizontally averaged velocity difference for each depth layer for each transect 
    depth_vel_error.append(std(list1)) # daily standard deviation value of the standard deviation value of the horizontally averaged velocity difference for each depth layer for each transect 
    mean_ADCPs_per_depth_square.append(sqrt(mean(list2))) # daily mean squared value of the horizontally averaged velocity for each depth layer for each transect
    depth_vel_diff.append(mean(list3)) # daily mean value of the horizontally averaged velocity difference for each depth layer for each transect 
    
  return means, mean_ADCPs, stds, plv_units, depth_vel_std, domain_depth, depth_vel_error, dep_longname, mean_ADCPs_per_depth_square, tidal_plot, depth_vel_diff

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
 
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def std_VS_tidal_current(instance_1,instance_2,instance_step,bad_fraction,pl_var):
  
  mean_ADCPs = []; stds = []
  # Year, month and error threshold
  yy='2021'; mm = '07';  error_thresh = 0.22
  # Days
  days = ['19','20','21','22','23','24']


  for dd in days:
    if dd =='19':
      instance_1 = 0; instance_2 = 31;
    else:
      instance_1 = 0; instance_2 = 32;
    means, mean_ADCP_sep, stds_dif, plv_units, depth_vel_std, domain_depth, depth_vel_error, dep_longname, mean_ADCPs_per_depth_square, tidal_plot, depth_vel_diff = mean_values(yy,mm,dd,instance_1,instance_2,error_thresh,pl_vars) 
    mean_ADCPs.extend(mean_ADCP_sep); stds.extend(stds_dif)

  # Distinguish the flood and ebb currents from the data
  mean_ADCPs_pos =[]; mean_ADCPs_neg =[]
  stds_pos=[]; stds_neg=[]
  for i in range(len(mean_ADCPs)):   
    if mean_ADCPs[i] > -0.05:
      mean_ADCPs_pos.append(mean_ADCPs[i]); mean_ADCPs_pos.append(-mean_ADCPs[i])
      stds_pos.append(stds[i]); stds_pos.append(stds[i])
    if mean_ADCPs[i] < 0.05:
      mean_ADCPs_neg.append(mean_ADCPs[i]); mean_ADCPs_neg.append(-mean_ADCPs[i])
      stds_neg.append(stds[i]); stds_neg.append(stds[i])
      
  
  # Parabola fitting for the flood current
  mean_ADCPs_pos, stds_pos = (list(t) for t in zip(*sorted(zip(mean_ADCPs_pos, stds_pos))))
  params_pos, params_cov_pos = optimize.curve_fit(parabola_func, array(mean_ADCPs_pos), array(stds_pos), p0=[0.5, 0.3])

  # Parabola fitting for the ebb current
  mean_ADCPs_neg, stds_neg = (list(t) for t in zip(*sorted(zip(mean_ADCPs_neg, stds_neg))))
  params_neg, params_cov_neg = optimize.curve_fit(parabola_func, array(mean_ADCPs_neg), array(stds_neg), p0=[0.5, 0.3])
  
  
  fignr=2  
  figure(fignr,(7,6),None,facecolor='w',edgecolor='k')
  
  scatter(mean_ADCPs,stds, color='k') 
  plot(linspace(0, max(mean_ADCPs_pos), 500), parabola_func(linspace(0, max(mean_ADCPs_pos), 500), params_pos[0], params_pos[1]), "r-", label='y = ax$^2$ + k' + '\na = {:1.3f}'.format(params_pos[0])+ ' $\pm$ {:1.3f}'.format(sqrt(params_cov_pos[0][0]))  + ', k = {:1.3f}'.format(params_pos[1])+ ' $\pm$ {:1.3f}'.format(sqrt(params_cov_pos[1][1])), linewidth=2)
  plot(linspace(min(mean_ADCPs_neg),0, 500) , parabola_func(linspace(min(mean_ADCPs_neg),0, 500), params_neg[0], params_neg[1]), "b-", label='a = {:1.3f}'.format(params_neg[0])+ ' $\pm$ {:1.3f}'.format(sqrt(params_cov_neg[0][0])) + ', k = {:1.3f}'.format(params_neg[1])+ ' $\pm$ {:1.3f}'.format(sqrt(params_cov_neg[1][1])), linewidth=2)
  
  if pl_vars=='EAST_VEL':
      ylabel('Std value of u$_{hd}$ - u$_{tx}$ ['+plv_units+']',fontsize=15)
      xlabel('Mean value of the eastward tidal current $\overline{u}$ ['+plv_units+']',fontsize=15)
  elif pl_vars=='NORTH_VEL':
      ylabel('Std value of v$_{hd}$ - v$_{tx}$ ['+plv_units+']',fontsize=15)
      xlabel('Mean value of the northward tidal current $\overline{v}$ ['+plv_units+']',fontsize=15)
  plt.yticks(fontsize=11)
  plt.xticks(fontsize=11)
  #ylim([0.18,0.30])
  #xlim([-1,1])
  legend()
  
  if savefigg == 1:
    figfilename=pl_var+'_parabolafitting'.zfill(2)+figtype
    savefig(figdir+'/'+figfilename,dpi=300)
    close(fignr)
  
  return mean_ADCPs, stds
  
# -----------------------------------------------------------------------------------

def std_VS_phase_tide(instance_1,instance_2,instance_step,bad_fraction,pl_var): 
  # We split the tidal cycle into four phases: the accelerating and the decelerating phase of the flood current and the accelerating and the decelerating phase of the ebb current
  # The data points of each phase is colored differently. 
  # Accelerating phase of the flood current: yellow; Decelerating phase of the flood current: green; Accelerating phase of the ebb current: red;  Decelerating phase of the ebb current: blue
   
  mean_ADCPs_x = []
  
  # Year, month and error threshold
  yy='2021'; mm = '07';  error_thresh = 0.22
  # Days
  days = ['01','02','03','05','06','07','08','09','10','12','13','14','15','17','19','20','21','22','23','24','26','27','28','29','30']
  
  # Save the values of the velocity of the tidal currents for the first day in the list
  means, mean_ADCP_sep, stds_dif, plv_units, depth_vel_std, domain_depth, depth_vel_error, dep_longname, mean_ADCPs_per_depth_square, tidal_plot, depth_vel_diff = mean_values(yy,mm,days[0],instance_1,instance_2,error_thresh,pl_vars) 
  mean_ADCPs_x.extend(mean_ADCP_sep)
  
  #---------------------------------------------- EASTWARD VELOCITY ---------------------------------------------------------#
  if pl_var == 'EAST_VEL':
    
    mean_ADCPs1 = []; stds1 = []
    mean_ADCPs2 = []; stds2 = []
    mean_ADCPs3 = []; stds3 = []
    mean_ADCPs4 = []; stds4 = []

    ### Plot of the std of the velocity difference as a function of tidal strength

    for dd in days:
      if dd =='19' or dd =='09' or dd =='29':
        instance_1 = 0; instance_2 = 31;
      else:
        instance_1 = 0; instance_2 = 32;
      mean_ADCPs = []; stds = []
      means, mean_ADCP_sep, stds_dif, plv_units, depth_vel_std, domain_depth, depth_vel_error, dep_longname, mean_ADCPs_per_depth_square, tidal_plot, depth_vel_diff = mean_values(yy,mm,dd,instance_1,instance_2,error_thresh,pl_vars) 
      mean_ADCPs.extend(mean_ADCP_sep); stds.extend(stds_dif)
      print(dd)

      # Distignuish the data into the four phases
      for i in range(len(mean_ADCPs)-1):
        if mean_ADCPs[i] > 0 and mean_ADCPs[i+1] > 0 and (mean_ADCPs[i+1]-mean_ADCPs[i]>0):
          mean_ADCPs1.append(mean_ADCPs[i]); stds1.append(stds[i])
          col1 = 'gold'
        if mean_ADCPs[i] > 0 and mean_ADCPs[i+1] > 0  and (mean_ADCPs[i+1]-mean_ADCPs[i]<0) and mean_ADCPs[i]!=max(mean_ADCPs):
          mean_ADCPs2.append(mean_ADCPs[i]); stds2.append(stds[i])
          col2 = 'limegreen'
        if mean_ADCPs[i] < 0 and mean_ADCPs[i+1] < 0  and (abs(mean_ADCPs[i+1])-abs(mean_ADCPs[i])>0):
          mean_ADCPs3.append(mean_ADCPs[i]); stds3.append(stds[i])
          col3 = 'salmon'
        if mean_ADCPs[i] < 0 and mean_ADCPs[i+1] < 0 and (abs(mean_ADCPs[i+1])-abs(mean_ADCPs[i])<0) and mean_ADCPs[i]!=min(mean_ADCPs):
          mean_ADCPs4.append(mean_ADCPs[i]); stds4.append(stds[i])
          col4 = 'lightskyblue'
        if  mean_ADCPs[i] < 0 and  mean_ADCPs[i+1] > 0:
          mean_ADCPs4.append(mean_ADCPs[i]); stds4.append(stds[i])
          col4 = 'lightskyblue'
        if  mean_ADCPs[i] > 0 and  mean_ADCPs[i+1] < 0:
          mean_ADCPs2.append(mean_ADCPs[i]); stds2.append(stds[i])
          col2 = 'limegreen'
        if mean_ADCPs[i]==max(mean_ADCPs):
          mean_ADCPs1.append(mean_ADCPs[i]); stds1.append(stds[i])
          col1 = 'gold'
        if mean_ADCPs[i]==min(mean_ADCPs):
          mean_ADCPs3.append(mean_ADCPs[i]); stds3.append(stds[i])
          col3 = 'salmon'
              
    mean_ADCPs11 =[]; mean_ADCPs22 =[]; mean_ADCPs33 =[]; mean_ADCPs44 =[]; stds11=[]; stds22=[];stds33=[]; stds44=[]; 
    for i in range(len(mean_ADCPs1)):   
      mean_ADCPs11.append(mean_ADCPs1[i]); mean_ADCPs11.append(-mean_ADCPs1[i])
      stds11.append(stds1[i]); stds11.append(stds1[i])
    for i in range(len(mean_ADCPs2)):   
      mean_ADCPs22.append(mean_ADCPs2[i]); mean_ADCPs22.append(-mean_ADCPs2[i])
      stds22.append(stds2[i]); stds22.append(stds2[i])
    for i in range(len(mean_ADCPs3)):   
      mean_ADCPs33.append(mean_ADCPs3[i]); mean_ADCPs33.append(-mean_ADCPs3[i])
      stds33.append(stds3[i]); stds33.append(stds3[i])
    for i in range(len(mean_ADCPs4)):   
      mean_ADCPs44.append(mean_ADCPs4[i]); mean_ADCPs44.append(-mean_ADCPs4[i])
      stds44.append(stds4[i]); stds44.append(stds4[i])  
        
    # Parabola fitting for each phase
    mean_ADCPs11, stds11 = (list(t) for t in zip(*sorted(zip(mean_ADCPs11, stds11))))
    params1, params_cov1 = optimize.curve_fit(parabola_func, array(mean_ADCPs11), array(stds11), p0=[0.8, 0.3])
  
    mean_ADCPs22, stds22 = (list(t) for t in zip(*sorted(zip(mean_ADCPs22, stds22))))
    params2, params_cov2 = optimize.curve_fit(parabola_func, array(mean_ADCPs22), array(stds22), p0=[0.5, 0.3])
    
    mean_ADCPs33, stds33 = (list(t) for t in zip(*sorted(zip(mean_ADCPs33, stds33))))
    params3, params_cov3 = optimize.curve_fit(parabola_func, array(mean_ADCPs33), array(stds33), p0=[0.5, 0.3])
    
    mean_ADCPs44, stds44 = (list(t) for t in zip(*sorted(zip(mean_ADCPs44, stds44))))
    params4, params_cov4 = optimize.curve_fit(parabola_func, array(mean_ADCPs44), array(stds44), p0=[0.5, 0.3])
    
    
    fignr=3  
    figure(fignr,(12,11),None,facecolor='w',edgecolor='k')
    
    col1 = 'orange' ; col2 = 'limegreen'; col3 = 'salmon';  col4 = 'lightskyblue'
    scatter(mean_ADCPs1,stds1, color=col1); scatter(mean_ADCPs2,stds2, color=col2); scatter(mean_ADCPs3,stds3, color=col3); scatter(mean_ADCPs4,stds4, color=col4)  
    
    plot(linspace(0, max(mean_ADCPs1), 500), parabola_func(linspace(0, max(mean_ADCPs1), 500), params1[0], params1[1]), 'darkorange', label='y = ax$^2$ + k' + '\na = {:1.3f}'.format(params1[0])+ ' $\pm$ {:1.3f}'.format(sqrt(params_cov1[0][0])) + ', k = {:1.3f}'.format(params1[1])+ ' $\pm$ {:1.3f}'.format(sqrt(params_cov1[1][1])), linewidth=2)
    
    plot(linspace(0, max(mean_ADCPs2), 500) , parabola_func(linspace(0, max(mean_ADCPs2), 500), params2[0], params2[1]), 'forestgreen', label='a = {:1.3f}'.format(params2[0])+ ' $\pm$ {:1.3f}'.format(sqrt(params_cov2[0][0]))+', k = {:1.3f}'.format(params2[1])+ ' $\pm$ {:1.3f}'.format(sqrt(params_cov2[1][1])), linewidth=2)
    
    plot(linspace(min(mean_ADCPs3),0, 500) , parabola_func(linspace(min(mean_ADCPs3),0, 500), params3[0], params3[1]), 'red', label='a = {:1.3f}'.format(params3[0])+ ' $\pm$ {:1.3f}'.format(sqrt(params_cov3[0][0])) + ', k = {:1.3f}'.format(params3[1])+ ' $\pm$ {:1.3f}'.format(sqrt(params_cov3[1][1])), linewidth=2)
    
    plot(linspace(min(mean_ADCPs4),0, 500) , parabola_func(linspace(min(mean_ADCPs4),0, 500), params4[0], params4[1]), 'b', label='a = {:1.3f}'.format(params4[0])+ ' $\pm$ {:1.3f}'.format(sqrt(params_cov4[0][0]))+ ', k = {:1.3f}'.format(params4[1])+ ' $\pm$ {:1.3f}'.format(sqrt(params_cov4[1][1])), linewidth=2)
    
    ylabel('Std value of u$_{hd}$ - u$_{tx}$ ['+plv_units+']',fontsize=21)
    xlabel('Mean value of the eastward tidal current $\overline{u}$ ['+plv_units+']',fontsize=21)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    legend(fontsize=16)
    
    
    if savefigg == 1:
      figfilename=pl_var+'_parabolafitting_phasetide'.zfill(2)+figtype
      savefig(figdir+'/'+figfilename,dpi=300)
      close(fignr)
    
    
    # ------------------------------------------------------------------------------------------------------------------
    # Plot of the timeseries of the tidal current
    
    fignr=4  
    figure(fignr,(9,8),None,facecolor='w',edgecolor='k')
    
    mean_ADCPs1 = []; x1 = []
    mean_ADCPs2 = []; x2 = []
    mean_ADCPs3 = []; x3 = []
    mean_ADCPs4 = []; x4 = []
    
    # Distignuish the data into the four phases
    for i in range(len(mean_ADCPs_x)-1):
      if mean_ADCPs_x[i] > 0 and mean_ADCPs_x[i+1] > 0 and (mean_ADCPs_x[i+1]-mean_ADCPs_x[i]>0):
        mean_ADCPs1.append(mean_ADCPs_x[i]); x1.append(i)
        col1 = 'gold'
      if mean_ADCPs_x[i] > 0 and mean_ADCPs_x[i+1] > 0 and (mean_ADCPs_x[i+1]-mean_ADCPs_x[i]<0) and mean_ADCPs_x[i]!=max(mean_ADCPs_x):
        mean_ADCPs2.append(mean_ADCPs_x[i]); x2.append(i)
        col2 = 'limegreen'
      if mean_ADCPs_x[i] < 0 and mean_ADCPs_x[i+1] < 0 and (abs(mean_ADCPs_x[i+1])-abs(mean_ADCPs_x[i])>0):
        mean_ADCPs3.append(mean_ADCPs_x[i]); x3.append(i)
        col3 = 'salmon'
      if mean_ADCPs_x[i] < 0 and mean_ADCPs_x[i+1] < 0 and (abs(mean_ADCPs_x[i+1])-abs(mean_ADCPs_x[i])<0) and mean_ADCPs_x[i]!=min(mean_ADCPs_x):
        mean_ADCPs4.append(mean_ADCPs_x[i]); x4.append(i)
        col4 = 'lightskyblue'
      if  mean_ADCPs_x[i] < 0 and  mean_ADCPs_x[i+1] > 0:
        mean_ADCPs4.append(mean_ADCPs_x[i]); x4.append(i)
        col4 = 'lightskyblue'
      if  mean_ADCPs_x[i] > 0 and  mean_ADCPs_x[i+1] < 0:
        mean_ADCPs2.append(mean_ADCPs_x[i]); x2.append(i)
        col2 = 'limegreen'
      if mean_ADCPs_x[i]==max(mean_ADCPs_x):
        mean_ADCPs1.append(mean_ADCPs_x[i]); x1.append(i)
        col1 = 'gold'
      if mean_ADCPs_x[i]==min(mean_ADCPs_x):
        mean_ADCPs3.append(mean_ADCPs_x[i]); x3.append(i)
        col3 = 'salmon'
    
    
    scatter(x1,mean_ADCPs1,linewidths = 5,color=col1, label = 'accelerating phase of the flood current')
    scatter(x2,mean_ADCPs2,linewidths = 5,color=col2, label = 'decelerating phase of the flood current')
    scatter(x3,mean_ADCPs3,linewidths = 5,color=col3, label = 'accelerating phase of the ebb current')
    scatter(x4,mean_ADCPs4,linewidths = 5,color=col4, label = 'decelerating phase of the ebb current')
    
    
    xlabel('Number of transect',fontsize=17)
    ylabel('Mean value of the eastward tidal current $\overline{u}$ ['+plv_units+']',fontsize=17)
    #title('Eastward direction')
    #legend(loc=1,fontsize=10)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    #lgd1 = legend(loc='upper center', bbox_to_anchor= (0.5, 1.1),fontsize=10)
    
    if savefigg == 1:
      figfilename=pl_var+'_parabolafitting_phasetide_timeseries'.zfill(2)+figtype
      savefig(figdir+'/'+figfilename,dpi=300)
      close(fignr)
    
    
    
  #---------------------------------------------- NORTHWARD VELOCITY ---------------------------------------------------------#
  if pl_var == 'NORTH_VEL':
  
    mean_ADCPs1 = []; stds1 = []
    mean_ADCPs2 = []; stds2 = []
    mean_ADCPs3 = []; stds3 = []
    mean_ADCPs4 = []; stds4 = []
     
    ### Plot of the std of the velocity difference as a function of tidal strength
    
    for dd in days:
      if dd =='19' or dd =='09' or dd =='29':
        instance_1 = 0; instance_2 = 31;
      else:
        instance_1 = 0; instance_2 = 32;
      mean_ADCPs = []
      stds = []
      means, mean_ADCP_sep, stds_dif, plv_units, depth_vel_std, domain_depth, depth_vel_error, dep_longname, mean_ADCPs_per_depth_square, tidal_plot, depth_vel_diff = mean_values(yy,mm,dd,instance_1,instance_2,error_thresh,pl_vars) 
      mean_ADCPs.extend(mean_ADCP_sep); stds.extend(stds_dif)
      print(dd)
    
     # Distignuish the data into the four phases
      for i in range(len(mean_ADCPs)-2):
      
        if mean_ADCPs[0] < 0:
        
          if mean_ADCPs[i] > 0  and i < where(mean_ADCPs == max(mean_ADCPs))[0][0] and mean_ADCPs[i] < mean_ADCPs[i+1]:
            mean_ADCPs1.append(mean_ADCPs[i]); stds1.append(stds[i])
            col1 = 'gold'
          if mean_ADCPs[i] > 0  and i < where(mean_ADCPs == max(mean_ADCPs))[0][0] and mean_ADCPs[i] > mean_ADCPs[i+1]:
            if mean_ADCPs[i+2] > mean_ADCPs[i+1]:
              mean_ADCPs1.append(mean_ADCPs[i]); stds1.append(stds[i])
              col1 = 'gold'  
          if mean_ADCPs[i] < 0 and mean_ADCPs[i+1] > 0 and  (abs(mean_ADCPs[i]) < mean_ADCPs[i+1]):
            mean_ADCPs1.append(mean_ADCPs[i]); stds1.append(stds[i])
            col1 = 'gold'
          if mean_ADCPs[i] == max(mean_ADCPs):
            mean_ADCPs1.append(mean_ADCPs[i]); stds1.append(stds[i])
            col1 = 'gold'
            
          if mean_ADCPs[i] > 0 and i > where(mean_ADCPs == max(mean_ADCPs))[0][0]  and mean_ADCPs[i] > mean_ADCPs[i+1] and mean_ADCPs[i] != max(mean_ADCPs) and i < where(mean_ADCPs == min(mean_ADCPs[13:32]))[0][0]:
            mean_ADCPs2.append(mean_ADCPs[i]); stds2.append(stds[i])
            col2 = 'limegreen'
          if mean_ADCPs[i] > 0  and i > where(mean_ADCPs == max(mean_ADCPs))[0][0] and mean_ADCPs[i] < mean_ADCPs[i+1] and mean_ADCPs[i] != max(mean_ADCPs) and i < where(mean_ADCPs == min(mean_ADCPs[13:32]))[0][0]:
            if mean_ADCPs[i+2] < mean_ADCPs[i+1]:
              mean_ADCPs2.append(mean_ADCPs[i]); stds2.append(stds[i])
              col2 = 'limegreen' 
          if mean_ADCPs[i] > 0 and mean_ADCPs[i+1] < 0 and  mean_ADCPs[i] > mean_ADCPs[i+1]:
            mean_ADCPs2.append(mean_ADCPs[i]); stds2.append(stds[i])
            col2 = 'limegreen'
            
          if mean_ADCPs[i] < 0  and i < where(mean_ADCPs == min(mean_ADCPs[0:13]))[0][0] and mean_ADCPs[i] > mean_ADCPs[i+1]:
            mean_ADCPs3.append(mean_ADCPs[i]); stds3.append(stds[i])
            col3 = 'salmon'
          if mean_ADCPs[i] < 0  and i < where(mean_ADCPs == min(mean_ADCPs[0:13]))[0][0] and mean_ADCPs[i] < mean_ADCPs[i+1]:
            if mean_ADCPs[i+2] < mean_ADCPs[i+1]:
              mean_ADCPs3.append(mean_ADCPs[i]); stds3.append(stds[i])
              col3 = 'salmon' 
          if mean_ADCPs[i] == min(mean_ADCPs[0:15]):
            mean_ADCPs3.append(mean_ADCPs[i]); stds3.append(stds[i])
            col3 = 'salmon'
            
          if mean_ADCPs[i] < 0  and i > where(mean_ADCPs == min(mean_ADCPs[0:13]))[0][0] and i < 13 and mean_ADCPs[i] < mean_ADCPs[i+1]:
            mean_ADCPs4.append(mean_ADCPs[i]); stds4.append(stds[i])
            col4 = 'lightskyblue'
          if mean_ADCPs[i] < 0  and i > where(mean_ADCPs== min(mean_ADCPs[0:13]))[0][0] and i < 13 and mean_ADCPs[i] > mean_ADCPs[i+1]:
            if mean_ADCPs[i+2] > mean_ADCPs[i+1]:
              mean_ADCPs4.append(mean_ADCPs[i]); stds4.append(stds[i])
              col4 = 'lightskyblue' 
    
          if mean_ADCPs[i] < 0  and i < where(mean_ADCPs == min(mean_ADCPs[13:32]))[0][0] and i > 13 and mean_ADCPs[i] > mean_ADCPs[i+1]:
            mean_ADCPs3.append(mean_ADCPs[i]); stds3.append(stds[i])
            col3 = 'salmon'
          if mean_ADCPs[i] < 0  and i < where(mean_ADCPs == min(mean_ADCPs[13:32]))[0][0] and i > 13 and mean_ADCPs[i] < mean_ADCPs[i+1]:
            mean_ADCPs3.append(mean_ADCPs[i]); stds3.append(stds[i])
            col3 = 'salmon' 
          if mean_ADCPs[i] == min(mean_ADCPs[15:32]):
            mean_ADCPs3.append(mean_ADCPs[i]); stds3.append(stds[i])
            col3 = 'salmon'
            
          if mean_ADCPs[i] < 0  and i > where(mean_ADCPs == min(mean_ADCPs[13:32]))[0][0] and i > 13 and mean_ADCPs[i] > mean_ADCPs[i+1]:
            mean_ADCPs4.append(mean_ADCPs[i]); stds4.append(stds[i])
            col4 = 'lightskyblue' 
          if mean_ADCPs[i] < 0  and i > where(mean_ADCPs == min(mean_ADCPs[13:32]))[0][0] and i > 13 and mean_ADCPs[i] < mean_ADCPs[i+1]:
            mean_ADCPs4.append(mean_ADCPs[i]); stds4.append(stds[i])
            col4 = 'lightskyblue'  
            
          if mean_ADCPs[i] > 0  and i > where(mean_ADCPs == min(mean_ADCPs[13:32]))[0][0] and i > 13 and mean_ADCPs[i] < mean_ADCPs[i+1]:
            mean_ADCPs1.append(mean_ADCPs[i]); stds1.append(stds[i])
            col1 = 'gold' 
          if mean_ADCPs[i] > 0  and i > where(mean_ADCPs == min(mean_ADCPs[13:32]))[0][0] and i > 13 and mean_ADCPs[i] > mean_ADCPs[i+1]:
            if mean_ADCPs[i+2] > mean_ADCPs[i+1]:
              mean_ADCPs1.append(mean_ADCPs[i]); stds1.append(stds[i])
              col1 = 'gold'  
         
        if mean_ADCPs[0] > 0:
          
          if mean_ADCPs[i] < 0  and i < where(mean_ADCPs == min(mean_ADCPs))[0][0] and mean_ADCPs[i] > mean_ADCPs[i+1]:
            mean_ADCPs3.append(mean_ADCPs[i]); stds3.append(stds[i])
            col3 = 'salmon'
          if mean_ADCPs[i] < 0  and i < where(mean_ADCPs == min(mean_ADCPs))[0][0] and mean_ADCPs[i] < mean_ADCPs[i+1]:
            if mean_ADCPs[i+2] < mean_ADCPs[i+1]:
              mean_ADCPs3.append(mean_ADCPs[i]); stds3.append(stds[i])
              col3 = 'salmon'  
          if mean_ADCPs[i] > 0 and mean_ADCPs[i+1] < 0 and  (mean_ADCPs[i]) > mean_ADCPs[i+1]:
            mean_ADCPs3.append(mean_ADCPs[i]); stds3.append(stds[i])
            col3 = 'salmon'
          if mean_ADCPs[i] == min(mean_ADCPs):
            mean_ADCPs3.append(mean_ADCPs[i]); stds3.append(stds[i])
            col3 = 'salmon'
            
          if mean_ADCPs[i] < 0 and i > where(mean_ADCPs == min(mean_ADCPs))[0][0]  and mean_ADCPs[i] < mean_ADCPs[i+1] and mean_ADCPs[i] != min(mean_ADCPs):
            mean_ADCPs4.append(mean_ADCPs[i]); stds4.append(stds[i])
            col4 = 'lightskyblue'
          if mean_ADCPs[i] < 0  and i > where(mean_ADCPs == min(mean_ADCPs))[0][0] and mean_ADCPs[i] > mean_ADCPs[i+1] and mean_ADCPs[i] != min(mean_ADCPs):
            if mean_ADCPs[i+2] > mean_ADCPs[i+1]:
              mean_ADCPs4.append(mean_ADCPs[i]); stds4.append(stds[i])
              col4 = 'lightskyblue' 
            
          if mean_ADCPs[i] > 0  and i < where(mean_ADCPs == max(mean_ADCPs[0:13]))[0][0] and mean_ADCPs[i] < mean_ADCPs[i+1]:
            mean_ADCPs1.append(mean_ADCPs[i]); stds1.append(stds[i])
            col1 = 'gold'
          if mean_ADCPs[i] > 0  and i < where(mean_ADCPs == max(mean_ADCPs[0:13]))[0][0] and mean_ADCPs[i] > mean_ADCPs[i+1]:
            if mean_ADCPs[i+2] > mean_ADCPs[i+1]:
              mean_ADCPs1.append(mean_ADCPs[i]); stds1.append(stds[i])
              col1 = 'gold' 
          if mean_ADCPs[i] == max(mean_ADCPs[0:15]):
            mean_ADCPs1.append(mean_ADCPs[i]); stds1.append(stds[i])
            col1 = 'gold'
            
          if mean_ADCPs[i] > 0  and i > where(mean_ADCPs == max(mean_ADCPs[0:13]))[0][0] and i < 13 and mean_ADCPs[i] > mean_ADCPs[i+1]:
            mean_ADCPs2.append(mean_ADCPs[i]); stds2.append(stds[i])
            col2 = 'limegreen' 
          if mean_ADCPs[i] > 0  and i > where(mean_ADCPs == max(mean_ADCPs[0:13]))[0][0] and i < 13 and mean_ADCPs[i] < mean_ADCPs[i+1]:
            if mean_ADCPs[i+2] < mean_ADCPs[i+1]:
              mean_ADCPs2.append(mean_ADCPs[i]); stds2.append(stds[i])
              col2 = 'limegreen' 
              
          if mean_ADCPs[i] > 0  and i < where(mean_ADCPs == max(mean_ADCPs[13:32]))[0][0] and i > 13 and mean_ADCPs[i] < mean_ADCPs[i+1]:
            mean_ADCPs1.append(mean_ADCPs[i]); stds1.append(stds[i])
            col1 = 'gold'
          if mean_ADCPs[i] > 0  and i < where(mean_ADCPs == max(mean_ADCPs[13:32]))[0][0] and i > 13 and mean_ADCPs[i] > mean_ADCPs[i+1]:
            mean_ADCPs1.append(mean_ADCPs[i]); stds1.append(stds[i])
            col1 = 'gold' 
          if mean_ADCPs[i] == max(mean_ADCPs[15:32]):
            mean_ADCPs1.append(mean_ADCPs[i]); stds1.append(stds[i])
            col1 = 'gold'
           
          if mean_ADCPs[i] > 0  and i > where(mean_ADCPs == max(mean_ADCPs[13:32]))[0][0] and i > 13 and mean_ADCPs[i] > mean_ADCPs[i+1]:
            mean_ADCPs2.append(mean_ADCPs[i]); stds2.append(stds[i])
            col2 = 'limegreen' 
          if mean_ADCPs[i] > 0  and i > where(mean_ADCPs == max(mean_ADCPs[13:32]))[0][0] and i > 13 and mean_ADCPs[i] < mean_ADCPs[i+1]:
            mean_ADCPs2.append(mean_ADCPs[i]); stds2.append(stds[i])
            col2 = 'limegreen' 
            
   
      if mean_ADCPs[0] < 0: # for the last two points of the timeseries (they were not included in the previous loops
        
        if mean_ADCPs[29] > 0  and 29 >= where(mean_ADCPs == min(mean_ADCPs[13:32]))[0][0]:
          mean_ADCPs1.append(mean_ADCPs[29]); mean_ADCPs1.append(mean_ADCPs[30]); stds1.append(stds[29]); stds1.append(stds[30])
          col1 = 'gold'
             
        if mean_ADCPs[29] < 0  and 29 <= where(mean_ADCPs == min(mean_ADCPs[13:32]))[0][0]:
          mean_ADCPs3.append(mean_ADCPs[29]); mean_ADCPs3.append(mean_ADCPs[30]); stds3.append(stds[29]); stds3.append(stds[30])
          col3 = 'salmon' 
          
        if mean_ADCPs[29] < 0  and 29 >= where(mean_ADCPs == min(mean_ADCPs[13:32]))[0][0]:
          mean_ADCPs4.append(mean_ADCPs[29]); mean_ADCPs4.append(mean_ADCPs[30]); stds4.append(stds[29]); stds4.append(stds[30])
          col4 = 'lightskyblue' 
          
      if mean_ADCPs[0] > 0: # for the last two points of the timeseries (they were not included in the previous loops
      
        if mean_ADCPs[29] < 0  and 29 >= where(mean_ADCPs == max(mean_ADCPs[13:32]))[0][0]:
          mean_ADCPs3.append(mean_ADCPs[29]); mean_ADCPs3.append(mean_ADCPs[30]); stds3.append(stds[29]); stds3.append(stds[30])
          col3 = 'salmon'
             
        if mean_ADCPs[29] > 0  and 29 <= where(mean_ADCPs == max(mean_ADCPs[13:32]))[0][0]:
          mean_ADCPs1.append(mean_ADCPs[29]); mean_ADCPs1.append(mean_ADCPs[30])
          stds1.append(stds[29]); stds1.append(stds[30])
          col1 = 'gold' 
          
        if mean_ADCPs[29] > 0  and 29 >= where(mean_ADCPs == max(mean_ADCPs[13:32]))[0][0]:
          mean_ADCPs2.append(mean_ADCPs[29]); mean_ADCPs2.append(mean_ADCPs[30])
          stds2.append(stds[29]); stds2.append(stds[30])
          col2 = 'limegreen' 
     
   
    mean_ADCPs11 =[]; mean_ADCPs22 =[]; mean_ADCPs33 =[]; mean_ADCPs44 =[]; stds11=[]; stds22=[];stds33=[]; stds44=[]; 
    for i in range(len(mean_ADCPs1)):   
      mean_ADCPs11.append(mean_ADCPs1[i]); mean_ADCPs11.append(-mean_ADCPs1[i])
      stds11.append(stds1[i]); stds11.append(stds1[i])
    for i in range(len(mean_ADCPs2)):   
      mean_ADCPs22.append(mean_ADCPs2[i]); mean_ADCPs22.append(-mean_ADCPs2[i])
      stds22.append(stds2[i]); stds22.append(stds2[i])
    for i in range(len(mean_ADCPs3)):   
      mean_ADCPs33.append(mean_ADCPs3[i]); mean_ADCPs33.append(-mean_ADCPs3[i])
      stds33.append(stds3[i]); stds33.append(stds3[i])
    for i in range(len(mean_ADCPs4)):   
      mean_ADCPs44.append(mean_ADCPs4[i]); mean_ADCPs44.append(-mean_ADCPs4[i])
      stds44.append(stds4[i]); stds44.append(stds4[i])  
        
    # Parabola fitting for each phase 
    mean_ADCPs11, stds11 = (list(t) for t in zip(*sorted(zip(mean_ADCPs11, stds11))))
    params1, params_cov1 = optimize.curve_fit(parabola_func, array(mean_ADCPs11), array(stds11), p0=[0.5, 0.3])
  
    mean_ADCPs22, stds22 = (list(t) for t in zip(*sorted(zip(mean_ADCPs22, stds22))))
    params2, params_cov2 = optimize.curve_fit(parabola_func, array(mean_ADCPs22), array(stds22), p0=[0.5, 0.3])
    
    mean_ADCPs33, stds33 = (list(t) for t in zip(*sorted(zip(mean_ADCPs33, stds33))))
    params3, params_cov3 = optimize.curve_fit(parabola_func, array(mean_ADCPs33), array(stds33), p0=[0.5, 0.3])
    
    mean_ADCPs44, stds44 = (list(t) for t in zip(*sorted(zip(mean_ADCPs44, stds44))))
    params4, params_cov4 = optimize.curve_fit(parabola_func, array(mean_ADCPs44), array(stds44), p0=[0.5, 0.3])
    

    fignr=5  
    figure(fignr,(12,11),None,facecolor='w',edgecolor='k')
  
    col1 = 'orange' ; col2 = 'limegreen';col3 = 'salmon';  col4 = 'lightskyblue'
    scatter(mean_ADCPs1,stds1, color=col1); scatter(mean_ADCPs2,stds2, color=col2); scatter(mean_ADCPs3,stds3, color=col3); scatter(mean_ADCPs4,stds4, color=col4) 
    
    plot(linspace(0, max(mean_ADCPs1), 500), parabola_func(linspace(0, max(mean_ADCPs1), 500), params1[0], params1[1]), 'darkorange', label='y = ax$^2$ + k' + '\na = {:1.3f}'.format(params1[0])+ ' $\pm$ {:1.3f}'.format(sqrt(params_cov1[0][0])) + ', k = {:1.3f}'.format(params1[1])+ ' $\pm$ {:1.3f}'.format(sqrt(params_cov1[1][1])), linewidth=2)
    
    plot(linspace(0, max(mean_ADCPs2), 500) , parabola_func(linspace(0, max(mean_ADCPs2), 500), params2[0], params2[1]), 'forestgreen', label='a = {:1.3f}'.format(params2[0])+ ' $\pm$ {:1.3f}'.format(sqrt(params_cov2[0][0]))  + ', k = {:1.3f}'.format(params2[1])+ ' $\pm$ {:1.3f}'.format(sqrt(params_cov2[1][1])), linewidth=2)
    
    plot(linspace(min(mean_ADCPs3),0, 500) , parabola_func(linspace(min(mean_ADCPs3),0, 500), params3[0], params3[1]), 'red', label='a = {:1.3f}'.format(params3[0])+ ' $\pm$ {:1.3f}'.format(sqrt(params_cov3[0][0]))  + ', k = {:1.3f}'.format(params3[1])+ ' $\pm$ {:1.3f}'.format(sqrt(params_cov3[1][1])), linewidth=2)
    
    plot(linspace(min(mean_ADCPs4),0, 500) , parabola_func(linspace(min(mean_ADCPs4),0, 500), params4[0], params4[1]), 'b', label='a = {:1.3f}'.format(params4[0])+ ' $\pm$ {:1.3f}'.format(sqrt(params_cov4[0][0]))  + ', k = {:1.3f}'.format(params4[1])+ ' $\pm$ {:1.3f}'.format(sqrt(params_cov4[1][1])), linewidth=2)
    
    
    ylabel('Std value of v$_{hd}$ - v$_{tx}$ ['+plv_units+']',fontsize=21)
    xlabel('Mean value of the northward tidal current $\overline{v}$ ['+plv_units+']',fontsize=21)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    legend(fontsize=16)
    
    if savefigg == 1:
      figfilename=pl_var+'_parabolafitting_phasetide'.zfill(2)+figtype
      savefig(figdir+'/'+figfilename,dpi=300)
      close(fignr)
    
    
    # ------------------------------------------------------------------
    # Plot of the timeseries of the tidal current
    
    fignr=6  
    figure(fignr,(9,8),None,facecolor='w',edgecolor='k')
    
    mean_ADCPs1 = []; x1 = []
    mean_ADCPs2 = []; x2 = []
    mean_ADCPs3 = []; x3 = []
    mean_ADCPs4 = []; x4 = []
       
    
    for i in range(len(mean_ADCPs_x)-2):
      
      if mean_ADCPs_x[0] < 0:
        if mean_ADCPs_x[i] > 0  and i < where(mean_ADCPs_x == max(mean_ADCPs_x))[0][0] and mean_ADCPs_x[i] < mean_ADCPs_x[i+1]:
          mean_ADCPs1.append(mean_ADCPs_x[i]); x1.append(i)
          col1 = 'gold'
        if mean_ADCPs_x[i] > 0  and i < where(mean_ADCPs_x == max(mean_ADCPs_x))[0][0] and mean_ADCPs_x[i] > mean_ADCPs_x[i+1]:
          if mean_ADCPs_x[i+2] > mean_ADCPs_x[i+1]:
            mean_ADCPs1.append(mean_ADCPs_x[i]); x1.append(i)
            col1 = 'gold'  
        if mean_ADCPs_x[i] < 0 and mean_ADCPs_x[i+1] > 0 and  (abs(mean_ADCPs_x[i]) < mean_ADCPs_x[i+1]):
          mean_ADCPs1.append(mean_ADCPs_x[i]); x1.append(i)
          col1 = 'gold'
        if mean_ADCPs_x[i] == max(mean_ADCPs_x):
          mean_ADCPs1.append(mean_ADCPs_x[i]); x1.append(i)
          col1 = 'gold'
          
        if mean_ADCPs_x[i] > 0 and i > where(mean_ADCPs_x == max(mean_ADCPs_x))[0][0]  and mean_ADCPs_x[i] > mean_ADCPs_x[i+1] and mean_ADCPs_x[i] != max(mean_ADCPs_x) and i < where(mean_ADCPs_x == min(mean_ADCPs_x[13:32]))[0][0]:
          mean_ADCPs2.append(mean_ADCPs_x[i]); x2.append(i)
          col2 = 'limegreen'
        if mean_ADCPs_x[i] > 0  and i > where(mean_ADCPs_x == max(mean_ADCPs_x))[0][0] and mean_ADCPs_x[i] < mean_ADCPs_x[i+1] and mean_ADCPs_x[i] != max(mean_ADCPs_x) and i < where(mean_ADCPs_x == min(mean_ADCPs_x[13:32]))[0][0]:
          if mean_ADCPs_x[i+2] < mean_ADCPs_x[i+1]:
            mean_ADCPs2.append(mean_ADCPs_x[i]); x2.append(i)
            col2 = 'limegreen' 
        if mean_ADCPs_x[i] > 0 and mean_ADCPs_x[i+1] < 0 and  mean_ADCPs_x[i] > mean_ADCPs_x[i+1]:
          mean_ADCPs2.append(mean_ADCPs_x[i]); x2.append(i)
          col2 = 'limegreen'
          
        if mean_ADCPs_x[i] < 0  and i < where(mean_ADCPs_x == min(mean_ADCPs_x[0:13]))[0][0] and mean_ADCPs_x[i] > mean_ADCPs_x[i+1]:
          mean_ADCPs3.append(mean_ADCPs_x[i]); x3.append(i)
          col3 = 'salmon'
        if mean_ADCPs_x[i] < 0  and i < where(mean_ADCPs_x == min(mean_ADCPs_x[0:13]))[0][0] and mean_ADCPs_x[i] < mean_ADCPs_x[i+1]:
          if mean_ADCPs_x[i+2] < mean_ADCPs_x[i+1]:
            mean_ADCPs3.append(mean_ADCPs_x[i]); x3.append(i)
            col3 = 'salmon' 
        if mean_ADCPs_x[i] == min(mean_ADCPs_x[0:15]):
          mean_ADCPs3.append(mean_ADCPs_x[i]); x3.append(i)
          col3 = 'salmon'
          
        if mean_ADCPs_x[i] < 0  and i > where(mean_ADCPs_x == min(mean_ADCPs_x[0:13]))[0][0] and i < 13 and mean_ADCPs_x[i] < mean_ADCPs_x[i+1]:
          mean_ADCPs4.append(mean_ADCPs_x[i]); x4.append(i)
          col4 = 'lightskyblue'
        if mean_ADCPs_x[i] < 0  and i > where(mean_ADCPs_x == min(mean_ADCPs_x[0:13]))[0][0] and i < 13 and mean_ADCPs_x[i] > mean_ADCPs_x[i+1]:
          if mean_ADCPs_x[i+2] > mean_ADCPs_x[i+1]:
            mean_ADCPs4.append(mean_ADCPs_x[i]); x4.append(i)
            col4 = 'lightskyblue' 
  
        if mean_ADCPs_x[i] < 0  and i < where(mean_ADCPs_x == min(mean_ADCPs_x[13:32]))[0][0] and i > 13 and mean_ADCPs_x[i] > mean_ADCPs_x[i+1]:
          mean_ADCPs3.append(mean_ADCPs_x[i]); x3.append(i)
          col3 = 'salmon'
        if mean_ADCPs_x[i] < 0  and i < where(mean_ADCPs_x == min(mean_ADCPs_x[13:32]))[0][0] and i > 13 and mean_ADCPs_x[i] < mean_ADCPs_x[i+1]:
          mean_ADCPs3.append(mean_ADCPs_x[i]); x3.append(i)
          col3 = 'salmon' 
        if mean_ADCPs_x[i] == min(mean_ADCPs_x[15:32]):
          mean_ADCPs3.append(mean_ADCPs_x[i]); x3.append(i)
          col3 = 'salmon'
          
        if mean_ADCPs_x[i] < 0  and i > where(mean_ADCPs_x == min(mean_ADCPs_x[13:32]))[0][0] and i > 13 and mean_ADCPs_x[i] > mean_ADCPs_x[i+1]:
          mean_ADCPs4.append(mean_ADCPs_x[i]); x4.append(i)
          col4 = 'lightskyblue' 
        if mean_ADCPs_x[i] < 0  and i > where(mean_ADCPs_x == min(mean_ADCPs_x[13:32]))[0][0] and i > 13 and mean_ADCPs_x[i] < mean_ADCPs_x[i+1]:
          mean_ADCPs4.append(mean_ADCPs_x[i]); x4.append(i)
          col4 = 'lightskyblue'  
          
        if mean_ADCPs_x[i] > 0  and i > where(mean_ADCPs_x == min(mean_ADCPs_x[13:32]))[0][0] and i > 13 and mean_ADCPs_x[i] < mean_ADCPs_x[i+1]:
          mean_ADCPs1.append(mean_ADCPs_x[i]); x1.append(i)
          col1 = 'gold' 
        if mean_ADCPs_x[i] > 0  and i > where(mean_ADCPs_x == min(mean_ADCPs_x[13:32]))[0][0] and i > 13 and mean_ADCPs_x[i] > mean_ADCPs_x[i+1]:
          if mean_ADCPs_x[i+2] > mean_ADCPs_x[i+1]:
            mean_ADCPs1.append(mean_ADCPs_x[i]); x1.append(i)
            col1 = 'gold'  
          
      if mean_ADCPs_x[0] > 0:
        
        if mean_ADCPs_x[i] < 0  and i < where(mean_ADCPs_x == min(mean_ADCPs_x))[0][0] and mean_ADCPs_x[i] > mean_ADCPs_x[i+1]:
          mean_ADCPs3.append(mean_ADCPs_x[i]); x3.append(i)
          col3 = 'salmon'
        if mean_ADCPs_x[i] < 0  and i < where(mean_ADCPs_x == min(mean_ADCPs_x))[0][0] and mean_ADCPs_x[i] < mean_ADCPs_x[i+1]:
          if mean_ADCPs_x[i+2] < mean_ADCPs_x[i+1]:
            mean_ADCPs3.append(mean_ADCPs_x[i]); x3.append(i)
            col3 = 'salmon'  
        if mean_ADCPs_x[i] > 0 and mean_ADCPs_x[i+1] < 0 and  (mean_ADCPs_x[i]) > mean_ADCPs_x[i+1]:
          mean_ADCPs3.append(mean_ADCPs_x[i]); x3.append(i)
          col3 = 'salmon'
        if mean_ADCPs_x[i] == min(mean_ADCPs_x):
          mean_ADCPs3.append(mean_ADCPs_x[i]); x3.append(i)
          col3 = 'salmon'
          
        if mean_ADCPs_x[i] < 0 and i > where(mean_ADCPs_x == min(mean_ADCPs_x))[0][0]  and mean_ADCPs_x[i] < mean_ADCPs_x[i+1] and mean_ADCPs_x[i] != min(mean_ADCPs_x):
          mean_ADCPs4.append(mean_ADCPs_x[i]); x4.append(i)
          col4 = 'lightskyblue'
        if mean_ADCPs_x[i] < 0  and i > where(mean_ADCPs_x == min(mean_ADCPs_x))[0][0] and mean_ADCPs_x[i] > mean_ADCPs_x[i+1] and mean_ADCPs_x[i] != min(mean_ADCPs_x):
          if mean_ADCPs_x[i+2] > mean_ADCPs_x[i+1]:
            mean_ADCPs4.append(mean_ADCPs_x[i]); x4.append(i)
            col4 = 'lightskyblue'
        
        if mean_ADCPs_x[i] > 0  and i < where(mean_ADCPs_x == max(mean_ADCPs_x[0:13]))[0][0] and mean_ADCPs_x[i] < mean_ADCPs_x[i+1]:
          mean_ADCPs1.append(mean_ADCPs_x[i]); x1.append(i)
          col1 = 'gold'
        if mean_ADCPs_x[i] > 0  and i < where(mean_ADCPs_x == max(mean_ADCPs_x[0:13]))[0][0] and mean_ADCPs_x[i] > mean_ADCPs_x[i+1]:
          if mean_ADCPs_x[i+2] > mean_ADCPs_x[i+1]:
            mean_ADCPs1.append(mean_ADCPs_x[i]); x1.append(i)
            col1 = 'gold' 
        if mean_ADCPs_x[i] == max(mean_ADCPs_x[0:15]):
          mean_ADCPs1.append(mean_ADCPs_x[i]); x1.append(i)
          col1 = 'gold'

        if mean_ADCPs_x[i] > 0  and i > where(mean_ADCPs_x == max(mean_ADCPs_x[0:13]))[0][0] and i < 13 and mean_ADCPs_x[i] > mean_ADCPs_x[i+1]:
          mean_ADCPs2.append(mean_ADCPs_x[i]);x2.append(i)
          col2 = 'limegreen' 
        if mean_ADCPs_x[i] > 0  and i > where(mean_ADCPs_x == max(mean_ADCPs_x[0:13]))[0][0] and i < 13 and mean_ADCPs_x[i] < mean_ADCPs_x[i+1]:
          if mean_ADCPs_x[i+2] < mean_ADCPs_x[i+1]:
            mean_ADCPs2.append(mean_ADCPs_x[i]); x2.append(i)
            col2 = 'limegreen' 
            
        if mean_ADCPs_x[i] > 0  and i < where(mean_ADCPs_x == max(mean_ADCPs_x[13:32]))[0][0] and i > 13 and mean_ADCPs_x[i] < mean_ADCPs_x[i+1]:
          mean_ADCPs1.append(mean_ADCPs_x[i]); x1.append(i)
          col1 = 'gold'
        if mean_ADCPs_x[i] > 0  and i < where(mean_ADCPs_x == max(mean_ADCPs_x[13:32]))[0][0] and i > 13 and mean_ADCPs_x[i] > mean_ADCPs_x[i+1]:
          mean_ADCPs1.append(mean_ADCPs_x[i]); x1.append(i)
          col1 = 'gold' 
        if mean_ADCPs_x[i] == max(mean_ADCPs_x[15:32]):
          mean_ADCPs1.append(mean_ADCPs_x[i]); x1.append(i)
          col1 = 'gold'
         
        if mean_ADCPs_x[i] > 0  and i > where(mean_ADCPs_x == max(mean_ADCPs_x[13:32]))[0][0] and i > 13 and mean_ADCPs_x[i] > mean_ADCPs_x[i+1]:
          mean_ADCPs2.append(mean_ADCPs_x[i]); x2.append(i)
          col2 = 'limegreen' 
        if mean_ADCPs_x[i] > 0  and i > where(mean_ADCPs_x == max(mean_ADCPs_x[13:32]))[0][0] and i > 13 and mean_ADCPs_x[i] < mean_ADCPs_x[i+1]:
          mean_ADCPs2.append(mean_ADCPs_x[i]); x2.append(i)
          col2 = 'limegreen' 
          
        
    if mean_ADCPs_x[0] < 0:
       
      if mean_ADCPs_x[29] > 0  and 29 >= where(mean_ADCPs_x == min(mean_ADCPs_x[13:32]))[0][0]:
        mean_ADCPs1.append(mean_ADCPs_x[29]); mean_ADCPs1.append(mean_ADCPs_x[30]); x1.append(29); x1.append(30)
        col1 = 'gold'
           
      if mean_ADCPs_x[29] < 0  and 29 <= where(mean_ADCPs_x == min(mean_ADCPs_x[13:32]))[0][0]:
        mean_ADCPs3.append(mean_ADCPs_x[29]); mean_ADCPs3.append(mean_ADCPs_x[30]); x3.append(29); x3.append(30)
        col3 = 'salmon' 
        
      if mean_ADCPs_x[29] < 0  and 29 >= where(mean_ADCPs_x == min(mean_ADCPs_x[13:32]))[0][0]:
        mean_ADCPs4.append(mean_ADCPs_x[29]); mean_ADCPs4.append(mean_ADCPs_x[30]); x4.append(29); x4.append(30)
        col4 = 'lightskyblue' 
        
    if mean_ADCPs_x[0] > 0:   
      if mean_ADCPs_x[29] < 0  and 29 >= where(mean_ADCPs_x == max(mean_ADCPs_x[13:32]))[0][0]:
        mean_ADCPs3.append(mean_ADCPs_x[29]); mean_ADCPs3.append(mean_ADCPs_x[30]); x3.append(29); x3.append(30)
        col3 = 'salmon'
           
      if mean_ADCPs_x[29] > 0  and 29 <= where(mean_ADCPs_x == max(mean_ADCPs_x[13:32]))[0][0]:
        mean_ADCPs1.append(mean_ADCPs_x[29]); mean_ADCPs1.append(mean_ADCPs_x[30]); x1.append(29); x1.append(30)
        col1 = 'gold' 
        
      if mean_ADCPs_x[29] > 0  and 29 >= where(mean_ADCPs_x == max(mean_ADCPs_x[13:32]))[0][0]:
        mean_ADCPs2.append(mean_ADCPs_x[29]); mean_ADCPs2.append(mean_ADCPs_x[30]); x2.append(29); x2.append(30)
        col2 = 'limegreen' 
    
    
    scatter(x1,mean_ADCPs1,linewidths = 5, color=col1, label = 'accelerating phase of the flood current')
    scatter(x2,mean_ADCPs2,linewidths = 5,color=col2, label = 'decelerating phase of the flood current')
    scatter(x3,mean_ADCPs3,linewidths = 5,color=col3, label = 'accelerating phase of the ebb current')
    scatter(x4,mean_ADCPs4,linewidths = 5,color=col4, label = 'decelerating phase of the ebb current')
    
   
    xlabel('Number of transect',fontsize=17)
    ylabel('Mean value of the northward tidal current $\overline{v}$ ['+plv_units+']',fontsize=17)
    legend(loc=1,fontsize=10)
    #title('Northward direction')
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    #lgd1 = legend(loc='upper center', bbox_to_anchor= (0.5, 1.1),fontsize=10)
    
    if savefigg == 1:
      figfilename=pl_var+'_parabolafitting_phasetide_timeseries'.zfill(2)+figtype
      savefig(figdir+'/'+figfilename,dpi=300)
      close(fignr)
    
  

  return  
  
  
# -----------------------------------------------------------------------------------

def std_VS_depth(instance_1,instance_2,instance_step,bad_fraction,pl_var):

  # Year, month, error threshold
  yy='2021'; mm = '07';  error_thresh = 0.22
  # Days
  days = ['20','28']  #20/07/2021 calm day; 28/07/2022 windy day
  # Crossings
  instance_1 = 0; instance_2 = 32
  # Number of the windy and no windy days
  num_wind = 1; num_no_wind = 1 
  
  values_wind = zeros([50,num_wind]); values_no_wind = zeros([50,num_no_wind])
  values_diff_wind = zeros([50,num_wind]); values_diff_no_wind = zeros([50,num_no_wind])
  rms_array_wind  =zeros([50,num_wind]); rms_array_no_wind  =zeros([50,num_no_wind])
  
  mean_stds_wind = []; mean_stds_no_wind = []; rms_wind =[]; rms_no_wind=[]; mean_diff_vel_no_wind = []; mean_diff_vel_wind= []; mean_diff_vel_wind =[]; mean_diff_vel_no_wind=[]
  sum_wind = 0; sum_no_wind =0 
  for dd in days:
    dep = []
    means, mean_ADCP_sep, stds_dif, plv_units, depth_vel_std, domain_depth, depth_vel_error, dep_longname, mean_ADCPs_per_depth_square, mean_ADCPs_per_depth, depth_vel_diff = mean_values(yy,mm,dd,instance_1,instance_2,error_thresh,pl_vars)
    dep.extend(domain_depth) 
    print(dd)
   
    for j in range(len(dep)):
      if dd =='28':
        values_wind[j,sum_wind] = depth_vel_std[j]
        rms_array_wind[j,sum_wind] = mean_ADCPs_per_depth_square[j]
        values_diff_wind[j,sum_wind] = depth_vel_diff[j]
      else: 
        values_no_wind[j,sum_no_wind] = depth_vel_std[j]
        rms_array_no_wind[j,sum_no_wind] = mean_ADCPs_per_depth_square[j]
        values_diff_no_wind[j,sum_no_wind] = depth_vel_diff[j]
    
    if dd =='28':
      sum_wind = sum_wind + 1
    else:
      sum_no_wind  = sum_no_wind + 1
       
  # Compute the mean value for each depth layer (when the number of the windy and no windy days > 1)
  for i in range(len(dep)): 
    mean_stds_wind.append(mean(values_wind[i,:]))
    mean_stds_no_wind.append(mean(values_no_wind[i,:]))
    mean_diff_vel_wind.append(mean(values_diff_wind[i,:]))
    mean_diff_vel_no_wind.append(mean(values_diff_no_wind[i,:]))
    rms_no_wind.append(mean(rms_array_no_wind[i,:]))
    rms_wind.append(mean(rms_array_wind[i,:]))
    
  #----------------------------------------------------  
  # Plot the timeseries of the tidal cycle for two different depth layers + rms velocity for each depth
  
  fignr=7 
  figure(fignr,(9,6),None,facecolor='w',edgecolor='k')
  scatter(arange(0,32,1), array(mean_ADCPs_per_depth[:,1]), color = 'royalblue',label = 'Depth: ' +str(round(dep[1],2))+' m')
  scatter(arange(0,32,1), array(mean_ADCPs_per_depth[:,30]), color = 'salmon',  label = 'Depth: ' +str(round(dep[30],2))+' m') 
  plot(linspace(0,32,100),ones(len(linspace(0,32,100)))*rms_array_no_wind[1,0],color = 'royalblue',Linestyle = '--', label = 'u$_{rms}$ at h = '+ str(round(dep[1],2))+' m')
  plot(linspace(0,32,100),ones(len(linspace(0,32,100)))*rms_array_no_wind[30,0],color = 'salmon',Linestyle = '--', label = 'u$_{rms}$ at h = '+ str(round(dep[30],2))+' m')
  xlabel('Number of transect',fontsize=13)
  if pl_var == 'EAST_VEL':
    ylabel('Mean eastward velocity [m/s]',fontsize=13)
  if pl_var == 'NORTH_VEL':
    ylabel('Mean northward velocity [m/s]',fontsize=13)
  legend()
  
  major_ticks_x = np.arange(0, 32, 5)
  minor_ticks_x = np.arange(0, 32, 1)
  ax = plt.gca()
  ax.set_xticks(major_ticks_x)
  ax.set_xticks(minor_ticks_x, minor=True)
  ax.grid(which='both')
  ax.grid(which='minor', alpha=0.3)
  ax.grid(which='major', alpha=0.6)


  if savefigg == 1:
    figfilename=pl_var+str(yy)+str(mm)+'_'+str(days)+'_tide_VS_depth_urms'.zfill(2)+figtype
    savefig(figdir+'/'+figfilename,dpi=300)
    close(fignr)
  
  
  # ----------------------------------------------------
  # Plot the std of the velocity difference over depth
  
  fignr=8 
  figure(fignr,(7,6),None,facecolor='w',edgecolor='k')
  scatter(mean_stds_no_wind,dep, s = 20, color='royalblue', label = '2 beaufort wind') 
  scatter(mean_stds_wind,dep, s = 20, color='salmon', label = '5 beaufort wind') 
  legend(fontsize=13)
  plt.yticks(fontsize=11)
  plt.xticks(fontsize=11)
  #errorbar(stds,dep,xerr=error_std,fmt='o', ecolor='black',capsize=3) 
  #xlim([0,0.5])
  #gca().invert_xaxis() 
  

  major_ticks_x = np.arange(-0.02, 0.4, 0.04)
  minor_ticks_x = np.arange(-0.02, 0.4, 0.02)
  major_ticks_y = np.arange(-30, -4, 5)
  minor_ticks_y = np.arange(-30, -4, 1)
  ax = plt.gca()
  ax.set_xticks(major_ticks_x)
  ax.set_xticks(minor_ticks_x, minor=True)
  ax.set_yticks(major_ticks_y)
  ax.set_yticks(minor_ticks_y, minor=True)
  ax.grid(which='both')
  ax.grid(which='minor', alpha=0.3)
  ax.grid(which='major', alpha=0.6)
  

  if pl_vars=='EAST_VEL':
      xlabel('Std value of u$_{hd}$ - u$_{tx}$ ['+plv_units+']',fontsize=13)
      ylabel(dep_longname+' [m]',fontsize=13)
  elif pl_vars=='NORTH_VEL':
      xlabel('Std value of v$_{hd}$ - v$_{tx}$ ['+plv_units+']',fontsize=13)
      ylabel(dep_longname+' [m]',fontsize=13)
  #legend()
  
  if savefigg == 1:
    figfilename=pl_var+str(yy)+str(mm)+'_'+str(days)+'_turbulence_VS_depth'.zfill(2)+figtype
    savefig(figdir+'/'+figfilename,dpi=300)
    close(fignr)
  

  # --------------------------------------------------
  # Plot the rms of the velocity difference over depth
  
  fignr=9 
  figure(fignr,(7,6),None,facecolor='w',edgecolor='k')
  scatter(rms_no_wind,dep, color='royalblue', label = '2 beaufort wind') 
  scatter(rms_wind,dep, color='salmon', label = '5 beaufort wind') 
  legend()

  if pl_vars=='EAST_VEL':
      xlabel('$u_{rms}$ ['+plv_units+']',fontsize=13)
      ylabel(dep_longname+' [m]',fontsize=13)
  elif pl_vars=='NORTH_VEL':
      xlabel('$v_{rms}$ ['+plv_units+']',fontsize=13)
      ylabel(dep_longname+' [m]',fontsize=13)
  
  if savefigg == 1:
    figfilename=pl_var+str(yy)+str(mm)+'_'+str(days)+'_VELrms_VS_depth'.zfill(2)+figtype
    savefig(figdir+'/'+figfilename,dpi=300)
    close(fignr)

  
  #-----------------------------------------
  # Plot the (u_hd-u_tx)/rms over depth

  vel_diff_over_rms_no_wind = zeros([len(dep),1])
  for i in range(1):
    for j in range(len(dep)):
      vel_diff_over_rms_no_wind[j,i] = values_diff_no_wind[j,i]/rms_array_no_wind[j,i]
      
  vel_diff_over_rms_wind = zeros([len(dep),1])
  for i in range(1):
    for j in range(len(dep)):
      vel_diff_over_rms_wind[j,i] = values_diff_wind[j,i]/rms_array_wind[j,i]   
   
     
  mean_vel_diff_over_rms_no_wind = []; mean_vel_diff_over_rms_wind = []
  for i in range(len(dep)): 
    mean_vel_diff_over_rms_no_wind.append(mean(vel_diff_over_rms_no_wind[i,:]))
    mean_vel_diff_over_rms_wind.append(mean(vel_diff_over_rms_wind[i,:]))

 
  
  fignr=10 
  figure(fignr,(7,6),None,facecolor='w',edgecolor='k')
  scatter(mean_vel_diff_over_rms_no_wind,dep, color='royalblue', label = '2 beaufort wind') 
  scatter(mean_vel_diff_over_rms_wind,dep, color='salmon', label = '5 beaufort wind') 
  legend()
  
  if pl_vars=='EAST_VEL':
      xlabel('($u_{hd}$-$u_{tx}$)/$u_{rms}$ ',fontsize=13)
      ylabel(dep_longname+' [m]',fontsize=13)
  elif pl_vars=='NORTH_VEL':
      xlabel('($v_{hd}$-$v_{tx}$)/$v_{rms}$ ',fontsize=13)
      ylabel(dep_longname+' [m]',fontsize=13)
     
  if savefigg == 1:   
    figfilename=pl_var+str(yy)+str(mm)+'_'+str(days)+'_dv_over_urms_VS_depth'.zfill(2)+figtype
    savefig(figdir+'/'+figfilename,dpi=300)
    close(fignr)
  
  
  #-----------------------------------------
  # Plot the std(u_hd-u_tx)/rms over depth

  vel_diff_over_rms_no_wind = zeros([len(dep),1])
  for i in range(1):
    for j in range(len(dep)):
      vel_diff_over_rms_no_wind[j,i] = values_no_wind[j,i]/rms_array_no_wind[j,i]
      
  vel_diff_over_rms_wind = zeros([len(dep),1])
  for i in range(1):
    for j in range(len(dep)):
      vel_diff_over_rms_wind[j,i] = values_wind[j,i]/rms_array_wind[j,i]   
   
     
  mean_vel_diff_over_rms_no_wind = []; mean_vel_diff_over_rms_wind = []
  for i in range(len(dep)): 
    mean_vel_diff_over_rms_no_wind.append(mean(vel_diff_over_rms_no_wind[i,:]))
    mean_vel_diff_over_rms_wind.append(mean(vel_diff_over_rms_wind[i,:]))

  fignr=11 
  figure(fignr,(7,6),None,facecolor='w',edgecolor='k')
  scatter(mean_vel_diff_over_rms_no_wind,dep, color='royalblue', label = '2 beaufort wind') 
  scatter(mean_vel_diff_over_rms_wind,dep, color='salmon', label = '5 beaufort wind') 
  legend()
  
  if pl_vars=='EAST_VEL':
      xlabel('std($u_{hd}$-$u_{tx}$)/$u_{rms}$ ',fontsize=13)
      ylabel(dep_longname+' [m]',fontsize=13)
  elif pl_vars=='NORTH_VEL':
      xlabel('std($v_{hd}$-$v_{tx}$)/$v_{rms}$ ',fontsize=13)
      ylabel(dep_longname+' [m]',fontsize=13)
   
  if savefigg == 1:   
    figfilename=pl_var+str(yy)+str(mm)+'_'+str(days)+'_std(dv)_over_urms_VS_depth'.zfill(2)+figtype
    savefig(figdir+'/'+figfilename,dpi=300)
    close(fignr)

  return  
  

############################### Main ##################################################3
indir = '/home/prdusr/data_out/TESO/daily'
names=['hd','tx']
figdir='/home/akaraoli/python_scripts/figures/'
figtype='.jpg'
savefigg = 0 

# Input of the date within the functions!
# Crossings
instance_1 = 0; instance_2 = 32; instance_step=1; bad_fraction=0.3

# Choose velocity 
pl_vars='EAST_VEL'
#pl_vars='NORTH_VEL'



#---------------------------- Plot the mean values of the differences in horizontal current speed along the transect for different days ---------------------------- 
plot_distributions(indir,names,instance_1,instance_2,instance_step,bad_fraction,pl_vars) # line 39

# --------------------------------Plot std of the horizontal velocity diffrence as a function of tidal current -------------------------------------------
std_VS_tidal_current(instance_1,instance_2,instance_step,bad_fraction,pl_vars)  # line 471

#-------------------- Plot std of the horizontal velocity diffrence as a function of tidal current - check of the phas of the tide -------------------------
std_VS_phase_tide(instance_1,instance_2,instance_step,bad_fraction,pl_vars) # line 537

#----------------------------------------------------- Turbulence as a function of depth ----------------------------------------------------------------------
std_VS_depth(instance_1,instance_2,instance_step,bad_fraction,pl_vars)  # line 1157


show()
