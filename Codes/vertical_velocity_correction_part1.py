#! /usr/bin/env python

# python script to plot TESO transects

# Author: Athina Karaoli, May 2022

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
from scipy.stats import skew
from tabulate import tabulate
from datetime import datetime, timedelta
import statsmodels.api as sm
import math
import matplotlib.colors as mcolors
import sgolay2
from scipy import interpolate
from scipy import ndimage
import scipy.signal as sg


############# Functions ###############################################

# ----------------------------------------------------------------------
def find_nearest(array, value):
    array = asarray(array)
    idx = (abs(array - value)).argmin()
    return array[idx]
    
#-----------------------------------------------------------------------
def func(x, a, b, c):
    return a * np.exp(b * x) + c
    
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
        
# ----------------------------------------------------------------------------------------------
# Function which describes the corrected vertical velocity for the ferry's motion

def poly2d_fun_front(xy, *coefficients):

    x = xy[:, 0]
    y = xy[:, 1]
    f = ((coefficients[0])*x + coefficients[1])*y + ((coefficients[2])*x + coefficients[3])
    return f
    
def poly2d_fun_back(xy, *coefficients):

    x = xy[:, 0]
    y = xy[:, 1]
    f = ((coefficients[0])*x + coefficients[1])*y + ((coefficients[2])*x + coefficients[3])
    return f


# ----------------------------------------------------------------------------------------------

def correction_vertical_velocity_for_ferry_veloicty(indir,yy,mm,dd,error_thresh,instance, fignr):

  # Estimation of the environmnetal flow
  environmental_flow_hd, environmental_flow_tx = compute_envir_flow(indir,names,yy,mm,dd,error_thresh, instance)
  
  for name in names:
    [lat,lat_units,lat_longname]=loadvar(indir,name,yy,mm,dd,'LATITUDE',instance)
    [dep,dep_units,dep_longname]=loadvar(indir,name,yy,mm,dd,'BINDEPTH',instance)
    [siv,siv_units,siv_longname]=loadvar(indir,name,yy,mm,dd,'ERROR_VEL',instance)
    [plv,plv_units,plv_longname]=loadvar(indir,name,yy,mm,dd,varname,instance)
    [bt,bt_units,bt_longname]=loadvar(indir,name,yy,mm,dd,bt_vars,instance)

    sd=shape(dep)
    depstep=dep[0,1]-dep[0,0]
    depax=arange(dep[0,0],dep[0,0]+sd[1]*depstep,depstep)
      
    latmax=0
    while lat[latmax]>0:
      latmax=latmax+1
      if latmax>len(lat):
        break
        
    # Direction of the ferry    
    if name=='hd':
      if (lat[100]-lat[200]) > 0:
        ar='right'
      else:
        ar='left'    

    # Save the latitude data and the bottom track velocity data for each ADCP separately 
    if name == 'hd':
      lat_hd =[]; bt_hd =[]; limit_hd = latmax
      for i in range(len(lat[0:latmax])):
        lat_hd.append(lat[i]); bt_hd.append(-bt[i])
    if name == 'tx':
      lat_tx = []; bt_tx = []; limit_tx = latmax
      for i in range(len(lat[0:latmax])):
        lat_tx.append(lat[i]); bt_tx.append(-bt[i]) 
    
        
    below_bottom=where(abs(plv)>10.,1,0)    
    plvp=where(abs(siv)>error_thresh,-9999.,plv)            # don't plot points with large error velocity
    plvp=where(below_bottom==1,NaN,plvp)
    plv_cleaned=remove_bad_columns(plvp)                    # remove columns with a large number of errors
    plvT=1*plv_cleaned.T

    # Save the velocity data for each ADCP separately
    if name == 'hd':
      vel_hd = zeros([len(depax),latmax]); limit_hd = latmax
      for i in range(len(depax)):
        for j in range(latmax):
          vel_hd[i,j] = plvT[i,j]
    if name == 'tx':
      vel_tx = zeros([len(depax),latmax]); limit_tx = latmax
      for i in range(len(depax)):
        for j in range(latmax):
          vel_tx[i,j] = plvT[i,j]
          
  # Exclude the data near the harbors
  depth_min = 0; depth_max = 30; lat_min = 52.967;  lat_max = 52.999
  for b in range(len(depax)):
    for c in range(len(lat_hd)):
      if depax[b] < depth_min or depax[b] > depth_max:
        vel_hd[b,:] = nan
        if lat_hd[c] < lat_min or lat_hd[c] > lat_max:
          vel_hd[b,c] = nan
      if lat_hd[c] < lat_min or lat_hd[c] > lat_max:
        vel_hd[b,c] = nan
        bt_hd[c] = nan
   
  for b in range(len(depax)):
    for c in range(len(lat_tx)):
      if depax[b] < depth_min or depax[b] > depth_max:
        vel_tx[b,:] = nan
        if lat_tx[c] < lat_min or lat_tx[c] > lat_max:
          vel_tx[b,c] = nan
      if lat_tx[c] < lat_min or lat_tx[c] > lat_max:
          vel_tx[b,c] = nan
          bt_tx[c] = nan

  # Define the front and the rear ADCP
  ship_speed_front =[];  ship_speed_back =[]
  if ar == 'right':
    lat_front = lat_hd; vel_front = vel_hd; limit_front = limit_hd;
    lat_back = lat_tx; vel_back = vel_tx; limit_back = limit_tx
    for k in range(len(bt_hd)):
      ship_speed_front.append(bt_hd[k] - environmental_flow_hd[k])
    for k in range(len(bt_tx)):
      ship_speed_back.append(bt_tx[k] - environmental_flow_tx[k])
  if ar=='left':
    lat_front = lat_tx; vel_front = vel_tx; limit_front = limit_tx;
    lat_back = lat_hd; vel_back = vel_hd; limit_back = limit_hd
    for k in range(len(bt_hd)):
      ship_speed_back.append(bt_hd[k] - environmental_flow_hd[k])
    for k in range(len(bt_tx)):
      ship_speed_front.append(bt_tx[k] - environmental_flow_tx[k])
    
  print(yy+mm+dd+'_'+str(instance)) 
  
  # -------------------------- Fit the measured data (2nd correctcion as defined in the report)-----------------------------------------------
  
  # Front ADCP
  target_coefficients = (0.1,0.1,0.1,0.1)  # guess the unknown coefficients 
  x = abs(array(ship_speed_front)); y = -depax
  im_x, im_y = np.meshgrid(x, y)
  xdata = np.c_[im_x.flatten(), im_y.flatten()]
  vel_ADCP_front = vel_front.flatten() # velocity measurements inlcuding nan values 
  vel_ADCP2_front = vel_ADCP_front[~np.isnan(vel_ADCP_front)] # velocity measurements excluding nan values 
  xdata2 = xdata[~np.isnan(vel_ADCP_front),:] # depth and ship speed velocity 
  ydata2=vel_ADCP2_front # velocity measurements by ADCP
  popt_front, pcov_front = curve_fit(poly2d_fun_front, xdata=xdata2, ydata=ydata2, p0=target_coefficients )
  fit_vel_front = poly2d_fun_front(xdata2, *popt_front)#.reshape(len(ydata2), len(xdata2))
  print('Front ADCP')
  print('w = ({:1.5f}'.format(popt_front[0])+'Vs + {:1.5f}'.format(popt_front[1])+')z + ({:1.5f}'.format(popt_front[2])+'Vs + {:1.5f}'.format(popt_front[3])+')')

  # Add the nan values from the original data to the corrected data
  nan_mask = isnan(vel_ADCP_front)
  fit_vel_new_front = empty(vel_ADCP_front.shape)
  fit_vel_new_front[~nan_mask] = fit_vel_front
  fit_vel_new_front[nan_mask] = nan
  fit_vel_new_front = fit_vel_new_front.reshape(len(y), len(x))  
  
  # Compute the final corrected vertical velocity 
  corrected_front = zeros([len(depax),len(lat_front)]) 
  for i in range(len(depax)):
    for j in range(len(lat_front)):
      corrected_front[i,j] = vel_front[i,j] -  fit_vel_new_front[i,j]
  
  
  # Back ADCP   
  target_coefficients = (0.1,0.1,0.1,0.1)  # guess the unknown coefficients 
  x = abs(array(ship_speed_back)); y = -depax
  im_x, im_y = np.meshgrid(x, y)
  xdata = np.c_[im_x.flatten(), im_y.flatten()]
  vel_ADCP_back = vel_back.flatten() # velocity measurements inlcuding nan values 
  vel_ADCP2_back = vel_ADCP_back[~np.isnan(vel_ADCP_back)] # velocity measurements excluding nan values 
  xdata2 = xdata[~np.isnan(vel_ADCP_back),:] # depth and ship speed velocity 
  ydata2=vel_ADCP2_back # velocity measurements by ADCP
  popt_back, pcov_back = curve_fit(poly2d_fun_back, xdata=xdata2, ydata=ydata2, p0=target_coefficients )
  fit_vel_back = poly2d_fun_back(xdata2, *popt_back)#.reshape(len(ydata2), len(xdata2))
  print('Rear ADCP')
  print('w = ({:1.5f}'.format(popt_back[0])+'Vs + {:1.5f}'.format(popt_back[1])+')z + ({:1.5f}'.format(popt_back[2])+'Vs + {:1.5f}'.format(popt_back[3])+')')

  # Add the nan values from the original data to the corrected data
  nan_mask = isnan(vel_ADCP_back)
  fit_vel_new_back = empty(vel_ADCP_back.shape)
  fit_vel_new_back[~nan_mask] = fit_vel_back
  fit_vel_new_back[nan_mask] = nan
  fit_vel_new_back = fit_vel_new_back.reshape(len(y), len(x))  
  
  # Compute the final corrected vertical velocity 
  corrected_back = zeros([len(depax),len(lat_back)]) 
  for i in range(len(depax)):
    for j in range(len(lat_back)):
      corrected_back[i,j] = vel_back[i,j] -  fit_vel_new_back[i,j]
    
  # -----------------------------------------------------------------------------------
  # ----------------------- Plots for the front ADCP  -----------------------------------
  # -----------------------------------------------------------------------------------
  
  # Max values for the colorbar
  vmax1=0.4; vmax2=0.2
  
  figure(fignr,(fs1+6,fs2+8),None,facecolor='w',edgecolor='k')
  nsub = 0; nrows = 3; ncols=2
   
  # ----------------- Plot the original/measured data -----------------------

  subplot(nrows,ncols,nsub+1)
  pcolormesh(array(lat_front),-depax,vel_front,shading='auto', vmin=-vmax1,vmax=vmax1,cmap='coolwarm')
  ylim([-30, -5])
  gca().invert_xaxis()
  colorbar()
  ylabel(dep_longname+' ['+dep_units+']',fontsize=fs)
  xlabel(lat_longname+' ['+lat_units+']',fontsize=fs)
  title('Front ADCP: '+plv_longname,fontsize=fs)
  plt.text(52.971,-29.5,'Den Helder')
  plt.text(53.002,-29.5,'Texel')
 
  if ar == 'right':
    arrow(52.985, -29.5, -0.005, 0.0,fc='k',ec='k',length_includes_head=True, head_width=0.5, head_length=0.001)
  elif ar == 'left':
    arrow(52.980, -29.5, 0.005, 0.0,fc='k',ec='k',length_includes_head=True, head_width=0.5, head_length=0.001)
    
  # Remove nan values from the data
  tot_vel_front =[]
  for i in range(len(depax)):
    for j in range(len(lat_front)): 
      bo = ~isnan(vel_front[i,j] )
      if  bo == True: 
        tot_vel_front.append(vel_front[i,j] )
 
  # Plot histograms for each ADCP
  subplot(nrows,ncols,nsub+2)
  hist(tot_vel_front, bins = 1000, range=(-vmax1, vmax1), color = 'skyblue')
  title('Original data', fontsize=13)
  xlabel(plv_longname+' ['+plv_units+']',fontsize=fs)
  
  # Generate text to write.
  text3 = 'mean = {:1.3f} '.format(mean(tot_vel_front))+ ' m/s'
  text4 = 'std =  ' + str(round(std(tot_vel_front),4))+ ' m/s'
  text = text3 + '\n' + text4 
  annotate(text, xy=(0.08, 1), xytext=(-15, -15), fontsize=14,xycoords='axes fraction', textcoords='offset points',bbox=dict(facecolor='white', alpha=0.8),horizontalalignment='left', verticalalignment='top')
  
  
  # ------------------- Plot the fitted data ------------------------------------
  
  subplot(nrows,ncols,nsub+3)
  pcolormesh(array(lat_front),-depax,fit_vel_new_front,shading='auto', vmin=-vmax1,vmax=vmax1,cmap='coolwarm')
  ylim([-30, -5])
  gca().invert_xaxis()
  colorbar()
  ylabel(dep_longname+' ['+dep_units+']',fontsize=fs)
  xlabel(lat_longname+' ['+lat_units+']',fontsize=fs)
  plt.text(52.971,-29.5,'Den Helder')
  plt.text(53.002,-29.5,'Texel')

  if ar == 'right':
    arrow(52.985, -29.5, -0.005, 0.0,fc='k',ec='k',length_includes_head=True, head_width=0.5, head_length=0.001)
  elif ar == 'left':
    arrow(52.980, -29.5, 0.005, 0.0,fc='k',ec='k',length_includes_head=True, head_width=0.5, head_length=0.001)
     
  # Remove the nan values
  tot_vel_front =[]
  for i in range(len(depax)):
    for j in range(len(lat_front)): 
      bo = ~isnan(fit_vel_new_front[i,j] )
      if  bo == True: 
        tot_vel_front.append(fit_vel_new_front[i,j] )
 
  # Plot histograms for each ADCP
  subplot(nrows,ncols,nsub+4)
  hist(tot_vel_front, bins = 1000, range=(-vmax1, vmax1), color = 'orange')
  title('Fitted data', fontsize=13)
  xlabel(plv_longname+' ['+plv_units+']',fontsize=fs)
  
  # Generate text to write.
  text3 = 'mean = {:1.3f} '.format(mean(tot_vel_front))+ ' m/s'
  text4 = 'std =  ' + str(round(std(tot_vel_front),4))+ ' m/s'
  text = text3 + '\n' + text4 
  annotate(text, xy=(0.08, 1), xytext=(-15, -15), fontsize=14,xycoords='axes fraction', textcoords='offset points',bbox=dict(facecolor='white', alpha=0.8),horizontalalignment='left', verticalalignment='top')
   
  # -------------------- Plot the corrected vertical velocity -----------------------------
  
  subplot(nrows,ncols,nsub+5)
  pcolormesh(array(lat_front),-depax,corrected_front,shading='auto', vmin=-vmax2,vmax=vmax2,cmap='coolwarm')
  ylim([-30, -5])
  gca().invert_xaxis()
  colorbar()
  ylabel(dep_longname+' ['+dep_units+']',fontsize=fs)
  xlabel(lat_longname+' ['+lat_units+']',fontsize=fs)
  plt.text(52.971,-29.5,'Den Helder')
  plt.text(53.002,-29.5,'Texel')
  
  if ar == 'right':
    arrow(52.985, -29.5, -0.005, 0.0,fc='k',ec='k',length_includes_head=True, head_width=0.5, head_length=0.001)
  elif ar == 'left':
    arrow(52.980, -29.5, 0.005, 0.0,fc='k',ec='k',length_includes_head=True, head_width=0.5, head_length=0.001)
  
  # Remove the nan values 
  tot_vel_front =[]
  for i in range(len(depax)):
    for j in range(len(lat_front)): 
      bo = ~isnan(corrected_front[i,j] )
      if  bo == True: 
        tot_vel_front.append(corrected_front[i,j] )
 
  # Plot histograms for each ADCP
  subplot(nrows,ncols,nsub+6)
  hist(tot_vel_front, bins = 1000, range=(-vmax2, vmax2), color = 'salmon')
  title('Corrected data', fontsize=13)
  xlabel(plv_longname+' ['+plv_units+']',fontsize=fs)
  
  # Generate text to write.
  text3 = 'mean = {:1.3e} '.format(mean(tot_vel_front))+ ' m/s'
  text4 = 'std =  ' + str(round(std(tot_vel_front),4))+ ' m/s'
  text = text3 + '\n' + text4 
  annotate(text, xy=(0.08, 1), xytext=(-15, -15), fontsize=14,xycoords='axes fraction', textcoords='offset points',bbox=dict(facecolor='white', alpha=0.8),horizontalalignment='left', verticalalignment='top')

  if savefigg ==1:
    figfilename=varname+'_'+str(yy)+str(mm)+str(dd)+'_'+str(instance)+'_CORRECTION_subplots_front'.zfill(2)+figtype
    savefig(figdir+'/'+figfilename,dpi=300)
    close(fignr)
    
  # -----------------------------------------------------------------------------------
  # ----------------------- Plots for the rear ADCP  -----------------------------------
  # -----------------------------------------------------------------------------------
 
  fignr = fignr + 1
  nrows = 3; ncols=2; nsub = 0 
  figure(fignr,(fs1+6,fs2+8),None,facecolor='w',edgecolor='k')
  
  # ----------------- Plot the original/measured data -----------------------

  subplot(nrows,ncols,nsub+1)
  pcolormesh(array(lat_back),-depax,vel_back,shading='auto',vmin=-vmax1,vmax=vmax1,cmap='coolwarm')
  ylim([-30, -5])
  gca().invert_xaxis()
  colorbar()
  ylabel(dep_longname+' ['+dep_units+']',fontsize=fs)
  xlabel(lat_longname+' ['+lat_units+']',fontsize=fs)
  title('Rear ADCP: '+plv_longname,fontsize=fs)
  plt.text(52.971,-29.5,'Den Helder')
  plt.text(53.002,-29.5,'Texel')
 
  if ar == 'right':
    arrow(52.985, -29.5, -0.005, 0.0,fc='k',ec='k',length_includes_head=True, head_width=0.5, head_length=0.001)
  elif ar == 'left':
    arrow(52.980, -29.5, 0.005, 0.0,fc='k',ec='k',length_includes_head=True, head_width=0.5, head_length=0.001)
     
  # Remove the nan values
  tot_vel_back =[]
  for i in range(len(depax)):
    for j in range(len(lat_back)): 
      bo = ~isnan(vel_back[i,j] )
      if  bo == True: 
        tot_vel_back.append(vel_back[i,j] )
 
  # Plot histograms for each ADCP
  subplot(nrows,ncols,nsub+2)
  hist(tot_vel_back, bins = 1000, range=(-vmax1, vmax1), color = 'skyblue')
  title('Original data', fontsize=13)
  xlabel(plv_longname+' ['+plv_units+']',fontsize=fs)
  
  # Generate text to write.
  text3 = 'mean = {:1.3f} '.format(mean(tot_vel_back))+ ' m/s'
  text4 = 'std =  ' + str(round(std(tot_vel_back),4))+ ' m/s'
  text = text3 + '\n' + text4 
  annotate(text, xy=(0.08, 1), xytext=(-15, -15), fontsize=14,xycoords='axes fraction', textcoords='offset points',bbox=dict(facecolor='white', alpha=0.8),horizontalalignment='left', verticalalignment='top')
  
  
  # ------------------- Plot the fitted data ------------------------------------
  
  subplot(nrows,ncols,nsub+3)
  pcolormesh(array(lat_back),-depax,fit_vel_new_back,shading='auto', vmin=-vmax1,vmax=vmax1,cmap='coolwarm')
  ylim([-30, -5])
  gca().invert_xaxis()
  colorbar()
  ylabel(dep_longname+' ['+dep_units+']',fontsize=fs)
  xlabel(lat_longname+' ['+lat_units+']',fontsize=fs)
  plt.text(52.971,-29.5,'Den Helder')
  plt.text(53.002,-29.5,'Texel')

  if ar == 'right':
    arrow(52.985, -29.5, -0.005, 0.0,fc='k',ec='k',length_includes_head=True, head_width=0.5, head_length=0.001)
  elif ar == 'left':
    arrow(52.980, -29.5, 0.005, 0.0,fc='k',ec='k',length_includes_head=True, head_width=0.5, head_length=0.001)
    
  # Remove the nan values
  tot_vel_back =[]
  for i in range(len(depax)):
    for j in range(len(lat_back)): 
      bo = ~isnan(fit_vel_new_back[i,j] )
      if  bo == True: 
        tot_vel_back.append(fit_vel_new_back[i,j] )
 
  # Plot histograms for each ADCP
  subplot(nrows,ncols,nsub+4)
  hist(tot_vel_back, bins = 1000, range=(-vmax1, vmax1), color = 'orange')
  title('Fitted data', fontsize=13)
  xlabel(plv_longname+' ['+plv_units+']',fontsize=fs)
  
  # Generate text to write.
  text3 = 'mean = {:1.3f} '.format(mean(tot_vel_back))+ ' m/s'
  text4 = 'std =  ' + str(round(std(tot_vel_back),4))+ ' m/s'
  text = text3 + '\n' + text4 
  annotate(text, xy=(0.08, 1), xytext=(-15, -15), fontsize=14,xycoords='axes fraction', textcoords='offset points',bbox=dict(facecolor='white', alpha=0.8),horizontalalignment='left', verticalalignment='top')
  
  # ------------- Plot the corrected vertical velocity ---------------------------------
  
  subplot(nrows,ncols,nsub+5)
  pcolormesh(array(lat_back),-depax,corrected_back, shading='auto',vmin=-vmax2,vmax=vmax2,cmap='coolwarm')
  ylim([-30, -5])
  gca().invert_xaxis()
  colorbar()
  ylabel(dep_longname+' ['+dep_units+']',fontsize=fs)
  xlabel(lat_longname+' ['+lat_units+']',fontsize=fs)
  plt.text(52.971,-29.5,'Den Helder')
  plt.text(53.002,-29.5,'Texel')
  
  if ar == 'right':
    arrow(52.985, -29.5, -0.005, 0.0,fc='k',ec='k',length_includes_head=True, head_width=0.5, head_length=0.001)
  elif ar == 'left':
    arrow(52.980, -29.5, 0.005, 0.0,fc='k',ec='k',length_includes_head=True, head_width=0.5, head_length=0.001)
  
  # Remove the nan values
  tot_vel_back =[]
  for i in range(len(depax)):
    for j in range(len(lat_back)): 
      bo = ~isnan(corrected_back[i,j] )
      if  bo == True: 
        tot_vel_back.append(corrected_back[i,j] )
 
  # Plot histograms for each ADCP
  subplot(nrows,ncols,nsub+6)
  hist(tot_vel_back, bins = 1000, range=(-vmax2, vmax2), color = 'salmon')
  title('Corrected data', fontsize=13)
  xlabel(plv_longname+' ['+plv_units+']',fontsize=fs)
  
  # Generate text to write.
  text3 = 'mean = {:1.3e} '.format(mean(tot_vel_back))+ ' m/s'
  text4 = 'std =  ' + str(round(std(tot_vel_back),4))+ ' m/s'
  text = text3 + '\n' + text4 
  annotate(text, xy=(0.08, 1), xytext=(-15, -15), fontsize=14,xycoords='axes fraction', textcoords='offset points',bbox=dict(facecolor='white', alpha=0.8),horizontalalignment='left', verticalalignment='top')
  
  if savefigg ==1:
    figfilename=varname+'_'+str(yy)+str(mm)+str(dd)+'_'+str(instance)+'_CORRECTION_subplots_back'.zfill(2)+figtype
    savefig(figdir+'/'+figfilename,dpi=300)
    close(fignr)

  return fignr
  
  
# ----------------------------------------------------------------------------------------------------------------------------------
    
def correction_vertical_velocity_smooth(indir, yy, mm, dd, error_thresh, instance, fignr):

  # Estimation of the environmental flow
  environmental_flow_hd, environmental_flow_tx = compute_envir_flow(indir,names,yy,mm,dd,error_thresh, instance)
  
  for name in names:
    [lat,lat_units,lat_longname]=loadvar(indir,name,yy,mm,dd,'LATITUDE',instance)
    [dep,dep_units,dep_longname]=loadvar(indir,name,yy,mm,dd,'BINDEPTH',instance)
    [siv,siv_units,siv_longname]=loadvar(indir,name,yy,mm,dd,'ERROR_VEL',instance)
    [plv,plv_units,plv_longname]=loadvar(indir,name,yy,mm,dd,varname,instance)
    [bt,bt_units,bt_longname]=loadvar(indir,name,yy,mm,dd,bt_vars,instance)

    sd=shape(dep)
    depstep=dep[0,1]-dep[0,0]
    depax=arange(dep[0,0],dep[0,0]+sd[1]*depstep,depstep)
      
    latmax=0
    while lat[latmax]>0:
      latmax=latmax+1
      if latmax>len(lat):
        break
        
    # Direction of the ferry     
    if name=='hd':
      if (lat[100]-lat[200]) > 0:
        ar='right'
      else:
        ar='left'    
        
    # Save the latitude data and the bottom track velocity data for each ADCP separately 
    if name == 'hd':
      dep_hd = dep; lat_hd =[]; bt_hd =[]; limit_hd = latmax
      for i in range(len(lat[0:latmax])):
        lat_hd.append(lat[i]); bt_hd.append(-bt[i])
    if name == 'tx':
      dep_tx = dep; lat_tx = []; bt_tx = []; limit_tx = latmax
      for i in range(len(lat[0:latmax])):
        lat_tx.append(lat[i]); bt_tx.append(-bt[i]) 
           
    below_bottom=where(abs(plv)>10.,1,0)    
    plvp=where(abs(siv)>error_thresh,-9999.,plv)            # don't plot points with large error velocity
    plvp=where(below_bottom==1,NaN,plvp)
    plv_cleaned=remove_bad_columns(plvp)                    # remove columns with a large number of errors
    plvT=1*plv_cleaned.T

    # Save the velocity data for each ADCP separately
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
          
  # Exclude the data near the harbors
  depth_min = 0; depth_max = 30; lat_min = 52.967;  lat_max = 52.999
  for b in range(len(depax)):
    for c in range(len(lat_hd)):
      if depax[b] < depth_min or depax[b] > depth_max:
        vel_hd[b,:] = nan
        if lat_hd[c] < lat_min or lat_hd[c] > lat_max:
          vel_hd[b,c] = nan
      if lat_hd[c] < lat_min or lat_hd[c] > lat_max:
        vel_hd[b,c] = nan
        bt_hd[c] = nan
   
  for b in range(len(depax)):
    for c in range(len(lat_tx)):
      if depax[b] < depth_min or depax[b] > depth_max:
        vel_tx[b,:] = nan
        if lat_tx[c] < lat_min or lat_tx[c] > lat_max:
          vel_tx[b,c] = nan
      if lat_tx[c] < lat_min or lat_tx[c] > lat_max:
          vel_tx[b,c] = nan
          bt_tx[c] = nan

  # Define the front and the rear ADCP 
  ship_speed_front =[];  ship_speed_back =[]
  if ar == 'right':
    lat_front = lat_hd; vel_front = vel_hd; limit_front = limit_hd; dep_front = dep_hd;
    lat_back = lat_tx; vel_back = vel_tx; limit_back = limit_tx; dep_back = dep_tx
    for k in range(len(bt_hd)):
      ship_speed_front.append(bt_hd[k] - environmental_flow_hd[k])
    for k in range(len(bt_tx)):
      ship_speed_back.append(bt_tx[k] - environmental_flow_tx[k])
  if ar=='left':
    lat_front = lat_tx; vel_front = vel_tx; limit_front = limit_tx; dep_front = dep_tx;
    lat_back = lat_hd; vel_back = vel_hd; limit_back = limit_hd; dep_back = dep_hd
    for k in range(len(bt_hd)):
      ship_speed_back.append(bt_hd[k] - environmental_flow_hd[k])
    for k in range(len(bt_tx)):
      ship_speed_front.append(bt_tx[k] - environmental_flow_tx[k])
    
  print(yy+mm+dd+'_'+str(instance)) 
  
  # ---------------------------------------------------------------------------------------------------
  # ---------------------------  Fit and smooth the data -----------------------------------------------
  # ----------------------------------------------------------------------------------------------------
  
  # ----------------------------- Front ADCP  ---------------------------------------- 
  target_coefficients = (0.1,0.1,0.1,0.1)  # guess the unknown coefficients 
  x = abs(array(ship_speed_front)); y = -depax
  im_x, im_y = np.meshgrid(x, y)
  xdata = np.c_[im_x.flatten(), im_y.flatten()]
  vel_ADCP_front = vel_front.flatten() # velocity measurements inlcuding nan values 
  vel_ADCP2_front = vel_ADCP_front[~np.isnan(vel_ADCP_front)] # velocity measurements excluding nan values 
  xdata2 = xdata[~np.isnan(vel_ADCP_front),:] # depth and ship speed velocity 
  ydata2=vel_ADCP2_front # velocity measurements by ADCP
  popt_front, pcov_front = curve_fit(poly2d_fun_front, xdata=xdata2, ydata=ydata2, p0=target_coefficients )
  fit_vel_front = poly2d_fun_front(xdata2, *popt_front)#.reshape(len(ydata2), len(xdata2))
  print('Front ADCP')
  print('w = ({:1.5f}'.format(popt_front[0])+'Vs + {:1.5f}'.format(popt_front[1])+')z + ({:1.5f}'.format(popt_front[2])+'Vs + {:1.5f}'.format(popt_front[3])+')')


  # Add the nan values in their previous indices
  nan_mask = isnan(vel_ADCP_front)
  fit_vel_new_front = empty(vel_ADCP_front.shape)
  fit_vel_new_front[~nan_mask] = fit_vel_front
  fit_vel_new_front[nan_mask] = nan
  fit_vel_new_front = fit_vel_new_front.reshape(len(y), len(x))  
  
  # computation of corrected vertical velocity 
  corrected_front = zeros([len(depax),len(lat_front)]) 
  for i in range(len(depax)):
    for j in range(len(lat_front)):
      corrected_front[i,j] = vel_front[i,j] -  fit_vel_new_front[i,j]
      
  # Substitute nan values with zero and compute the smoothed data 
  vel = corrected_front.flatten()
  vel[np.isnan(vel)] = 0
  corrected_front2 = vel.reshape(len(depax), len(ship_speed_front))  
  smooth_front = ndimage.uniform_filter(corrected_front2, size=5)

  
  # Find the max depth for each latitute and compute the smoothed topography
  max_depth_front=[]
  for l in range(len(lat_front)):
    listt=[]
    for d in range(len(depax)):
      if dep_front[l,d] > 0:
        listt.append(dep_front[l,d])
    max_depth_front.append(max(listt))
  max_depth_front = ndimage.uniform_filter(max_depth_front, size=5)
  
  # Restore nan values for the harbours and for depths larger than the larger depths of the topography
  for b in range(len(depax)):
    for c in range(len(lat_front)):
      if depax[b] < depth_min:
        smooth_front[b,:] = nan
      if depax[b] > max_depth_front[c]:
        smooth_front[b,c] = nan
        if lat_front[c] < lat_min or lat_front[c] > lat_max:
          smooth_front[b,c] = nan
      if lat_front[c] < lat_min or lat_front[c] > lat_max:
        smooth_front[b,c] = nan
  
  
  #------------------------ Back ADCP-------------------------------------------
   
  target_coefficients = (0.1,0.1,0.1,0.1)  # guess the unknown coefficients 
  x = abs(array(ship_speed_back)); y = -depax
  im_x, im_y = np.meshgrid(x, y)
  xdata = np.c_[im_x.flatten(), im_y.flatten()]
  vel_ADCP_back = vel_back.flatten() # velocity measurements inlcuding nan values 
  vel_ADCP2_back = vel_ADCP_back[~np.isnan(vel_ADCP_back)] # velocity measurements excluding nan values 
  xdata2 = xdata[~np.isnan(vel_ADCP_back),:] # depth and ship speed velocity 
  ydata2=vel_ADCP2_back # velocity measurements by ADCP
  popt_back, pcov_back = curve_fit(poly2d_fun_back, xdata=xdata2, ydata=ydata2, p0=target_coefficients )
  fit_vel_back = poly2d_fun_back(xdata2, *popt_back)#.reshape(len(ydata2), len(xdata2))
  print('Rear ADCP')
  print('w = ({:1.5f}'.format(popt_back[0])+'Vs + {:1.5f}'.format(popt_back[1])+')z + ({:1.5f}'.format(popt_back[2])+'Vs + {:1.5f}'.format(popt_back[3])+')')

  # Add the nan values in their previous indices
  nan_mask = isnan(vel_ADCP_back)
  fit_vel_new_back = empty(vel_ADCP_back.shape)
  fit_vel_new_back[~nan_mask] = fit_vel_back
  fit_vel_new_back[nan_mask] = nan
  fit_vel_new_back = fit_vel_new_back.reshape(len(y), len(x))  

  # computation of corrected vertical velocity 
  corrected_back = zeros([len(depax),len(lat_back)]) 
  for i in range(len(depax)):
    for j in range(len(lat_back)):
      corrected_back[i,j] = vel_back[i,j] -  fit_vel_new_back[i,j]
      
  # Substitute nan values with zero and compute the smoothed data
  vel = corrected_back.flatten()
  vel[np.isnan(vel)] = 0
  corrected_back2 = vel.reshape(len(depax), len(ship_speed_back))  
  smooth_back = ndimage.uniform_filter(corrected_back2, size=5)
  
  # Find the max depth for each latitute and compute the smoothed topography
  max_depth_back=[]
  for l in range(len(lat_back)):
    listt=[]
    for d in range(len(depax)):
      if dep_back[l,d] > 0:
        listt.append(dep_back[l,d])
    max_depth_back.append(max(listt))
  max_depth_back = ndimage.uniform_filter(max_depth_back, size=5)
  
  # Restore nan values for the harbours and for depths larger than the larger depths of the topography
  for b in range(len(depax)):
    for c in range(len(lat_back)):
      if depax[b] < depth_min or depax[b] > depth_max:
        smooth_back[b,:] = nan
      if depax[b] > max_depth_back[c]:
        smooth_back[b,c] = nan
        if lat_back[c] < lat_min or lat_back[c] > lat_max:
          smooth_back[b,c] = nan
      if lat_back[c] < lat_min or lat_back[c] > lat_max:
        smooth_back[b,c] = nan
   
  # --------------------------------------------------------------------------     
  #--------------------  PLOTS ------------------------------------------------
  # --------------------------------------------------------------------------
  
  vmax2 = 0.2
  nrows = 2; ncols=2
  figure(fignr,(fs1,fs2+2),None,facecolor='w',edgecolor='k')
  nsub = 0 

  #  ------------- Plot the corrected vertical velocity (front ADCP) -----------------------------------
  subplot(nrows,ncols,nsub+1)
  pcolormesh(array(lat_front),-depax,corrected_front, shading='auto',vmin=-vmax2,vmax=vmax2,cmap='coolwarm')
  ylim([-30, -5])
  gca().invert_xaxis()
  colorbar()
  ylabel(dep_longname+' ['+dep_units+']',fontsize=fs)
  xlabel(lat_longname+' ['+lat_units+']',fontsize=fs)
  title('Front ADCP: Corrected vertical velocity [m/s]',fontsize=fs)
  plt.text(52.971,-29.5,'Den Helder')
  plt.text(53.002,-29.5,'Texel')
  
  if ar == 'right':
    arrow(52.985, -29.5, -0.005, 0.0,fc='k',ec='k',length_includes_head=True, head_width=0.5, head_length=0.001)
  elif ar == 'left':
    arrow(52.980, -29.5, 0.005, 0.0,fc='k',ec='k',length_includes_head=True, head_width=0.5, head_length=0.001)

  
  #  ------------- Plot the smoothed vertical velocity (front ADCP) -----------------------------------
  subplot(nrows,ncols,nsub+2)
  pcolormesh(array(lat_front),-depax,smooth_front, shading='auto',vmin=-vmax2,vmax=vmax2,cmap='coolwarm')
  ylim([-30, -5])
  gca().invert_xaxis()
  colorbar()
  ylabel(dep_longname+' ['+dep_units+']',fontsize=fs)
  xlabel(lat_longname+' ['+lat_units+']',fontsize=fs)
  title('Front ADCP: Corrected and smoothed vertical velocity [m/s]',fontsize=fs)
  plt.text(52.971,-29.5,'Den Helder')
  plt.text(53.002,-29.5,'Texel')
  
  if ar == 'right':
    arrow(52.985, -29.5, -0.005, 0.0,fc='k',ec='k',length_includes_head=True, head_width=0.5, head_length=0.001)
  elif ar == 'left':
    arrow(52.980, -29.5, 0.005, 0.0,fc='k',ec='k',length_includes_head=True, head_width=0.5, head_length=0.001)
    
  
  #  ------------- Plot the corrected vertical velocity (rear ADCP) -----------------------------------
  subplot(nrows,ncols,nsub+3)
  pcolormesh(array(lat_back),-depax,corrected_back, shading='auto',vmin=-vmax2,vmax=vmax2,cmap='coolwarm')
  ylim([-30, -5])
  gca().invert_xaxis()
  colorbar()
  ylabel(dep_longname+' ['+dep_units+']',fontsize=fs)
  xlabel(lat_longname+' ['+lat_units+']',fontsize=fs)
  title('Rear ADCP: Corrected vertical velocity [m/s]',fontsize=fs)
  plt.text(52.971,-29.5,'Den Helder')
  plt.text(53.002,-29.5,'Texel')
  
  if ar == 'right':
    arrow(52.985, -29.5, -0.005, 0.0,fc='k',ec='k',length_includes_head=True, head_width=0.5, head_length=0.001)
  elif ar == 'left':
    arrow(52.980, -29.5, 0.005, 0.0,fc='k',ec='k',length_includes_head=True, head_width=0.5, head_length=0.001)
    
  #  ------------- Plot the smoothed vertical velocity (rear ADCP) -----------------------------------
  subplot(nrows,ncols,nsub+4)
  pcolormesh(array(lat_back),-depax,smooth_back, shading='auto',vmin=-vmax2,vmax=vmax2,cmap='coolwarm')
  ylim([-30, -5])
  gca().invert_xaxis()
  colorbar()
  ylabel(dep_longname+' ['+dep_units+']',fontsize=fs)
  xlabel(lat_longname+' ['+lat_units+']',fontsize=fs)
  title('Rear ADCP: Corrected and smoothed vertical velocity [m/s]',fontsize=fs)
  plt.text(52.971,-29.5,'Den Helder')
  plt.text(53.002,-29.5,'Texel')
  
  if ar == 'right':
    arrow(52.985, -29.5, -0.005, 0.0,fc='k',ec='k',length_includes_head=True, head_width=0.5, head_length=0.001)
  elif ar == 'left':
    arrow(52.980, -29.5, 0.005, 0.0,fc='k',ec='k',length_includes_head=True, head_width=0.5, head_length=0.001)
  

  if savefigg ==1:
    figfilename=varname+'_'+str(yy)+str(mm)+str(dd)+'_'+str(instance)+'_CORRECTION_FRONT_BACK_SMOOTH'.zfill(2)+figtype
    savefig(figdir+'/'+figfilename,dpi=300)
    close(fignr)
    
  return fignr
  
#-----------------------------------------------------------------------

def compute_envir_flow(indir,names,yy,mm,dd,error_thresh, instance):
  # estimated environmnetal northward flow by using the measured eastward velocity 
  
  environmental_flow_hd = [];environmental_flow_tx = []

  varname1 = 'EAST_VEL'

  # loop over instruments
  for name in names:
    #print(name)
    [lat,lat_units,lat_longname]=loadvar(indir,name,yy,mm,dd,'LATITUDE',instance)
    [dep,dep_units,dep_longname]=loadvar(indir,name,yy,mm,dd,'BINDEPTH',instance)
    [siv,siv_units,siv_longname]=loadvar(indir,name,yy,mm,dd,'ERROR_VEL',instance)
    [plv,plv_units,plv_longname]=loadvar(indir,name,yy,mm,dd,varname1,instance)
    [qc,qc_units,qc_longname]=loadvar(indir,name,yy,mm,dd,'General_QC',instance)

    sd=shape(dep)
    depstep=dep[0,1]-dep[0,0]
    depax=arange(dep[0,0],dep[0,0]+sd[1]*depstep,depstep)
      
    latmax=0
    while lat[latmax]>0:
      latmax=latmax+1
      if latmax>len(lat):
        break
             
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

  # Exlude the first 5m below the hull to avoid disturbance due to ferry's motion
  depth_min = 9; depth_max = 30; lat_min = 52.9;  lat_max = 53.1
  for b in range(len(depax)):
    for c in range(len(lat_hd)):
      if depax[b] < depth_min or depax[b] > depth_max:
        vel_hd[b,:] = nan
        if lat_hd[c] < lat_min or lat_hd[c] > lat_max:
          vel_hd[b,c] = nan
      if lat_hd[c] < lat_min or lat_hd[c] > lat_max:
        vel_hd[b,c] = nan
  for b in range(len(depax)):
    for c in range(len(lat_tx)):
      if depax[b] < depth_min or depax[b] > depth_max:
        vel_tx[b,:] = nan
        if lat_tx[c] < lat_min or lat_tx[c] > lat_max:
          vel_tx[b,c] = nan
      if lat_tx[c] < lat_min or lat_tx[c] > lat_max:
          vel_tx[b,c] = nan
          

 # We consider the angle that the tidal currents flow out and into of the Wadden sea (reference: Buijsman PhD) 
  for j in range(len(lat_hd)):
    tot_vel_hd =[]
    if lat_hd[j] <= 52.970:
      angle = 10
    if lat_hd[j] >= 52.970 and lat_hd[j] <= 52.980:
      angle = 21
    if lat_hd[j] >= 52.980 and lat_hd[j] <= 52.995:
      angle = 32
    if lat_hd[j] >= 52.995:
      angle = 27
    for i in range(len(depax)):
      bo_hd = ~isnan(vel_hd[i,j])
      if  bo_hd == True: 
        tot_vel_hd.append(vel_hd[i,j])  # vertically averaged velocity for each water column 
    environmental_flow_hd.append(mean(tot_vel_hd)*sin(angle*pi/180)/cos(angle*pi/180))
  
  for j in range(len(lat_tx)):
    tot_vel_tx =[]
    if lat_tx[j] <= 52.970:
      angle = 10
    if lat_tx[j] >= 52.970 and lat_tx[j] <= 52.980:
      angle = 21
    if lat_tx[j] >= 52.980 and lat_tx[j] <= 52.995:
      angle = 32
    if lat_tx[j] >= 52.995:
      angle = 27
    for i in range(len(depax)):
      bo_tx = ~isnan(vel_tx[i,j])
      if  bo_tx == True: 
        tot_vel_tx.append(vel_tx[i,j])  # vertically averaged velocity for each water column 
    environmental_flow_tx.append(mean(tot_vel_tx)*sin(angle*pi/180)/cos(angle*pi/180))
  
  tight_layout()
  
  return  environmental_flow_hd, environmental_flow_tx
  
#-----------------------------------------------------------------------

############ Settings ######################################################

# The names of the ADCPs
names=['hd','tx']

# Choose date, error threshold and crossing
yy='2021'; mm = '07';  dd = '15'; error_thresh = 0.35; instance = 5
#yy='2021'; mm = '07';  dd = '15'; error_thresh = 0.41; instance = 28
#yy='2022'; mm = '04';  dd = '16'; error_thresh = 0.20; instance = 21 # Presence of internal waves 
#yy='2022'; mm = '06';  dd = '01'; error_thresh = 0.29; instance = 19 # Presence of fronts and boils
instance_step=1

# Velocity
varname = 'VERT_VEL'; bt_vars = 'BT_NORTH_VEL'

# Bad_fraction
bad_fraction = 0.3      # don't plot columns with a fraction of points with error velocity over error_threshold larger than bad_fraction

# Location of the data
if yy =='2022' and mm == '06' and dd == '01':
  indir = '/home/prdusr/data_out/TESO/daily'
else: 
  indir='/home/jvandermolen/data_out/TESO/daily'

 
# max value for the colorbar 
vmax=0.2

fs1 = 16; fs2 = 9; fs = 12 
dpi_setting=300
figtype='.jpg'
figdir='/home/akaraoli/python_scripts/figures/'
figlabel=['a','b','c','d','e','f','g','h','i','j']
savefigg = 0


############ Main ######################################################

fignr = 2
fignr = correction_vertical_velocity_for_ferry_veloicty(indir, yy, mm, dd, error_thresh, instance, fignr) # 2nd correction

fignr = fignr + 1
fignr = correction_vertical_velocity_smooth(indir, yy, mm, dd, error_thresh, instance, fignr) # 2nd correction smoothed

show()  