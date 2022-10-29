#! /usr/bin/env python

# python script to plot TESO transects

# Author: Athina Karaoli, July 2022

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


#---------------------------------------------------------------------------------------
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

def vertical_vel_VS_bottom_gradient(indir,yy,mm,dd,instances, error_thresh, bad_fraction, varname, bt_vars):

  for instance in instances:
    #Estimated environmental flow 
    environmental_flow_hd, environmental_flow_tx = compute_envir_flow(indir,names,yy,mm,dd,error_thresh, instance)
    
    for name in names:
      #print(name)
      [lat,lat_units,lat_longname]=loadvar(indir,name,yy,mm,dd,'LATITUDE',instance)
      [dep,dep_units,dep_longname]=loadvar(indir,name,yy,mm,dd,'BINDEPTH',instance)
      [siv,siv_units,siv_longname]=loadvar(indir,name,yy,mm,dd,'ERROR_VEL',instance)
      [plv,plv_units,plv_longname]=loadvar(indir,name,yy,mm,dd,varname,instance)
      [bt,bt_units,bt_longname]=loadvar(indir,name,yy,mm,dd,bt_vars,instance)
      [beam1,beam1_units,beam1_longname]=loadvar(indir,name,yy,mm,dd,'D1',instance)
      [beam2,beam2_units,beam2_longname]=loadvar(indir,name,yy,mm,dd,'D2',instance)
      [beam3,beam3_units,beam3_longname]=loadvar(indir,name,yy,mm,dd,'D3',instance)
      [beam4,beam4_units,beam4_longname]=loadvar(indir,name,yy,mm,dd,'D4',instance)
      
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
      
      if name == 'hd':
        lat_hd =[]; bt_hd =[]; dep_hd = dep
        for i in range(len(lat[0:latmax])):
          lat_hd.append(lat[i]); bt_hd.append(bt[i])
      if name == 'tx':
        lat_tx = []; bt_tx = []; dep_tx = dep
        for i in range(len(lat[0:latmax])):
          lat_tx.append(lat[i]); bt_tx.append(bt[i]) 
      
          
      below_bottom=where(abs(plv)>10.,1,0)    
      plvp=where(abs(siv)>error_thresh,-9999.,plv)            # don't plot points with large error velocity
      plvp=where(below_bottom==1,NaN,plvp)
      plv_cleaned=remove_bad_columns(plvp)                    # remove columns with a large number of errors
      plvT=1*plv_cleaned.T
      levs=arange(-1.3,1.4,0.05)
      
      if name == 'hd':
        vel_hd = zeros([len(depax),latmax]); beam_hd = zeros(latmax)
        limit_hd = latmax
        for i in range(len(depax)):
          for j in range(latmax):
            vel_hd[i,j] = plvT[i,j]
        # Compute the averaged topography for each ADCP
        for j in range(latmax):    
          beam_hd[j] = (beam1[j] + beam2[j] + beam3[j] + beam4[j])/4
      if name == 'tx':
        vel_tx = zeros([len(depax),latmax])
        beam_tx = zeros(latmax)
        limit_tx = latmax
        for i in range(len(depax)):
          for j in range(latmax):
            vel_tx[i,j] = plvT[i,j]
        # Compute the averaged topography for each ADCP
        for j in range(latmax):    
          beam_tx[j] = (beam1[j] + beam2[j] + beam3[j] + beam4[j])/4
            
    # Exclude the data near the harbours  
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
   
   
    mean_depth_hd = zeros(len(depax)); mean_depth_tx = zeros(len(depax));
    #Compute the mean value of the velocities fro each depth layer
    for b in range(len(depax)):
      tot_vel_hd = []
      for c in range(len(lat_hd)):
        bo_hd = ~isnan(vel_hd[b,c] )
        if  bo_hd == True: 
          tot_vel_hd.append(vel_hd[b,c] )
      mean_depth_hd[b] = mean(tot_vel_hd)
    for b in range(len(depax)):
      tot_vel_tx = []
      for c in range(len(lat_tx)):
        bo_tx = ~isnan(vel_tx[b,c])
        if  bo_tx == True: 
          tot_vel_tx.append(vel_tx[b,c])
      mean_depth_tx[b] = mean(tot_vel_tx)
  
    # Dinstinguish the two ADCPs in front and rear ADCP
    ship_speed_front = []; ship_speed_back = []
    if ar == 'right':
      lat_front =  lat_hd;lat_back =   lat_tx
      dep_front = dep_hd;dep_back = dep_tx
      vel_front = vel_hd; vel_back = vel_tx
      mean_depth_front = mean_depth_hd; mean_depth_back = mean_depth_tx
      beam_front = beam_hd; beam_back = beam_tx
      limit_front = limit_hd; limit_back = limit_tx
      for k in range(len(bt_hd)):
        ship_speed_front.append(bt_hd[k] - environmental_flow_hd[k])
      for k in range(len(bt_tx)):
        ship_speed_back.append(bt_tx[k] - environmental_flow_tx[k])
    if ar == 'left':
      lat_front =  lat_tx; lat_back =   lat_hd
      dep_front = dep_tx; dep_back = dep_hd
      vel_front = vel_tx; vel_back = vel_hd
      mean_depth_front = mean_depth_tx; mean_depth_back = mean_depth_hd
      beam_front = beam_tx; beam_back = beam_hd
      limit_front = limit_tx; limit_back = limit_hd
      for k in range(len(bt_hd)):
        ship_speed_back.append(bt_hd[k] - environmental_flow_hd[k])
      for k in range(len(bt_tx)):
        ship_speed_front.append(bt_tx[k] - environmental_flow_tx[k])
      
    # Subtract the horizontally averaged vertical velocity from the measured vertical velocity 
    # 1st step correction
    final_vel_front = zeros([len(depax), len(lat_front)]); final_vel_back = zeros([len(depax),len(lat_back)]); 
    for b in range(len(depax)):
      for c in range(len(lat_front)):
        final_vel_front[b,c] =  vel_front[b,c]  - mean_depth_front[b]
      for c in range(len(lat_back)):
        final_vel_back[b,c] =  vel_back[b,c]  - mean_depth_back[b] 
      
    # -------------------------- Fit the data by using ferry's motion function -----------------------------------------------
    
    # Front ADCP 
    target_coefficients = (0.1,0.1,0.1,0.1) # guess the unknown coefficients 
    x = abs(array(ship_speed_front)); y = -depax
    im_x, im_y = np.meshgrid(x, y)
    xdata = np.c_[im_x.flatten(), im_y.flatten()]
    vel_ADCP_front = final_vel_front.flatten() # 1st step corrected velocity measurements inlcuding nan values 
    vel_ADCP2_front = vel_ADCP_front[~np.isnan(vel_ADCP_front)] # 1st step corrected velocity measurements excluding nan values 
    xdata2 = xdata[~np.isnan(vel_ADCP_front),:] # depth and ship speed velocity 
    ydata2 = vel_ADCP2_front # 1st step corrected velocity measurements excluding nan values
    popt_front, pcov_front = curve_fit(poly2d_fun_front, xdata=xdata2, ydata=ydata2, p0=target_coefficients )
    fit_vel_front = poly2d_fun_front(xdata2, *popt_front) #.reshape(len(ydata2), len(xdata2))

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
        corrected_front[i,j] = final_vel_front[i,j] -  fit_vel_new_front[i,j]
    
    
    # Back ADCP    
    target_coefficients = (0.1,0.1,0.1,0.1)  # guess the unknown coefficients 
    x = abs(array(ship_speed_back)); y = -depax
    im_x, im_y = np.meshgrid(x, y)
    xdata = np.c_[im_x.flatten(), im_y.flatten()]
    vel_ADCP_back = final_vel_back.flatten() # 1st step corrected velocity measurements inlcuding nan values 
    vel_ADCP2_back = vel_ADCP_back[~np.isnan(vel_ADCP_back)] # 1st step corrected velocity measurements excluding nan values 
    xdata2 = xdata[~np.isnan(vel_ADCP_back),:] # depth and ship speed velocity 
    ydata2=vel_ADCP2_back # 1st step corrected velocity measurements excluding nan values
    popt_back, pcov_back = curve_fit(poly2d_fun_back, xdata=xdata2, ydata=ydata2, p0=target_coefficients )
    fit_vel_back = poly2d_fun_back(xdata2, *popt_back)#.reshape(len(ydata2), len(xdata2))

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
        corrected_back[i,j] = final_vel_back[i,j] -  fit_vel_new_back[i,j]
  
 
    # Max depth measured by the front ADCP (front ADCP is more reliable then the rear ADCP)
    bathymetry = -beam_front[0:limit_front]

    # Find the vertical velocity at the max depth for each lat
    max_depth_front=[];  max_depth_front_vel =[]
    for k in range(len(lat_front)):
      listt=[]
      for d in range(len(depax)):
        if dep_front[k,d] > 0:
          listt.append(dep_front[k,d])
      max_depth_front.append(max(listt))
      max_depth_front_vel.append(corrected_front[where(dep_front[k,:] == max(listt))[0][0],k])
      
    # Estimate the bottom gradient by using two grid points each time 
    estim_slope =[]
    for i in range(limit_front-1):
      if lat_front[i]==lat_front[i+1]: # remove the case where the bottom is flat between two grid points 
        estim_slope.append(nan)
      elif ar == 'right'  and lat_front[i] < lat_front[i+1]: # remove the cases where the ferry moves a little backwards
        estim_slope.append(nan)
      elif ar == 'left'  and lat_front[i] > lat_front[i+1]: # remove the cases where the ferry moves a little backwards
        estim_slope.append(nan)
      elif (bathymetry[i] - bathymetry[i+1])/((lat_front[i]-lat_front[i+1])*111000) > 0.5  or (bathymetry[i] - bathymetry[i+1])/((lat_front[i]-lat_front[i+1])*111000) < -0.5:  # 1lat = 111000m
        estim_slope.append(nan) # remove the cases when the bottom gradient is larger than 0.5
      else:
        estim_slope.append((bathymetry[i] - bathymetry[i+1])/((lat_front[i]-lat_front[i+1])*111000))
        
    vel = np.array(max_depth_front_vel[0:limit_front -1])
    estim_slope = np.array(estim_slope)
    
    # Remove nan values to compute the correlation 
    vel_none_nan = vel[~isnan(vel)]
    slope_none_nan = estim_slope[~isnan(vel)]
    
    slope_none_nan2 = slope_none_nan[~isnan(slope_none_nan)]
    vel_none_nan2 = vel_none_nan[~isnan(slope_none_nan)]
  
    corr_coef = corrcoef(slope_none_nan2,vel_none_nan2)

    # Fitting 
    Polynomial_Regression = poly1d(polyfit(slope_none_nan2 ,vel_none_nan2,1))
    line = linspace(min(slope_none_nan2 ), max(slope_none_nan2 ), 500)
    
    N = len(slope_none_nan2 )
    x = array(slope_none_nan2 )
    sigmay = np.sqrt((1/(N-2))*np.sum((array(vel_none_nan2) - Polynomial_Regression[0] - Polynomial_Regression[1]*x)**2))
    std_slope = np.sqrt(sigmay**2/(N*(np.mean(x**2)-np.mean(x)**2)))
    std_interc = np.sqrt(sigmay**2*np.mean(x**2)/(N*(np.mean(x**2)-np.mean(x)**2)))
    
    
    # Plots
    fignr=int(instance)
    figure(fignr,(7,5),None,facecolor='w',edgecolor='k')
    scatter(estim_slope,vel, color = 'royalblue', s = 10)
    plot(line, Polynomial_Regression(line),'r',linewidth = 2,label='y = ({:1.3f} '.format(Polynomial_Regression[1]) + ' $\pm$ {:1.3f})'.format(std_slope) + 'x + ({:1.3f}'.format(Polynomial_Regression[0]) + ' $\pm$ {:1.3f}'.format(std_interc) + ')\ncorrelation coefficient: ' +str(round(corr_coef[0][1],3)))
    ylim([-0.7,0.7])
    xlabel('bottom gradient [dimensionless]',fontsize=fs)
    ylabel('Vertical velocity at the seabed [m/s]',fontsize=fs)
    if ar == 'right':
      arrow = '$\longrightarrow$'
    if ar == 'left':
      arrow = '$\longleftarrow$'
    title('Transect '+str(instance) + ' (tx'+arrow+'hd)')
    grid()
    legend()
    
    if savefigg == 1:
      figfilename=str(yy)+str(mm)+str(dd)+'_'+str(instance)+'_correlation_vert_vel_bottom_gradient_2nd_way_zoom_out_plus_line_05'.zfill(2)+figtype
      savefig(figdir+'/'+figfilename,dpi=300)
      close(fignr)
  
 
      tight_layout()
  
  return 
  
# ----------------------------------------------------------------------------------------------
def vert_vel_VS_bottom_gradient_without_plots(indir,yy,mm,dd,instance,error_thresh,varname):

  environmental_flow_hd, environmental_flow_tx = compute_envir_flow(indir,names,yy,mm,dd,error_thresh, instance)
  
  for name in names:
    #print(name)
    [lat,lat_units,lat_longname]=loadvar(indir,name,yy,mm,dd,'LATITUDE',instance)
    [dep,dep_units,dep_longname]=loadvar(indir,name,yy,mm,dd,'BINDEPTH',instance)
    [siv,siv_units,siv_longname]=loadvar(indir,name,yy,mm,dd,'ERROR_VEL',instance)
    [plv,plv_units,plv_longname]=loadvar(indir,name,yy,mm,dd,varname,instance)
    [bt,bt_units,bt_longname]=loadvar(indir,name,yy,mm,dd,bt_vars,instance)
    [beam1,beam1_units,beam1_longname]=loadvar(indir,name,yy,mm,dd,'D1',instance)
    [beam2,beam2_units,beam2_longname]=loadvar(indir,name,yy,mm,dd,'D2',instance)
    [beam3,beam3_units,beam3_longname]=loadvar(indir,name,yy,mm,dd,'D3',instance)
    [beam4,beam4_units,beam4_longname]=loadvar(indir,name,yy,mm,dd,'D4',instance)
    
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
    
    if name == 'hd':
      lat_hd =[]; bt_hd =[]; dep_hd = dep
      for i in range(len(lat[0:latmax])):
        lat_hd.append(lat[i]); bt_hd.append(bt[i])
    if name == 'tx':
      lat_tx = []; bt_tx = []; dep_tx = dep
      for i in range(len(lat[0:latmax])):
        lat_tx.append(lat[i]); bt_tx.append(bt[i]) 
    
        
    below_bottom=where(abs(plv)>10.,1,0)    
    plvp=where(abs(siv)>error_thresh,-9999.,plv)            # don't plot points with large error velocity
    plvp=where(below_bottom==1,NaN,plvp)
    plv_cleaned=remove_bad_columns(plvp)                    # remove columns with a large number of errors
    plvT=1*plv_cleaned.T
    levs=arange(-1.3,1.4,0.05)
    
    if name == 'hd':
      vel_hd = zeros([len(depax),latmax]); beam_hd = zeros(latmax)
      limit_hd = latmax
      for i in range(len(depax)):
        for j in range(latmax):
          vel_hd[i,j] = plvT[i,j]
      # Compute the averaged topography for each ADCP
      for j in range(latmax):    
        beam_hd[j] = (beam1[j] + beam2[j] + beam3[j] + beam4[j])/4
    if name == 'tx':
      vel_tx = zeros([len(depax),latmax])
      beam_tx = zeros(latmax)
      limit_tx = latmax
      for i in range(len(depax)):
        for j in range(latmax):
          vel_tx[i,j] = plvT[i,j]
      # Compute the averaged topography for each ADCP
      for j in range(latmax):    
        beam_tx[j] = (beam1[j] + beam2[j] + beam3[j] + beam4[j])/4
          
  # Exclude the data near the harbours  
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
 
 
  mean_depth_hd = zeros(len(depax)); mean_depth_tx = zeros(len(depax));
  #Compute the mean value of the velocities fro each depth layer
  for b in range(len(depax)):
    tot_vel_hd = []
    for c in range(len(lat_hd)):
      bo_hd = ~isnan(vel_hd[b,c] )
      if  bo_hd == True: 
        tot_vel_hd.append(vel_hd[b,c] )
    mean_depth_hd[b] = mean(tot_vel_hd)
  for b in range(len(depax)):
    tot_vel_tx = []
    for c in range(len(lat_tx)):
      bo_tx = ~isnan(vel_tx[b,c])
      if  bo_tx == True: 
        tot_vel_tx.append(vel_tx[b,c])
    mean_depth_tx[b] = mean(tot_vel_tx)

  # Dinstinguish the two ADCPs in front and rear ADCP
  ship_speed_front = []; ship_speed_back = []
  if ar == 'right':
    lat_front =  lat_hd;lat_back =   lat_tx
    dep_front = dep_hd;dep_back = dep_tx
    vel_front = vel_hd; vel_back = vel_tx
    mean_depth_front = mean_depth_hd; mean_depth_back = mean_depth_tx
    beam_front = beam_hd; beam_back = beam_tx
    limit_front = limit_hd; limit_back = limit_tx
    for k in range(len(bt_hd)):
      ship_speed_front.append(bt_hd[k] - environmental_flow_hd[k])
    for k in range(len(bt_tx)):
      ship_speed_back.append(bt_tx[k] - environmental_flow_tx[k])
  if ar == 'left':
    lat_front =  lat_tx; lat_back =   lat_hd
    dep_front = dep_tx; dep_back = dep_hd
    vel_front = vel_tx; vel_back = vel_hd
    mean_depth_front = mean_depth_tx; mean_depth_back = mean_depth_hd
    beam_front = beam_tx; beam_back = beam_hd
    limit_front = limit_tx; limit_back = limit_hd
    for k in range(len(bt_hd)):
      ship_speed_back.append(bt_hd[k] - environmental_flow_hd[k])
    for k in range(len(bt_tx)):
      ship_speed_front.append(bt_tx[k] - environmental_flow_tx[k])
    
  # Subtract the horizontally averaged vertical velocity from the measured vertical velocity 
  # 1st step correction
  final_vel_front = zeros([len(depax), len(lat_front)]); final_vel_back = zeros([len(depax),len(lat_back)]); 
  for b in range(len(depax)):
    for c in range(len(lat_front)):
      final_vel_front[b,c] =  vel_front[b,c]  - mean_depth_front[b]
    for c in range(len(lat_back)):
      final_vel_back[b,c] =  vel_back[b,c]  - mean_depth_back[b] 
    
  # -------------------------- Fit the data by using ferry's motion function -----------------------------------------------
  
  # Front ADCP 
  target_coefficients = (0.1,0.1,0.1,0.1) # guess the unknown coefficients 
  x = abs(array(ship_speed_front)); y = -depax
  im_x, im_y = np.meshgrid(x, y)
  xdata = np.c_[im_x.flatten(), im_y.flatten()]
  vel_ADCP_front = final_vel_front.flatten() # 1st step corrected velocity measurements inlcuding nan values 
  vel_ADCP2_front = vel_ADCP_front[~np.isnan(vel_ADCP_front)] # 1st step corrected velocity measurements excluding nan values 
  xdata2 = xdata[~np.isnan(vel_ADCP_front),:] # depth and ship speed velocity 
  ydata2 = vel_ADCP2_front # 1st step corrected velocity measurements excluding nan values
  popt_front, pcov_front = curve_fit(poly2d_fun_front, xdata=xdata2, ydata=ydata2, p0=target_coefficients )
  fit_vel_front = poly2d_fun_front(xdata2, *popt_front) #.reshape(len(ydata2), len(xdata2))

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
      corrected_front[i,j] = final_vel_front[i,j] -  fit_vel_new_front[i,j]
  
  
  # Back ADCP    
  target_coefficients = (0.1,0.1,0.1,0.1)  # guess the unknown coefficients 
  x = abs(array(ship_speed_back)); y = -depax
  im_x, im_y = np.meshgrid(x, y)
  xdata = np.c_[im_x.flatten(), im_y.flatten()]
  vel_ADCP_back = final_vel_back.flatten() # 1st step corrected velocity measurements inlcuding nan values 
  vel_ADCP2_back = vel_ADCP_back[~np.isnan(vel_ADCP_back)] # 1st step corrected velocity measurements excluding nan values 
  xdata2 = xdata[~np.isnan(vel_ADCP_back),:] # depth and ship speed velocity 
  ydata2=vel_ADCP2_back # 1st step corrected velocity measurements excluding nan values
  popt_back, pcov_back = curve_fit(poly2d_fun_back, xdata=xdata2, ydata=ydata2, p0=target_coefficients )
  fit_vel_back = poly2d_fun_back(xdata2, *popt_back)#.reshape(len(ydata2), len(xdata2))

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
      corrected_back[i,j] = final_vel_back[i,j] -  fit_vel_new_back[i,j]


  # Max depth measured by the front ADCP (front ADCP is more reliable then the rear ADCP)
  bathymetry = -beam_front[0:limit_front]

  # Find the vertical velocity at the max depth for each lat
  max_depth_front=[];  max_depth_front_vel =[]
  for k in range(len(lat_front)):
    listt=[]
    for d in range(len(depax)):
      if dep_front[k,d] > 0:
        listt.append(dep_front[k,d])
    if not listt:
        max_depth_front_vel.append(nan); max_depth_front.append(nan)
    else:
      max_depth_front.append(max(listt))
      max_depth_front_vel.append(corrected_front[where(dep_front[k,:] == max(listt))[0][0],k])  # vertical velocity exactly at the bottom
    
  # Estimate the bottom gradient by using two grid points each time 
  estim_slope =[]
  for i in range(limit_front-1):
    if lat_front[i]==lat_front[i+1]: # remove the case where the bottom is flat between two grid points 
      estim_slope.append(nan)
    elif ar == 'right'  and lat_front[i] < lat_front[i+1]: # remove the cases where the ferry moves a little backwards
      estim_slope.append(nan)
    elif ar == 'left'  and lat_front[i] > lat_front[i+1]: # remove the cases where the ferry moves a little backwards
      estim_slope.append(nan)
    elif (bathymetry[i] - bathymetry[i+1])/((lat_front[i]-lat_front[i+1])*111000) > 0.5  or (bathymetry[i] - bathymetry[i+1])/((lat_front[i]-lat_front[i+1])*111000) < -0.5:  # 1lat = 111000m
      estim_slope.append(nan) # remove the cases when the bottom gradient is larger than 0.5
    else:
      estim_slope.append((bathymetry[i] - bathymetry[i+1])/((lat_front[i]-lat_front[i+1])*111000))
      
  vel = np.array(max_depth_front_vel[0:limit_front -1])
  estim_slope = np.array(estim_slope)
  
  # Remove nan values to compute the correlation 
  vel_none_nan = vel[~isnan(vel)]
  slope_none_nan = estim_slope[~isnan(vel)]
  
  slope_none_nan2 = slope_none_nan[~isnan(slope_none_nan)]
  vel_none_nan2 = vel_none_nan[~isnan(slope_none_nan)]

  corr_coef = corrcoef(slope_none_nan2,vel_none_nan2)

  # Fitting 
  Polynomial_Regression = poly1d(polyfit(slope_none_nan2 ,vel_none_nan2,1))
  line = linspace(min(slope_none_nan2 ), max(slope_none_nan2 ), 500)
  
  N = len(slope_none_nan2 )
  x = array(slope_none_nan2 )
  sigmay = np.sqrt((1/(N-2))*np.sum((array(vel_none_nan2) - Polynomial_Regression[0] - Polynomial_Regression[1]*x)**2))
  std_slope = np.sqrt(sigmay**2/(N*(np.mean(x**2)-np.mean(x)**2)))
  std_interc = np.sqrt(sigmay**2*np.mean(x**2)/(N*(np.mean(x**2)-np.mean(x)**2)))
  
    
  tight_layout()
  
  return Polynomial_Regression[1], corr_coef[0][1] 


#-----------------------------------------------------------------------

def estim_northward_vel(indir,names,yy,mm,dd,instance, error_thresh): 

  varname = 'NORTH_VEL'
  
  # loop over instruments
  for name in names:
    #print(name)
    [lat,lat_units,lat_longname]=loadvar(indir,name,yy,mm,dd,'LATITUDE',instance)
    [dep,dep_units,dep_longname]=loadvar(indir,name,yy,mm,dd,'BINDEPTH',instance)
    [siv,siv_units,siv_longname]=loadvar(indir,name,yy,mm,dd,'ERROR_VEL',instance)
    [plv,plv_units,plv_longname]=loadvar(indir,name,yy,mm,dd,varname,instance)
    [qc,qc_units,qc_longname]=loadvar(indir,name,yy,mm,dd,'General_QC',instance)

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
    
    below_bottom=where(abs(plv)>10.,1,0)    
    plvp=where(abs(siv)>error_thresh,-9999.,plv)            # don't plot points with large error velocity
    plvp=where(below_bottom==1,NaN,plvp)
    plv_cleaned=remove_bad_columns(plvp)                    # remove columns with a large number of errors
    plvT=1*plv_cleaned.T
    
    # Compute the max_depth for each lat
    max_depth_front=[]; 
    for k in range(latmax):
      listt=[]
      for d in range(len(depax)):
        if dep[k,d] > 0:
          listt.append(dep[k,d])
      if not listt:  # nan values in the whole water column
        max_depth_front.append(nan)
      else:
        max_depth_front.append(max(listt))
    
    # Average of the whole transect excluding harbors
    depth_min = 19; depth_max = 30; lat_min = 52.967;  lat_max = 52.999
    for c in range(latmax):
      for b in range(len(depax)):
        if depax[b] < max_depth_front[c] - 2: # we are going to compute the mean velocity of the last 2m for each water column
          plvT[b,c] = nan
      if lat[c] < lat_min or lat[c] > lat_max:
        plvT[:,c] = nan
 
    #calculate the mean and the std deviation 
    tot_vel =[]
    for i in range(len(depax)):
      for j in range(latmax):
        bo = ~isnan(plvT[i,j])
        if  bo == True: 
          tot_vel.append(plvT[i,j])
       
    mean_val = mean(tot_vel); standev = std(tot_vel)
      
    if name == 'hd':
      v_hd = mean_val; std_hd = standev
    if name ==  'tx':
      v_tx = mean_val; std_tx = standev

  if ar == 'right':
    v_front = v_hd; v_back = v_tx
    std_front = std_hd; std_back = std_tx;
  if ar == 'left':
    v_front = v_tx; v_back = v_hd
    std_front = std_tx; std_back = std_hd;


  return v_front

# --------------------------------------------------------------------------------------------
def estimated_vel_VS_measured_vel():

  north_vel_ar = []; slope_bottom_ar = []; correlation_coef = []
  
  # Date 
  yy='2021'; mm = '07'; error_thresh = 0.22
  # Days
  days = ['12','13','14','15','16','17'] 

  
  # ---------------------------- For all crossings ------------------------------------------ 
  
  for dd in days:
    print(dd)
    for instance in range(16): 
      north_vel = estim_northward_vel(indir,names,yy,mm,dd,instance, error_thresh)
      slope_bottom, corr_coef = vert_vel_VS_bottom_gradient_without_plots(indir,yy,mm,dd,instance,error_thresh,varname)
      north_vel_ar.append(north_vel)
      slope_bottom_ar.append(slope_bottom)
      correlation_coef.append(abs(corr_coef))
    
  # Linear fitting
  Polynomial_Regression = poly1d(polyfit(north_vel_ar,slope_bottom_ar,1))
  line = linspace(min(north_vel_ar), max(north_vel_ar), 500)
  
  # Errors of the slope and the intercept
  N = len(north_vel_ar)
  x = array(north_vel_ar)
  sigmay = np.sqrt((1/(N-2))*np.sum((array(slope_bottom_ar) - Polynomial_Regression[0] - Polynomial_Regression[1]*x)**2))
  std_slope = np.sqrt(sigmay**2/(N*(np.mean(x**2)-np.mean(x)**2)))
  std_interc = np.sqrt(sigmay**2*np.mean(x**2)/(N*(np.mean(x**2)-np.mean(x)**2)))
  
  #Correlation
  corr_coef = corrcoef(north_vel_ar,slope_bottom_ar)
  
  # Plot
  fignr = 40
  fig = figure(fignr,(8,6),None,facecolor='w',edgecolor='k')
  plot(linspace(-0.8,0.8,100), linspace(-0.8,0.8,100),'grey', linewidth = 1, label='y = x')
  cm = plt.cm.get_cmap('autumn') 
  sc = scatter(north_vel_ar,slope_bottom_ar, c = correlation_coef, vmin=0, vmax=0.7, s=20, cmap=cm)
  plot(line, Polynomial_Regression(line),'darkblue',label='y = ({:1.3f} '.format(Polynomial_Regression[1]) + ' $\pm$ {:1.3f})'.format(std_slope) + 'x + ({:1.3f}'.format(Polynomial_Regression[0]) + ' $\pm$ {:1.3f}'.format(std_interc) + ')\ncorrelation coefficient: ' +str(round(corr_coef[0][1],3)))
  xlabel('Mean northward velocity measured by the front ADCP [m/s]')
  ylabel('Computed northward velocity by using the \nrelation topography-vertical velocity [m/s]')
  ylim([-0.8,0.8])
  xlim([-0.8,0.8])
  legend()

  bounds = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7]
  norm = matplotlib.colors.BoundaryNorm(bounds, cm.N, extend='both')
  fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cm),orientation='vertical')
  
  major_ticks_y = np.arange(-0.8, 1, 0.2)
  minor_ticks_y = np.arange(-0.8, 0.85, 0.05)
  major_ticks_x = np.arange(-0.8, 1, 0.2)
  minor_ticks_x = np.arange(-0.8, 0.85, 0.05) 
  ax = plt.gca()
  ax.set_xticks(major_ticks_x)
  ax.set_xticks(minor_ticks_x, minor=True)
  ax.set_yticks(major_ticks_y)
  ax.set_yticks(minor_ticks_y, minor=True)
  ax.grid(which='both')
  ax.grid(which='minor', alpha=0.3)
  ax.grid(which='major', alpha=0.6)
  ax.set_aspect('equal')

  if savefigg == 1:
    figfilename=str(yy)+str(mm)+str(days)+'_correl_topography_northward_vel_VS_bottom_gradient_for_all_crossings'.zfill(2)+figtype
    savefig(figdir+'/'+figfilename,dpi=300)
    close(fignr)
    
  # ------------------ For crossings tx -> hd --------------------------------------------
  
  north_vel_ar = []; slope_bottom_ar = []; correlation_coef = []
  instances = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30] 
  
  for dd in days:
    print(dd)
    for instance in instances: 
      north_vel = estim_northward_vel(indir,names,yy,mm,dd,instance, error_thresh)
      slope_bottom, corr_coef = vert_vel_VS_bottom_gradient_without_plots(indir,yy,mm,dd,instance,error_thresh,varname)
      north_vel_ar.append(north_vel)
      slope_bottom_ar.append(slope_bottom)
      correlation_coef.append(abs(corr_coef))
    
  # Linear fitting
  Polynomial_Regression = poly1d(polyfit(north_vel_ar,slope_bottom_ar,1))
  line = linspace(min(north_vel_ar), max(north_vel_ar), 500)
  
  # Errors of the slope and the intercept
  N = len(north_vel_ar)
  x = array(north_vel_ar)
  sigmay = np.sqrt((1/(N-2))*np.sum((array(slope_bottom_ar) - Polynomial_Regression[0] - Polynomial_Regression[1]*x)**2))
  std_slope = np.sqrt(sigmay**2/(N*(np.mean(x**2)-np.mean(x)**2)))
  std_interc = np.sqrt(sigmay**2*np.mean(x**2)/(N*(np.mean(x**2)-np.mean(x)**2)))
  
  #Correlation
  corr_coef = corrcoef(north_vel_ar,slope_bottom_ar)
  
  # Plot
  fignr = 41
  fig = figure(fignr,(8,6),None,facecolor='w',edgecolor='k')
  plot(linspace(-0.8,0.8,100), linspace(-0.8,0.8,100),'grey', linewidth = 1, label='y = x')
  cm = plt.cm.get_cmap('autumn') 
  sc = scatter(north_vel_ar,slope_bottom_ar, c = correlation_coef, vmin=0, vmax=0.7, s=20, cmap=cm)
  plot(line, Polynomial_Regression(line),'darkblue',label='y = ({:1.3f} '.format(Polynomial_Regression[1]) + ' $\pm$ {:1.3f})'.format(std_slope) + 'x + ({:1.3f}'.format(Polynomial_Regression[0]) + ' $\pm$ {:1.3f}'.format(std_interc) + ')\ncorrelation coefficient: ' +str(round(corr_coef[0][1],3)))
  xlabel('Mean northward velocity measured by the front ADCP [m/s]')
  ylabel('Computed northward velocity by using the \nrelation topography-vertical velocity [m/s]')
  ylim([-0.8,0.8])
  xlim([-0.8,0.8])
  legend()

  bounds = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7]
  norm = matplotlib.colors.BoundaryNorm(bounds, cm.N, extend='both')
  fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cm),orientation='vertical')
  
  major_ticks_y = np.arange(-0.8, 1, 0.2)
  minor_ticks_y = np.arange(-0.8, 0.85, 0.05)
  major_ticks_x = np.arange(-0.8, 1, 0.2)
  minor_ticks_x = np.arange(-0.8, 0.85, 0.05) 
  ax = plt.gca()
  ax.set_xticks(major_ticks_x)
  ax.set_xticks(minor_ticks_x, minor=True)
  ax.set_yticks(major_ticks_y)
  ax.set_yticks(minor_ticks_y, minor=True)
  ax.grid(which='both')
  ax.grid(which='minor', alpha=0.3)
  ax.grid(which='major', alpha=0.6)
  ax.set_aspect('equal')

  if savefigg == 1:
    figfilename=str(yy)+str(mm)+str(days)+'_correl_topography_northward_vel_VS_bottom_gradient_even_crossings'.zfill(2)+figtype
    savefig(figdir+'/'+figfilename,dpi=300)
    close(fignr)

  return 
  
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

#-----------------------------------------------------------------------

def compute_envir_flow(indir,names,yy,mm,dd,error_thresh, instance):
  # estimated environmnetal northward flow by using the measured eastward velocity 
  
  environmental_flow_hd = []; environmental_flow_tx = []

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
      bo = ~isnan(vel_hd[i,j])
      if  bo== True: 
        tot_vel_hd.append(vel_hd[i,j]) # vertically averaged velocity for each water column 
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
      bo = ~isnan(vel_tx[i,j])
      if  bo == True: 
        tot_vel_tx.append(vel_tx[i,j]) # vertically averaged velocity for each water column 
    environmental_flow_tx.append(mean(tot_vel_tx)*sin(angle*pi/180)/cos(angle*pi/180))

  tight_layout()
  
  
  return  environmental_flow_hd, environmental_flow_tx  
    

#-----------------------------------------------------------------------

########################### Settings ######################################

# The names of the ADCPs
names=['hd','tx']

# Choose date
yy='2021'; mm = '07'; dd = '15';  

# Crossings
instance_step=1;
instances = arange(5,7,instance_step) 
 
# Velocity
varname = 'VERT_VEL'; bt_vars = 'BT_NORTH_VEL'

# Error threshold
error_thresh = 0.35   # mask using error velocity 0.2

# Bad_fraction
bad_fraction = 0.3      # don't plot columns with a fraction of points with error velocity over error_threshold larger than bad_fraction

# Location of the data
if yy =='2022' and mm == '06' and dd == '01':
  indir = '/home/prdusr/data_out/TESO/daily'
else: 
  indir='/home/jvandermolen/data_out/TESO/daily'

fs = 12 
dpi_setting=300
figtype='.jpg'
figdir='/home/akaraoli/python_scripts/figures/'
savefigg = 0


########################### Main ######################################


# ----- Correlation between the corrected vertical velocity and the bottom gradient -------------
vertical_vel_VS_bottom_gradient(indir,yy,mm,dd,instances, error_thresh, bad_fraction, varname, bt_vars)


# ----- Correlation between the measured vertical velocity and the estimated northward velocity through the equation w = -u dh/dx -v dh/dy --------
#Change the date in the function
estimated_vel_VS_measured_vel()

show()  