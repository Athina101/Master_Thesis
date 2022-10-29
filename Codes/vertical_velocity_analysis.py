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
    
#-------------------------------------------------------------------------
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
        
#---------------------------------------------------------------------------
def vertical_vel_profile(indir,name,yy,mm,dd,varname,instance,fignr,error_thresh,varname_bt): # for one wtaer column

  print(yy+mm+dd+' transect: '+str(instance))
  
  # Chosse latitude 
  latt = [52.9876]
  
  # For the figures
  nsub=1; nrows=2; ncols=1
  
  for name in names:
    figure(fignr,(8,10),None,facecolor='w',edgecolor='k')
    
    for l in latt:
      [lat,lat_units,lat_longname]=loadvar(indir,name,yy,mm,dd,'LATITUDE',instance)
      [dep,dep_units,dep_longname]=loadvar(indir,name,yy,mm,dd,'BINDEPTH',instance)
      [siv,siv_units,siv_longname]=loadvar(indir,name,yy,mm,dd,'ERROR_VEL',instance)
      [plv,plv_units,plv_longname]=loadvar(indir,name,yy,mm,dd,varname,instance)
      
      # Direction of the ferry
      if name=='hd':
        if (lat[100]-lat[200]) > 0:
          ar='right'
        else:
          ar='left'

      sd=shape(dep)
      depstep=dep[0,1]-dep[0,0]
      depax=arange(dep[0,0],dep[0,0]+sd[1]*depstep,depstep)
      
      latmax=0
      while lat[latmax]>0:
        latmax=latmax+1
        if latmax>len(lat):
          break
  
      below_bottom=where(abs(plv)>10.,1,0)    
      plvp=where(abs(siv)>error_thresh,-9999.,plv)            # don't plot points with large error velocity
      plvp=where(below_bottom==1,NaN,plvp)
      plv_cleaned=remove_bad_columns(plvp)                    # remove columns with a large number of errors
      plvT=1*plv_cleaned.T
      
      if name == 'hd':
        lat_hd =[]; vel_hd = zeros([len(depax),latmax]);
        limit_hd = latmax
        for i in range(len(lat[0:latmax])):
          lat_hd.append(lat[i])
        for i in range(len(depax)):
          for j in range(latmax):
            vel_hd[i,j] = plvT[i,j]     
      if name == 'tx':
        lat_tx = []; vel_tx = zeros([len(depax),latmax]); 
        limit_tx = latmax
        for i in range(len(lat[0:latmax])):
          lat_tx.append(lat[i]) 
        for i in range(len(depax)):
          for j in range(latmax):
            vel_tx[i,j] = plvT[i,j]
    
      # Find the vertical velocity at that spceific location 
      if name == 'hd':
        vel = vel_hd[:,where(lat == find_nearest(lat, l))[0][0]]
      if name == 'tx':
        vel = vel_tx[:,where(lat == find_nearest(lat, l))[0][0]]  
    
      # Remove the nan values in order to fit the data 
      tot_vel = []; tot_dep = []
      for j in range(len(vel)):
        res = vel[j]
        res1 = ~isnan(res)
        if  res1 == True: 
          tot_vel.append(res)
          tot_dep.append(-depax[j])
          
      # colors for the figures
      if name=='hd':
        c = 'royalblue'
      if name =='tx':
        c = 'orange'

      # Linear fitting
      Polynomial_Regression = poly1d(polyfit(tot_vel,tot_dep,1))
      line = linspace(min(tot_vel), max(tot_vel), 500)
      
      # Errors of the slope and the intercept after the fitting process
      N = len(tot_vel)
      x = array(tot_vel)
      sigmay = np.sqrt((1/(N-2))*np.sum((array(tot_dep) - Polynomial_Regression[0] - Polynomial_Regression[1]*x)**2))
      std_slope = np.sqrt(sigmay**2/(N*(np.mean(x**2)-np.mean(x)**2)))
      std_interc = np.sqrt(sigmay**2*np.mean(x**2)/(N*(np.mean(x**2)-np.mean(x)**2)))
      
      # Plots 
      # Vertical velocity profile for each ADCPs
      subplot(nrows,ncols,nsub)
      scatter(tot_vel,tot_dep, color = c )
      plot(line, Polynomial_Regression(line),'r',label='w = ({:1.3f} '.format(Polynomial_Regression[1]) + ' $\pm$ {:1.3f})'.format(std_slope) + 'z + ({:1.3f}'.format(Polynomial_Regression[0]) + ' $\pm$ {:1.3f}'.format(std_interc) + ')')
      title('Latitude '+str(round(lat[where(lat == find_nearest(lat, l))[0][0]],3)),fontsize=fs+2)
      grid()
      plt.yticks(fontsize=12)
      plt.xticks(fontsize=12)
      xlabel('Vertical velocity ['+plv_units+']',fontsize=fs+1)
      ylabel(dep_longname+' ['+dep_units+']',fontsize=fs+1)
      if name == 'hd':
          title('hd ADCP',fontsize=fs)
      if name == 'tx':
        title('tx ADCP',fontsize=fs)
         
      nsub=nsub+1
      
  if savefigg == 1:
    figfilename=varname+'_'+str(yy)+str(mm)+str(dd)+'_'+str(instance)+'_decayscale'.zfill(2)+figtype
    savefig(figdir+'/'+figfilename,dpi=300)
    close(fignr)
      
      
  # Vertical velocity profile for both ADCPs
  fignr = fignr + 1
  figure(fignr,(9,10),None,facecolor='w',edgecolor='k')
  plot(vel_hd[:,where(lat_hd == find_nearest(lat_hd, l))[0][0]],-depax, label = 'hd' ,linewidth =3)
  plot(vel_tx[:,where(lat_tx == find_nearest(lat_tx, l))[0][0]],-depax, label = 'tx',linewidth =3 )
  plt.xlim([-0.5,0.5])
  plt.ylim([-27,-5])
  xlabel('Vertical velocity ['+plv_units+']',fontsize=fs+1)
  ylabel(dep_longname+' ['+dep_units+']',fontsize=fs+1)
  legend();grid()
  
  if savefigg == 1:
    figfilename=varname+'_'+str(yy)+str(mm)+str(dd)+'_'+str(instance)+'_profile'.zfill(2)+figtype
    savefig(figdir+'/'+figfilename,dpi=300)
    close(fignr)
  
  tight_layout()

  return


#---------------------------------------------------------------------------

def linear_relation_analysis(indir,names,yy,mm,dd,fignr,varname,instance,error_thresh):
  
  # loop over instruments
  for name in names:
    [lat,lat_units,lat_longname]=loadvar(indir,name,yy,mm,dd,'LATITUDE',instance)
    [dep,dep_units,dep_longname]=loadvar(indir,name,yy,mm,dd,'BINDEPTH',instance)
    [siv,siv_units,siv_longname]=loadvar(indir,name,yy,mm,dd,'ERROR_VEL',instance)
    [plv,plv_units,plv_longname]=loadvar(indir,name,yy,mm,dd,varname,instance)
    
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
    
    # Save the latitude data for each ADCP separately     
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
        
    #Compute the decay scale and the vertical velocity at the sea surface for all latitudes in one transect
    if name == 'hd':
      decay_scale_hd = []; latt_hd = []; intercept_hd = []
      for i in range(len(lat_hd)):
        tot_dep_hd = []; tot_vel_hd = []
        for j in range(len(vel_hd[:,i])):
          bo = ~isnan(vel_hd[j,i])
          if  bo == True: 
            tot_vel_hd.append(vel_hd[j,i])
            tot_dep_hd.append(-depax[j])
        if tot_dep_hd:  # if the arrays are not empty then compute the polynomial regression
          # Polynomial Regression
          Polynomial_Regression_hd = poly1d(polyfit(tot_dep_hd, tot_vel_hd,1))
          decay_scale_hd.append(Polynomial_Regression_hd[1])
          intercept_hd.append(Polynomial_Regression_hd[0])
          latt_hd.append(lat_hd[i])
        else:  # if the arrays are empty then continue the for loop
          continue
        
        
    if name == 'tx':
      decay_scale_tx = []; latt_tx = []; intercept_tx = []
      for i in range(len(lat_tx)):
        tot_dep_tx = []; tot_vel_tx = []
        for j in range(len(vel_tx[:,i])):
          bo = ~isnan(vel_tx[j,i])
          if  bo == True: 
            tot_vel_tx.append(vel_tx[j,i])
            tot_dep_tx.append(-depax[j])
        if tot_dep_tx:  # if the arrays are not empty then compute the polynomial regression
          # Polynomial Regression
          Polynomial_Regression_tx = poly1d(polyfit(tot_dep_tx, tot_vel_tx,1))
          decay_scale_tx.append(Polynomial_Regression_tx[1])
          intercept_tx.append(Polynomial_Regression_tx[0])
          latt_tx.append(lat_tx[i])
        else:   # if the arrays are empty then continue the for loop
          continue
   
  #Remove the data near the coasts in order to compute the average value
  x = linspace(52.960,53.100,100) # raneg of latitude 
  latt_without_coasts_hd = []; decay_scale_without_coasts_hd = []; intercept_without_coasts_hd = []
  latt_without_coasts_tx = []; decay_scale_without_coasts_tx = []; intercept_without_coasts_tx = []
  for i in range(len(latt_hd)):
    if latt_hd[i] >= 52.967 and latt_hd[i] <= 52.999:
      latt_without_coasts_hd.append(latt_hd[i])
      decay_scale_without_coasts_hd.append(decay_scale_hd[i])
      intercept_without_coasts_hd.append(intercept_hd[i])
  for i in range(len(latt_tx)):
    if latt_tx[i] >= 52.967 and latt_tx[i] <= 52.999:
      latt_without_coasts_tx.append(latt_tx[i])
      decay_scale_without_coasts_tx.append(decay_scale_tx[i])
      intercept_without_coasts_tx.append(intercept_tx[i])
  
  
  # Define the front and the rear ADCP  
  if ar == 'right':
    latt_front = latt_hd; decay_scale_front = decay_scale_hd; intercept_front = intercept_hd
    latt_back = latt_tx; decay_scale_back = decay_scale_tx; intercept_back = intercept_tx
  if ar == 'left':
    latt_front = latt_tx; decay_scale_front = decay_scale_tx; intercept_front = intercept_tx
    latt_back = latt_hd; decay_scale_back = decay_scale_hd; intercept_back = intercept_hd
  
       
  if ar == 'right':
    decay_scale_without_coasts_front = decay_scale_without_coasts_hd; intercept_without_coasts_front = intercept_without_coasts_hd
    decay_scale_without_coasts_back = decay_scale_without_coasts_tx; intercept_without_coasts_back = intercept_without_coasts_tx
  if ar == 'left':
    decay_scale_without_coasts_front = decay_scale_without_coasts_tx; intercept_without_coasts_front = intercept_without_coasts_tx
    decay_scale_without_coasts_back = decay_scale_without_coasts_hd; intercept_without_coasts_back = intercept_without_coasts_hd 

  
  
  # ------------------ Plots --------------------------------------------------

  figure(fignr,(fs1,fs2-4),None,facecolor='w',edgecolor='k')
  nsub=0; nrows=1; ncols=2
  
  subplot(nrows,ncols,nsub+1)
  scatter(latt_front, decay_scale_front, color ='royalblue')
  xlabel(lat_longname+' ['+lat_units+']',fontsize=fs)
  xlim([min(latt_front)-0.003,max(latt_front)+0.003])
  ylim([-0.08,0.130])
  ylabel('Decay scale of the vertical velocity [s$^{-1}$]',fontsize=fs)
  if ar == 'left':
    title('Front ADCP: Transect ' + str(instance) + ' (tx $\longleftarrow$ hd)', fontsize=fs)
  elif ar == 'right':
    title('Front ADCP: Transect ' + str(instance) + ' (tx $\longrightarrow$ hd)', fontsize=fs)
  #print('Front ADCP: Transect ' + str(instance) + ' (tx $\longleftarrow$ hd)' + 'average decay scale: ' + str(round(mean(decay_scale_without_coasts_front),4)) + ' s$^{-1}$')  
  plot(x,ones(len(x))*mean(decay_scale_without_coasts_front),color = 'darkred',linestyle='--',label= 'average decay scale')
  grid()
  legend(loc = 9)
  
  
  subplot(nrows,ncols,nsub+2)
  scatter(latt_back, decay_scale_back, color ='salmon')
  xlabel(lat_longname+' ['+lat_units+']',fontsize=fs)
  xlim([min(latt_back)-0.003,max(latt_back)+0.003])
  ylim([-0.08,0.130]) 
  ylabel('Decay scale of the vertical velocity [s$^{-1}$]',fontsize=fs)
  if ar == 'left':
    title('Rear ADCP: Transect ' + str(instance) + ' (tx $\longleftarrow$ hd)', fontsize=fs)
  elif ar == 'right':
    title('Rear ADCP: Transect ' + str(instance) + ' (tx $\longrightarrow$ hd)', fontsize=fs)
  #print('Rear ADCP: Transect ' + str(instance) + ' (tx $\longleftarrow$ hd)' + 'average decay scale: ' + str(round(mean(decay_scale_without_coasts_back),4)) + ' s$^{-1}$') 
  plot(x,ones(len(x))*mean(decay_scale_without_coasts_back),color = 'darkred',linestyle='--',label= 'average decay scale')
  grid()
  legend(loc = 9)
  
 
  if savefigg == 1:
    figfilename='VERT_VEL_'+yy+mm+dd+'_'+str(instance)+'_decayVSlat'.zfill(2)+figtype
    savefig(figdir+'/'+figfilename,dpi=300)
    close(fignr)
    
    
  # ------------------------------------
  
  fignr = fignr + 1
  figure(fignr,(fs1,fs2-4),None,facecolor='w',edgecolor='k')
  nsub=0; nrows=1; ncols=2
   
  subplot(nrows,ncols,nsub+1)
  scatter(latt_front, intercept_front, color ='royalblue')
  xlabel(lat_longname+' ['+lat_units+']',fontsize=fs)
  xlim([min(latt_front)-0.003,max(latt_front)+0.003])
  ylim([-0.8,1.3]) 
  ylabel('Vertical velocity at the sea surface [m/s]',fontsize=fs)
  if ar == 'left':
    title('Front ADCP: Transect ' + str(instance) + ' (tx $\longleftarrow$ hd)', fontsize=fs)
  elif ar == 'right':
    title('Front ADCP: Transect ' + str(instance) + ' (tx $\longrightarrow$ hd)', fontsize=fs)  
  #print('Front ADCP: Transect ' + str(instance) + ' (tx $\longleftarrow$ hd)' + 'averaged value = ' + str(round(mean(intercept_without_coasts_front),4)) + ' m/s')  
  plot(x,ones(len(x))*mean(intercept_without_coasts_front),color = 'darkred',linestyle='--',label= 'average vertical velocity ')
  legend(loc = 9)
  grid()
  
  subplot(nrows,ncols,nsub+2)
  scatter(latt_back, intercept_back, color ='salmon')
  xlabel(lat_longname+' ['+lat_units+']',fontsize=fs)
  xlim([min(latt_back)-0.003,max(latt_back)+0.003])
  ylim([-0.8,1.3])
  ylabel('Vertical velocity at the sea surface [m/s]',fontsize=fs)
  if ar == 'left':
    title('Rear ADCP: Transect ' + str(instance) + ' (tx $\longleftarrow$ hd)', fontsize=fs)
  elif ar == 'right':
    title('Rear ADCP: Transect ' + str(instance) + ' (tx $\longrightarrow$ hd)', fontsize=fs) 
  #print('Rear ADCP: Transect ' + str(instance) + ' (tx $\longleftarrow$ hd)' + 'averaged value = ' + str(round(mean(intercept_without_coasts_back),4)) + ' m/s')  
  plot(x,ones(len(x))*mean(intercept_without_coasts_back),color = 'darkred',linestyle='--',label= 'average vertical velocity')
  legend(loc = 9)
  grid()
  
  if savefigg ==1:
    figfilename='VERT_VEL_'+yy+mm+dd+'_'+str(instance)+'_interceptVSlat'.zfill(2)+figtype
    savefig(figdir+'/'+figfilename,dpi=300)
    close(fignr)
    
  tight_layout()

  return

#---------------------------------------------------------------------------

def avg_decay_scale_and_intercept(indir,names,yy,mm,dd,fignr,varname,instance,varname_bt,error_thresh):
  
  # Estimation of the environmnetal flow
  envir_flow_hd, envir_flow_tx = compute_envir_flow(indir,names,yy,mm,dd,error_thresh, instance)
  
  # loop over instruments
  for name in names:
    [lat,lat_units,lat_longname]=loadvar(indir,name,yy,mm,dd,'LATITUDE',instance)
    [dep,dep_units,dep_longname]=loadvar(indir,name,yy,mm,dd,'BINDEPTH',instance)
    [siv,siv_units,siv_longname]=loadvar(indir,name,yy,mm,dd,'ERROR_VEL',instance)
    [plv,plv_units,plv_longname]=loadvar(indir,name,yy,mm,dd,varname,instance)
    [bt,bt_units,bt_longname]=loadvar(indir,name,yy,mm,dd,varname_bt,instance)
    
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
    
    # Save the latitude data for each ADCP separately    
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
    
    # Save the velocity data and the bottom track velocity data for each ADCP separately
    if name == 'hd':
      vel_hd = zeros([len(depax),latmax]); bt_hd = zeros(latmax); limit_hd = latmax
      for i in range(len(depax)):
        for j in range(latmax):
          vel_hd[i,j] = plvT[i,j]
      for w in range(latmax):
        bt_hd[w] = bt[w]
    if name == 'tx':
      vel_tx = zeros([len(depax),latmax]); bt_tx = zeros(latmax); limit_tx = latmax
      for i in range(len(depax)):
        for j in range(latmax):
          vel_tx[i,j] = plvT[i,j]
      for w in range(latmax):
        bt_tx[w] = bt[w]
        
    #Compute the decay scale for all latitudes in one transect
    if name == 'hd':
      decay_scale_hd = [];latt_hd = []; intercept_hd = []; btt_hd = []
      for i in range(len(lat_hd)):
        tot_dep_hd = []; tot_vel_hd = []
        for j in range(len(vel_hd[:,i])):
          bo = ~isnan(vel_hd[j,i])
          if  bo == True: 
            tot_vel_hd.append(vel_hd[j,i])
            tot_dep_hd.append(-depax[j])
        if tot_dep_hd: # if the arrays are not empty then compute the polynomial regression
          # Polynomial Regression
          Polynomial_Regression_hd = poly1d(polyfit(tot_dep_hd, tot_vel_hd,1))
          decay_scale_hd.append(Polynomial_Regression_hd[1])
          intercept_hd.append(Polynomial_Regression_hd[0])
          btt_hd.append(-bt_hd[i]-envir_flow_hd[i])
          latt_hd.append(lat_hd[i])
        else:   # if the arrays are empty then continue the for loop
          continue
        
        
    if name == 'tx':
      decay_scale_tx = []; latt_tx = []; intercept_tx = []; btt_tx = []
      for i in range(len(lat_tx)):
        tot_dep_tx = []; tot_vel_tx = []
        for j in range(len(vel_tx[:,i])):
          bo = ~isnan(vel_tx[j,i])
          if  bo == True: 
            tot_vel_tx.append(vel_tx[j,i])
            tot_dep_tx.append(-depax[j])
        if tot_dep_tx:  # if the arrays are not empty then compute the polynomial regression
          # Polynomial Regression
          Polynomial_Regression_tx = poly1d(polyfit(tot_dep_tx, tot_vel_tx,1))
          decay_scale_tx.append(Polynomial_Regression_tx[1])
          intercept_tx.append(Polynomial_Regression_tx[0])
          btt_tx.append(-bt_tx[i]-envir_flow_tx[i])
          latt_tx.append(lat_tx[i])
        else:  # if the arrays are empty then continue the for loop
          continue
    
  #Remove the coasts
  x = linspace(52.960,53.100,100)
  latt_without_coasts_hd = []; decay_scale_without_coasts_hd = []; intercept_without_coasts_hd = []; btt_hd_without_coasts = []
  latt_without_coasts_tx = []; decay_scale_without_coasts_tx = []; intercept_without_coasts_tx = []; btt_tx_without_coasts = []
  for i in range(len(latt_hd)):
    if latt_hd[i] >= 52.967 and latt_hd[i] <= 52.999:
      latt_without_coasts_hd.append(latt_hd[i])
      decay_scale_without_coasts_hd.append(decay_scale_hd[i])
      intercept_without_coasts_hd.append(intercept_hd[i])
      btt_hd_without_coasts.append(abs(btt_hd[i]))
  for i in range(len(latt_tx)):
    if latt_tx[i] >= 52.967 and latt_tx[i] <= 52.999:
      latt_without_coasts_tx.append(latt_tx[i])
      decay_scale_without_coasts_tx.append(decay_scale_tx[i])
      intercept_without_coasts_tx.append(intercept_tx[i])
      btt_tx_without_coasts.append(abs(btt_tx[i]))
   
    
  return mean(decay_scale_without_coasts_hd), mean(decay_scale_without_coasts_tx), mean(btt_hd_without_coasts), mean(btt_tx_without_coasts), mean(intercept_without_coasts_hd), mean(intercept_without_coasts_tx), ar

  
  
#-----------------------------------------------------------------------
def relation_between_vertical_vel_and_ferry_velocity(fignr):
 
  pl_vars = 'VERT_VEL'
  decay_scale_front = []; ship_vel_front = []; surface_velocity_front = []; decay_scale_back = []; ship_vel_back = []; surface_velocity_back = []
  
  # Date
  yy='2021'; mm = '07';  error_thresh = 0.22
  # Days and crossings 
  days = ['12','13','14','15','16','17']
  instances = arange(0,32,1)
  
  for dd in days: 
    for instance in instances:
      mean_decay_scale_hd, mean_decay_scale_tx, ship_speed_hd, ship_speed_tx, mean_surf_values_hd, mean_surf_values_tx, ar = avg_decay_scale_and_intercept(indir,names,yy,mm,dd,fignr,pl_vars,instance,bt_vars,error_thresh) 

      # Define the front and the rear ADCP 
      if ar == 'right':
        decay_scale_front.append(mean_decay_scale_hd); ship_vel_front.append(abs(ship_speed_hd)); surface_velocity_front.append(mean_surf_values_hd)
        decay_scale_back.append(mean_decay_scale_tx); ship_vel_back.append(abs(ship_speed_tx)); surface_velocity_back.append(mean_surf_values_tx)
      if ar == 'left':  
        decay_scale_front.append(mean_decay_scale_tx); ship_vel_front.append(abs(ship_speed_tx)); surface_velocity_front.append(mean_surf_values_tx)
        decay_scale_back.append(mean_decay_scale_hd); ship_vel_back.append(abs(ship_speed_hd)); surface_velocity_back.append(mean_surf_values_hd)

      print(dd, instance)
      
  # ---------------------------- Decay scale = f (ferry's velocity) ---------------------------------------------------------------

 # Polynomial Regression Front ADCP
  Polynomial_Regression_front = poly1d(polyfit(ship_vel_front, decay_scale_front,1))
  line_front = linspace(min(ship_vel_front), max(ship_vel_front), 500)
 
  N = len(ship_vel_front)
  x = array(ship_vel_front)
  sigmay = np.sqrt((1/(N-2))*np.sum((array(decay_scale_front) - Polynomial_Regression_front[0] - Polynomial_Regression_front[1]*x)**2))
  std_slope_front = np.sqrt(sigmay**2/(N*(np.mean(x**2)-np.mean(x)**2)))
  std_interc_front = np.sqrt(sigmay**2*np.mean(x**2)/(N*(np.mean(x**2)-np.mean(x)**2)))
  
  fignr = fignr 
  figure(fignr,(fs1-7,fs2),None,facecolor='w',edgecolor='k')  
  scatter(ship_vel_front, decay_scale_front, color = 'royalblue')
  plot(line_front, Polynomial_Regression_front(line_front),'k',label='y = ({:1.6f} '.format(Polynomial_Regression_front[1]) + ' $\pm$ {:1.6f})'.format(std_slope_front) + 'x + ({:1.6f}'.format(Polynomial_Regression_front[0]) + ' $\pm$ {:1.6f}'.format(std_interc_front) + ')')
  xlabel('Mean value of the speed of the ship relative to the water [m/s]',fontsize=13)
  ylabel('Mean value of the decay scale of w$_{front}$ [s$^{-1}$]',fontsize=13)
  title('Front ADCP')
  legend(fontsize=14)
  plt.yticks(fontsize=15)
  plt.xticks(fontsize=15)
  grid()
  
  if savefigg == 1:
    figfilename='VERT_VEL_decay_scale_VS_ship_speed_front'.zfill(2)+figtype
    savefig(figdir+'/'+figfilename,dpi=300)
    close(fignr)
  
  # Polynomial Regression Rear ADCP
  Polynomial_Regression_back = poly1d(polyfit(ship_vel_back, decay_scale_back,1))
  line_back = linspace(min(ship_vel_back), max(ship_vel_back), 500)
 
  N = len(ship_vel_back)
  x = array(ship_vel_back)
  sigmay = np.sqrt((1/(N-2))*np.sum((array(decay_scale_back) - Polynomial_Regression_back[0] - Polynomial_Regression_back[1]*x)**2))
  std_slope_back = np.sqrt(sigmay**2/(N*(np.mean(x**2)-np.mean(x)**2)))
  std_interc_back = np.sqrt(sigmay**2*np.mean(x**2)/(N*(np.mean(x**2)-np.mean(x)**2)))
  
  fignr = fignr + 1 
  figure(fignr,(fs1-7,fs2),None,facecolor='w',edgecolor='k') 
  scatter(ship_vel_back, decay_scale_back, color = 'salmon')
  plot(line_back, Polynomial_Regression_back(line_back),'k',label='y = ({:1.6f} '.format(Polynomial_Regression_back[1]) + ' $\pm$ {:1.6f})'.format(std_slope_back) + 'x + ({:1.6f}'.format(Polynomial_Regression_back[0]) + ' $\pm$ {:1.6f}'.format(std_interc_back) + ')')
  xlabel('Mean value of the speed of the ship relative to the water [m/s]',fontsize=13)
  ylabel('Mean value of the decay scale of w$_{back}$ [s$^{-1}$]',fontsize=13)
  title('Rear ADCP')
  legend(fontsize=14)
  plt.yticks(fontsize=15)
  plt.xticks(fontsize=15)
  grid()
  
 
  if savefigg == 1:
    figfilename='VERT_VEL_decay_scale_VS_ship_speed_back'.zfill(2)+figtype
    savefig(figdir+'/'+figfilename,dpi=300)
    close(fignr)
  

  # ---------------------------- Vertical velocity at the sea surface = f (ferry's velocity) ---------------------------------------------------------------

 # Polynomial Regression Front ADCP
  Polynomial_Regression_front = poly1d(polyfit(ship_vel_front, surface_velocity_front,1))
  line_front = linspace(min(ship_vel_front), max(ship_vel_front), 500)
 
  N = len(ship_vel_front)
  x = array(ship_vel_front)
  sigmay = np.sqrt((1/(N-2))*np.sum((array(surface_velocity_front) - Polynomial_Regression_front[0] - Polynomial_Regression_front[1]*x)**2))
  std_slope_front = np.sqrt(sigmay**2/(N*(np.mean(x**2)-np.mean(x)**2)))
  std_interc_front = np.sqrt(sigmay**2*np.mean(x**2)/(N*(np.mean(x**2)-np.mean(x)**2)))
 
  fignr = fignr + 1
  figure(fignr,(fs1-7,fs2),None,facecolor='w',edgecolor='k')  
  scatter(ship_vel_front, surface_velocity_front, color = 'royalblue')
  plot(line_front, Polynomial_Regression_front(line_front),'k',label='y = ({:1.6f} '.format(Polynomial_Regression_front[1]) + ' $\pm$ {:1.6f})'.format(std_slope_front) + 'x + ({:1.6f}'.format(Polynomial_Regression_front[0]) + ' $\pm$ {:1.6f}'.format(std_interc_front) + ')')
  xlabel('Mean value of the speed of the ship relative to the water [m/s]',fontsize=13)
  ylabel('Mean value of w$_{front}$ at the surface [m/s]',fontsize=13)
  title('Front ADCP')
  legend(fontsize=14)
  plt.yticks(fontsize=15)
  plt.xticks(fontsize=15)
  grid()
  
  
  if savefigg == 1:
    figfilename='VERT_VEL_intercept_VS_ship_speed_front'.zfill(2)+figtype
    savefig(figdir+'/'+figfilename,dpi=300)
    close(fignr)
  
  
  # Polynomial Regression Rear ADCP
  Polynomial_Regression_back = poly1d(polyfit(ship_vel_back, surface_velocity_back,1))
  line_back = linspace(min(ship_vel_back), max(ship_vel_back), 500)
 
  N = len(ship_vel_back)
  x = array(ship_vel_back)
  sigmay = np.sqrt((1/(N-2))*np.sum((array(surface_velocity_back) - Polynomial_Regression_back[0] - Polynomial_Regression_back[1]*x)**2))
  std_slope_back = np.sqrt(sigmay**2/(N*(np.mean(x**2)-np.mean(x)**2)))
  std_interc_back = np.sqrt(sigmay**2*np.mean(x**2)/(N*(np.mean(x**2)-np.mean(x)**2)))
  
  fignr = fignr + 1
  figure(fignr,(fs1-7,fs2),None,facecolor='w',edgecolor='k')
  scatter(ship_vel_back, surface_velocity_back, color = 'salmon')
  plot(line_back, Polynomial_Regression_back(line_back),'k',label='y = ({:1.6f} '.format(Polynomial_Regression_back[1]) + ' $\pm$ {:1.6f})'.format(std_slope_back) + 'x + ({:1.6f}'.format(Polynomial_Regression_back[0]) + ' $\pm$ {:1.6f}'.format(std_interc_back) + ')')
  xlabel('Mean value of the speed of the ship relative to the water [m/s]',fontsize=13)
  ylabel('Mean value of w$_{back}$ at the surface [m/s]',fontsize=13)
  title('Rear ADCP')
  legend(fontsize=14)
  plt.yticks(fontsize=15)
  plt.xticks(fontsize=15)
  grid()

  
  if savefigg == 1:
    figfilename='VERT_VEL_intercept_VS_ship_speed_back'.zfill(2)+figtype
    savefig(figdir+'/'+figfilename,dpi=300)
    close(fignr)
  
  
  tight_layout()    
      
  
  return   
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

  # Exlude the first 4m below the hull to avoid disturbance due to ferry's motion
  depth_min = 7; depth_max = 30; lat_min = 52.9;  lat_max = 53.1
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

# Choose date
yy='2021'; mm = '07'; dd = '15';  

# Crossings
instance_1 = 5; instance_2 = 6; instance_step=1;

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
 
# max value for the colorbar 
vmax=0.2

fs1 = 16; fs2 = 9; fs = 12 
dpi_setting=300
figtype='.jpg'
figdir='/home/akaraoli/python_scripts/figures/'
figlabel=['a','b','c','d','e','f','g','h','i','j']
savefigg = 0


############ Main ######################################################


# -------------- Vertical velocity profile ----------------------------
fignr = 2
for instance in range(instance_1,instance_2,instance_step):
  vertical_vel_profile(indir,names,yy,mm,dd,varname,instance,fignr,error_thresh,bt_vars)
  fignr=fignr + 2


# -------------- Analysis of the linear decay of the vertical velocity ----------------------------
for instance in range(instance_1,instance_2,instance_step):
  linear_relation_analysis(indir,names,yy,mm,dd,fignr,varname,instance,error_thresh)
  fignr=fignr + 2


# ------- Find the relation between the measured vertical velocity and the ferry's speed -----------
relation_between_vertical_vel_and_ferry_velocity(fignr) 


show()  