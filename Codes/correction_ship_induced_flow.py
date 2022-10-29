#! /usr/bin/env python

# python script to plot TESO transects

# Author: Athina Karaoli, March 2022

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

############# Functions ###############################################

#------------------------------------------------------------------------

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

# ----------------------------------------------------------------------
def find_nearest(array, value):
    array = asarray(array)
    idx = (abs(array - value)).argmin()
    return array[idx]
  
#---------------------------------------------------------------------------------------
  
def ferry_speed_subsequent_transects(indir,names,yy,mm,dd,instance_1, instance_2, error_thresh, varname, varname_bt):


  fignr = 1
  figure(fignr,(10,8),None,facecolor='w',edgecolor='k')
  nsub=1

  for instance in range(instance_1, instance_2, instance_step):
    environmental_hd,environmental_tx = compute_envir_flow(indir,names,yy,mm,dd,error_thresh, instance)
    
    # loop over instruments
    for name in names:
      [lat,lat_units,lat_longname]=loadvar(indir,name,yy,mm,dd,'LATITUDE',instance)
      [dep,dep_units,dep_longname]=loadvar(indir,name,yy,mm,dd,'BINDEPTH',instance)
      [siv,siv_units,siv_longname]=loadvar(indir,name,yy,mm,dd,'ERROR_VEL',instance)
      [plv,plv_units,plv_longname]=loadvar(indir,name,yy,mm,dd,varname,instance)
      [bt,bt_units,bt_longname]=loadvar(indir,name,yy,mm,dd,varname_bt,instance)
  
      # Direction of the ferry
      if (lat[100]-lat[200]) > 0:
        ar='right'; arrow  = '$\longrightarrow$'
      else:
        ar='left'; arrow  = '$\longleftarrow$' 
        
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
        environmental_flow = environmental_hd
      if name == 'tx':
        environmental_flow = environmental_tx
      
      # Compute the ferry's velocity through the water
      ship_speed = []; latitude = []
      for k in range(latmax):
        ship_speed.append(-bt[k] - environmental_flow[k])
        latitude.append(lat[k])

      if name=='hd':
        nsub = 1
        subplot(2,1,nsub)  
        title('hd ADCP')
        scatter(latitude,abs(array(ship_speed)), label = 'transect ' + str(instance) + ' (tx ' + arrow +  ' hd)' )
        gca().invert_xaxis()
        ylabel('Ferry\'s speed ' +' ['+bt_units+']',fontsize=fs)
        ylim([0,9])
        legend()
        if instance == instance_1:
          text(52.967,0.2,'Den Helder')
          text(53.004,0.2,'Texel')
      
      if name=='tx':
        nsub = 2
        subplot(2,1,nsub)  
        title('tx ADCP')
        scatter(latitude,abs(array(ship_speed)), label = 'transect ' + str(instance) + ' (tx ' + arrow +  ' hd)' )
        gca().invert_xaxis()
        ylabel('Ferry\'s speed ' +' ['+bt_units+']',fontsize=fs)
        xlabel(lat_longname+' ['+lat_units+']',fontsize=fs)
        ylim([0,9])
        if instance == instance_1:
          text(52.967,0.2,'Den Helder')
          text(53.004,0.2,'Texel')
        
  if savefigg == 1:
    figfilename=pl_vars+'_'+str(yy)+str(mm)+str(dd)+'_['+str(num1)+', '+str(num2)+', '+str(num3)+']_Vs'.zfill(2)+figtype
    savefig(figdir+'/'+figfilename,dpi=300)
    close(fignr)
  

  tight_layout()


  return
  
#---------------------------------------------------------------------------------------
  
def ferry_speed_no_subsequent_transects(indir,names,yy,mm,dd,instance_1, instance_2, error_thresh, varname, varname_bt):

  instances = [instance_1, instance_2]

 
  for i in range(len(instances)):
  
    environmental_hd,environmental_tx = compute_envir_flow(indir,names,yy,mm,dd,error_thresh, instances[i])
    
    # loop over instruments
    for name in names:
      [lat,lat_units,lat_longname]=loadvar(indir,name,yy,mm,dd,'LATITUDE',instances[i])
      [dep,dep_units,dep_longname]=loadvar(indir,name,yy,mm,dd,'BINDEPTH',instances[i])
      [siv,siv_units,siv_longname]=loadvar(indir,name,yy,mm,dd,'ERROR_VEL',instances[i])
      [plv,plv_units,plv_longname]=loadvar(indir,name,yy,mm,dd,varname,instances[i])
      [bt,bt_units,bt_longname]=loadvar(indir,name,yy,mm,dd,varname_bt,instances[i])
  
    
      if (lat[100]-lat[200]) > 0:
        ar='right'
      else:
        ar='left' 
        
      if ar == 'right':
        arrow  = '$\longrightarrow$'
      elif ar == 'left':
        arrow  = '$\longleftarrow$'
    
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
    
     
      # Compute the ferry's velocity through the water
      ship_speed_hd = []; ship_speed_tx = []
      if name == 'hd':
        for k in range(len(bt[0:latmax])):
          ship_speed_hd.append(-bt[k] - environmental_hd[k])
      if name == 'tx':
        for k in range(len(bt[0:latmax])):
          ship_speed_tx.append(-bt[k] - environmental_tx[k])
        
      if i == 0 and name == 'hd':
        lat_hd_trans0 = lat[0:latmax]
        ferry_ship_hd_trans0 = abs(array(ship_speed_hd))
        ar0_hd = arrow
        instance0 = instances[i]
      if i == 0 and name == 'tx':
        lat_tx_trans0 = lat[0:latmax]
        ferry_ship_tx_trans0 = abs(array(ship_speed_tx))
        ar0_tx = arrow
      if i == 1 and name == 'hd':
        lat_hd_trans1 = lat[0:latmax]
        ferry_ship_hd_trans1 = abs(array(ship_speed_hd))
        ar1_hd = arrow
        instance1 = instances[i]
      if i == 1 and name == 'tx':
        lat_tx_trans1 = lat[0:latmax]
        ferry_ship_tx_trans1 = abs(array(ship_speed_tx))
        ar1_tx = arrow
      
   
  fignr = 4
  figure(fignr,(19,5),None,facecolor='w',edgecolor='k') 
    
  subplot(1,2,1)  
  scatter(lat_hd_trans0,ferry_ship_hd_trans0, color = 'limegreen', label = 'transect ' + str(instance0) + ' (tx ' + ar0_hd +  ' hd)' )
  scatter(lat_hd_trans1,ferry_ship_hd_trans1, color = 'darkviolet', label = 'transect ' + str(instance1) + ' (tx ' + ar1_hd +  ' hd)' )
  ylim([0,7])
  title('hd ADCP',fontsize=fs+2)
  ylabel('Ferry\'s speed ' +' ['+bt_units+']',fontsize=fs+3)
  xlabel(lat_longname+' ['+lat_units+']',fontsize=fs+3)
  text(52.969,0.2,'Den Helder',fontsize=13)
  text(53.003,0.2,'Texel',fontsize=13)
  plt.yticks(fontsize=13)
  plt.xticks(fontsize=13)
  gca().invert_xaxis()
  grid()
  legend(loc = 8,fontsize=13)   


  subplot(1,2,2)  
  title('tx ADCP',fontsize=fs+2)
  scatter(lat_tx_trans0,ferry_ship_tx_trans0, color = 'limegreen', label = 'transect ' + str(instance0) + ' (tx ' + ar0_tx +  ' hd)' )
  scatter(lat_tx_trans1,ferry_ship_tx_trans1, color = 'darkviolet', label = 'transect ' + str(instance1) + ' (tx ' + ar1_tx +  ' hd)' )
  #ylabel('Absolute northward bottom \ntrack velocity ' +' ['+bt_units+']',fontsize=fs)
  #ylabel('Ferry\'s speed ' +' ['+bt_units+']',fontsize=fs)
  xlabel(lat_longname+' ['+lat_units+']',fontsize=fs+3)
  ylim([0,7])
  text(52.969,0.2,'Den Helder',fontsize=13)
  text(53.003,0.2,'Texel',fontsize=13)
  plt.yticks(fontsize=13)
  plt.xticks(fontsize=13)
  gca().invert_xaxis()
  grid()
  legend(loc = 8,fontsize=13)     
         

  if savefigg == 1:
    figfilename=pl_vars+'_'+str(yy)+str(mm)+str(dd)+'_['+str(instances[0])+', '+str(instances[1])+']_Vs'.zfill(2)+figtype
    savefig(figdir+'/'+figfilename,dpi=300)
    close(fignr)


  tight_layout()


  return
  


#---------------------------------------------------------------------------------------
  
def backflow(indir,names,yy,mm,dd,instance_1, instance_2, error_thresh, varname, varname_bt):

  fignr = 3
  figure(fignr,(fs1,fs2+2),None,facecolor='w',edgecolor='k')
  nsub=0; nrows = 2; ncols = 2;
  instance = [instance_1, instance_2]

  for transect in range(len(instance)):
    
    #Estimated environmental flow 
    environmental_hd,environmental_tx = compute_envir_flow(indir,names,yy,mm,dd,error_thresh, instance[transect])  

    # loop over instruments
    for name in names:
      [lat,lat_units,lat_longname]=loadvar(indir,name,yy,mm,dd,'LATITUDE',instance[transect])
      [dep,dep_units,dep_longname]=loadvar(indir,name,yy,mm,dd,'BINDEPTH',instance[transect])
      [siv,siv_units,siv_longname]=loadvar(indir,name,yy,mm,dd,'ERROR_VEL',instance[transect])
      [plv,plv_units,plv_longname]=loadvar(indir,name,yy,mm,dd,varname,instance[transect])
      [bt,bt_units,bt_longname]=loadvar(indir,name,yy,mm,dd,varname_bt,instance[transect])
      
      # Direction of the ferry
      if (lat[100]-lat[200]) > 0:
        ar='right'
        arrow  = '$\longrightarrow$'
      else:
        ar='left' 
        arrow  = '$\longleftarrow$'
     
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

 
      # Distinguish the latitute data in front and rear ADCP for the two transects    
      if (transect == 0 and ar == 'right' and name=='hd') or (transect == 0 and ar == 'left' and name=='tx'):
        lat_front_ins0 =[]
        latmax_front_ins0= latmax
        for i in range(len(lat[0:latmax])):
          lat_front_ins0.append(lat[i])
      elif (transect == 0 and ar == 'left' and name=='hd') or (transect == 0 and ar == 'right' and name=='tx'):
        lat_back_ins0=[]
        latmax_back_ins0= latmax
        for i in range(len(lat[0:latmax])):
          lat_back_ins0.append(lat[i])
          
      if (transect == 1 and ar == 'right' and name=='hd') or (transect == 1 and ar == 'left' and name=='tx'):
        lat_front_ins1 =[]
        latmax_front_ins1= latmax
        for i in range(len(lat[0:latmax])):
          lat_front_ins1.append(lat[i])
      elif (transect == 1 and ar == 'left' and name=='hd') or (transect == 1 and ar == 'right' and name=='tx'):
        lat_back_ins1 =[]
        latmax_back_ins1= latmax
        for i in range(len(lat[0:latmax])):
          lat_back_ins1.append(lat[i])
          
          
      # Distinguish the velocity data in front and rear ADCP for the two transects   
      if (transect == 0 and ar == 'right' and name=='hd') or (transect == 0 and ar == 'left' and name=='tx'):
        vel_front_ins0 = zeros([len(depax),latmax])
        for i in range(len(depax)):
          for j in range(latmax):
            if (ar == 'right' and name=='hd'):
              vel_front_ins0[i,j] = plvT[i,j] 
            elif (ar == 'left' and name=='tx'):
              vel_front_ins0[i,j] = plvT[i,j] 
      elif (transect == 0 and ar == 'left' and name=='hd') or (transect == 0 and ar == 'right' and name=='tx'):
        vel_back_ins0 = zeros([len(depax),latmax])
        for i in range(len(depax)):
          for j in range(latmax):
            vel_back_ins0[i,j] = plvT[i,j]
            
      if (transect == 1 and ar == 'right' and name=='hd') or (transect == 1 and ar == 'left' and name=='tx'):
        vel_front_ins1 = zeros([len(depax),latmax])
        for i in range(len(depax)):
          for j in range(latmax):
            if (ar == 'right' and name=='hd'):
              vel_front_ins1[i,j] = plvT[i,j] 
            elif (ar == 'left' and name=='tx'):
              vel_front_ins1[i,j] = plvT[i,j] 
      elif (transect == 1 and ar == 'left' and name=='hd') or (transect == 1 and ar == 'right' and name=='tx'):
        vel_back_ins1 = zeros([len(depax),latmax])
        for i in range(len(depax)):
          for j in range(latmax):
            vel_back_ins1[i,j] = plvT[i,j]
            
   
  # Reverse the latitude and velocity data of the 2nd transect in order to subtract the two profiles  
  lat_front_ins1.reverse()
  lat_back_ins1.reverse()  
  for w in range(len(depax)):            
    vel_front_ins1[w,:] = vel_front_ins1[w,::-1]
    vel_back_ins1[w,:] = vel_back_ins1[w,::-1]
  
  
  # Find the common latitudes for the front and the rear ADCP
  value_tol = 0.000020
        
  lat_common_front = []; index_front0 = [];index_front1 = [];
  if latmax_front_ins0 > latmax_front_ins1:
    for i in range(latmax_front_ins1):
      a = find_nearest(lat_front_ins0, lat_front_ins1[i])
      if abs(a-lat_front_ins1[i]) <= value_tol:
        lat_common_front.append(find_nearest(lat_front_ins0, lat_front_ins1[i]))
        index_front0.append(where(lat_front_ins0 == a)[0][0])
        index_front1.append(where(lat_front_ins1== lat_front_ins1[i])[0][0])   
    latmax2_front = len(lat_common_front)
  else:
    latmax2_front = latmax_front_ins0
    for i in range(latmax_front_ins0):
      a = find_nearest(lat_front_ins1, lat_front_ins0[i])
      if abs(a-lat_front_ins0[i]) <= value_tol:
        lat_common_front.append(find_nearest(lat_front_ins1, lat_front_ins0[i]))
        index_front1.append(where(lat_front_ins1 == a)[0][0])
        index_front0.append(where(lat_front_ins0 == lat_front_ins0[i])[0][0])
      latmax2_front = len(lat_common_front)  
          
  lat_common_back = [];
  index_back0 = [];index_back1 = [];
  if latmax_back_ins0 > latmax_back_ins1:
    for i in range(latmax_back_ins1):
      a = find_nearest(lat_back_ins0, lat_back_ins1[i])
      if abs(a-lat_back_ins1[i]) <= value_tol:
        lat_common_back.append(find_nearest(lat_back_ins0, lat_back_ins1[i]))
        index_back0.append(where(lat_back_ins0 == a)[0][0])
        index_back1.append(where(lat_back_ins1== lat_back_ins1[i])[0][0])   
    latmax2_back = len(lat_common_back)
  else:
    latmax2_back = latmax_back_ins0
    for i in range(latmax_back_ins0):
      a = find_nearest(lat_back_ins1, lat_back_ins0[i])
      if abs(a-lat_back_ins0[i]) <= value_tol:
        lat_common_back.append(find_nearest(lat_back_ins1, lat_back_ins0[i]))
        index_back1.append(where(lat_back_ins1 == a)[0][0])
        index_back0.append(where(lat_back_ins0 == lat_back_ins0[i])[0][0])
      latmax2_back = len(lat_common_back) 


  vel_front0 = zeros([len(depax),len(lat_common_front)]); vel_front1 = zeros([len(depax),len(lat_common_front)])
  vel_back0 = zeros([len(depax),len(lat_common_back)]); vel_back1 = zeros([len(depax),len(lat_common_back)])
  
  for i in range(len(depax)):
    for j in range(len(lat_common_front)):
      vel_front1[i,j] = vel_front_ins1[i,index_front1[j]]
      vel_front0[i,j] = vel_front_ins0[i,index_front0[j]]
  
  for i in range(len(depax)):
    for j in range(len(lat_common_back)):
      vel_back1[i,j] = vel_back_ins1[i,index_back1[j]]
      vel_back0[i,j] = vel_back_ins0[i,index_back0[j]]

  
  # Compute the difference of the velocity profiles for two subsequent transects
  values_front = zeros([len(depax), latmax2_front]); values_back = zeros([len(depax), latmax2_back]);URmax_values= []
  
  # Exclude the data near the harbours and the data at the 5m below the hull 
  depth_min = 9; depth_max = 30; lat_min = 52.967;  lat_max = 52.999
  
  for b in range(len(depax)):
    for c in range(latmax2_front):
      if depax[b] > depth_min and depax[b] < depth_max:
        if lat_common_front[c] > lat_min and lat_common_front[c] < lat_max:
          values_front[b,c] = (vel_front1[b,c] - vel_front0[b,c])
        else:
          values_front[b,c] = nan  
      else:
        values_front[b,c] = nan
        
  for b in range(len(depax)):
    for c in range(latmax2_back):
      if depax[b] > depth_min and depax[b] < depth_max:
        if lat_common_back[c] > lat_min and lat_common_back[c] < lat_max:
          values_back[b,c] = (vel_back1[b,c] - vel_back0[b,c])
        else:
          values_back[b,c] = nan  
      else:
        values_back[b,c] = nan
        
  # Remove the nan values
  hist_values_front = []      
  for b in range(len(depax)):
    for c in range(latmax2_front):
      booll = ~isnan(values_front[b,c])
      if booll == True:
        hist_values_front.append(values_front[b,c])         
              
  hist_values_back = []      
  for b in range(len(depax)):
    for c in range(latmax2_back):
      booll = ~isnan(values_back[b,c])
      if booll == True:
        hist_values_back.append(values_back[b,c])
        

  # ----------------- Plots --------------------------------
     
  subplot(nrows,ncols,nsub+1)
  pcolormesh(lat_common_front,-depax,values_front,vmin=-max_ax,vmax=max_ax,cmap='coolwarm')
  ylim([-30, -5])
  gca().invert_xaxis()
  colorbar()
  ylabel(dep_longname+' ['+dep_units+']',fontsize=fs)
  title('Front ADCP: v (trans' + str(instance[1]) + ') - v (trans' + str(instance[0])+')',fontsize=fs)
  
  subplot(nrows,ncols,nsub+3)
  pcolormesh(lat_common_back,-depax,values_back,vmin=-max_ax,vmax=max_ax,cmap='coolwarm')
  ylim([-30, -5])
  gca().invert_xaxis()
  colorbar()
  ylabel(dep_longname+' ['+dep_units+']',fontsize=fs)
  title('Back ADCP: v (trans' + str(instance[1]) + ') - v (trans' + str(instance[0])+')',fontsize=fs)
  xlabel(lat_longname+' ['+lat_units+']',fontsize=fs)
    
  # Plot histograms for each ADCP
  subplot(nrows,ncols,nsub+2)
  hist(hist_values_front, bins = 1000, range=(-1, 1), color = "skyblue")
  title('Front ADCP')
  xlabel('v (trans' + str(instance[1]) + ') - v (trans' + str(instance[0])+')',fontsize=fs)
  
  # Generate text to write.
  text3 = 'mean  =  ' + str(round(mean(hist_values_front),3))+ ' m/s'
  text4 = 'std =  ' + str(round(std(hist_values_front),3))+ ' m/s'
  text =  text3 + '\n' + text4 
  annotate(text, xy=(0.08, 1), xytext=(-15, -15), fontsize=10,xycoords='axes fraction', textcoords='offset points',bbox=dict(facecolor='white', alpha=0.8),horizontalalignment='left', verticalalignment='top')
   
  subplot(nrows,ncols,nsub+4)
  hist(hist_values_back, bins = 1000, range=(-1, 1), color = "skyblue")
  title('Back ADCP')
  xlabel('v (trans' + str(instance[1]) + ') - v (trans' + str(instance[0])+')',fontsize=fs)
  
  # Generate text to write.
  text3 = 'mean =  ' + str(round(mean(hist_values_back),3))+ ' m/s'
  text4 = 'std  =  ' + str(round(std(hist_values_back),3))+ ' m/s'
  text = text3 + '\n' + text4 
  annotate(text, xy=(0.08, 1), xytext=(-15, -15), fontsize=10,xycoords='axes fraction', textcoords='offset points',bbox=dict(facecolor='white', alpha=0.8),horizontalalignment='left', verticalalignment='top') 
      
  if savefigg == 1:
    figfilename=pl_vars+'_'+str(yy)+str(mm)+str(dd)+'_'+str(instance-1)+'_'+str(instance)+'_subsequent_transects_subtraction'.zfill(2)+figtype
    savefig(figdir+'/'+figfilename,dpi=300)
    close(fignr)
  

  tight_layout()

  return
    
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
yy='2021'; mm = '07'; dd = '15'; instance_1 = 6; instance_2 = 9
instance_step=1;

# Velocity
pl_vars = 'NORTH_VEL'; bt_vars = 'BT_NORTH_VEL'

# Error threshold
error_thresh = 0.28   # mask using error velocity 0.2

# Bad_fraction
bad_fraction = 0.3      # don't plot columns with a fraction of points with error velocity over error_threshold larger than bad_fraction

# Location of the data
if yy =='2022' and mm == '06' and dd == '01':
  indir = '/home/prdusr/data_out/TESO/daily'
else: 
  indir='/home/jvandermolen/data_out/TESO/daily'
  
# max value for the colorbar 
max_ax=0.5

fs1=16; fs2=9; fs=12
dpi_setting=300
figtype='.jpg'
figdir='/home/akaraoli/python_scripts/figures/'
savefigg = 0


############ Main ######################################################

# Uncomment the function that you want to use


# -------- Plot the ship's velocity for three subsequent transects ------------------------
ferry_speed_subsequent_transects(indir,names,yy,mm,dd,instance_1, instance_2, error_thresh, pl_vars, bt_vars)


'''
# --------Plot the ship's velocity for two no subsequent transects -----------------
ferry_speed_no_subsequent_transects(indir,names,yy,mm,dd,instance_1, instance_2, error_thresh, pl_vars, bt_vars)
'''

'''
# -------- Try to compute the backflow by subtracting two sunsequent transects -----------------
backflow(indir,names,yy,mm,dd,instance_1, instance_2, error_thresh, pl_vars, bt_vars)
'''

show()
