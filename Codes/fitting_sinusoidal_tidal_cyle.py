#! /usr/bin/env python

# python script to plot TESO transects

# Author: Athina Karaoli, June 2022

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

def parabola_func(x, a,c):
    return a*(x)**2 + c
# ----------------------------------------------------------------------
def find_nearest(array, value):
    array = asarray(array)
    idx = (abs(array - value)).argmin()
    return array[idx]

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

def averaged_vel_for_one_transect(indir,name,yy,mm,dd,varname,instance,error_thresh):

  for name in names:
    [lat,lat_units,lat_longname]=loadvar(indir,name,yy,mm,dd,'LATITUDE',instance)
    [dep,dep_units,dep_longname]=loadvar(indir,name,yy,mm,dd,'BINDEPTH',instance)
    [siv,siv_units,siv_longname]=loadvar(indir,name,yy,mm,dd,'ERROR_VEL',instance)
    [plv,plv_units,plv_longname]=loadvar(indir,name,yy,mm,dd,varname,instance)
    [qc,qc_units,qc_longname]=loadvar(indir,name,yy,mm,dd,'General_QC',instance)
    [t,t_units,t_longname]=loadvar(indir,name,yy,mm,dd,'TIME',instance)
    [d,d_units,d_longname]=loadvar(indir,name,yy,mm,dd,'DAY',instance)
    
    if name=='hd':
      time_hd = t
    elif name=='tx':
      time_tx = t
    
    # Ferry's direction
    if (lat[100]-lat[200]) > 0:
      ar_hd='right'; ar_tx='right'
    else:
      ar_hd='left'; ar_tx='left'

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

  for b in range(len(depax)):
    for c in range(len(lat_tx)):
      if depax[b] < depth_min or depax[b] > depth_max:
        vel_tx[b,:] = nan
        if lat_tx[c] < lat_min or lat_tx[c] > lat_max:
          vel_tx[b,c] = nan
      if lat_tx[c] < lat_min or lat_tx[c] > lat_max:
          vel_tx[b,c] = nan
 
            
  #calculate the mean and the std deviation for each ADCP
  tot_vel_hd =[]; tot_vel_tx =[]
  for i in range(len(depax)):
    for j in range(limit_hd):
      v_hd = vel_hd[i,j] 
      bo_hd = ~isnan(v_hd)
      if  bo_hd == True: 
        tot_vel_hd.append(v_hd)
    for j in range(limit_tx):
      v_tx = vel_tx[i,j] 
      bo_tx = ~isnan(v_tx)
      if  bo_tx == True: 
        tot_vel_tx.append(v_tx)
  
  mean_val_hd = mean(tot_vel_hd); mean_val_tx = mean(tot_vel_tx)

  # Time of the first measurements
  t_hd = datetime(int(yy),int(mm),int(dd),0,0,0) + timedelta(seconds=int(time_hd[0])+3600)
  t_tx = datetime(int(yy),int(mm),int(dd),0,0,0) + timedelta(seconds=int(time_tx[0])+3600)
  

  return mean_val_hd, mean_val_tx, t_hd, t_tx, plv_units, ar_hd, ar_tx
#-----------------------------------------------------------------------

def mean_values(yy,mm,dd,instance_1,instance_2,error_thresh,pl_var):
  
  mean_front = []; mean_back =[]; time_front =[]; time_back =[]; distr = []; arrow_front = []; arrow_back = []
  
  for instance in range(instance_1,instance_2,instance_step):
    m_hd, m_tx, t_hd, t_tx, plv_units, ar_hd, ar_tx= averaged_vel_for_one_transect(indir,names,yy,mm,dd,pl_vars,instance,error_thresh) 

    if ar_hd == 'right':
      mean_front.append(m_hd); mean_back.append(m_tx); time_front.append(t_hd); time_back.append(t_tx); arrow_front.append(ar_hd); arrow_back.append(ar_tx)  
    if ar_hd == 'left':
      mean_front.append(m_tx); mean_back.append(m_hd); time_front.append(t_tx); time_back.append(t_hd); arrow_front.append(ar_tx); arrow_back.append(ar_hd)
       
  return mean_front, mean_back, time_front, time_back, plv_units, arrow_front, arrow_back

# -----------------------------------------------------------------------------------

def velocity_timeseries(fignr,yy, mm, days, instance_1,instance_2,instance_step,bad_fraction,pl_var):
  
  figure(fignr,(22,7),None,facecolor='w',edgecolor='k')
  
  indir = '/home/jvandermolen/data_out/TESO/daily'
  
  for dd in days:
    mean_hd = []; mean_tx = []; time_hd = []; time_tx = []; ferry_dir_hd = []; ferry_dir_tx = []
    m_hd, m_tx, t_hd, t_tx, plv_units, arrow_hd, arrow_tx= mean_values(yy,mm,dd,instance_1,instance_2,error_thresh,pl_vars) 
    mean_hd.extend(m_hd); mean_tx.extend(m_tx); time_hd.extend(t_hd); time_tx.extend(t_tx); ferry_dir_hd.extend(arrow_hd); ferry_dir_tx.extend(arrow_tx)
    print(dd)
   
    # ---------------------------------------- hd ------------------------------------------
    subplot(1,2,1)
    # Denote the two crossings with different colors 
    for n in range(0,32):
      if ferry_dir_hd[n] == 'right':
        m_hd = 'o'; c_hd = 'royalblue'
      else:
        m_hd = 'o'; c_hd = 'salmon'
      if ferry_dir_tx[n] == 'right':
        m_tx = 'o'; c_tx = 'royalblue'
      else:
        m_tx = 'o'; c_tx = 'salmon'
      if n == 0:
        la = 'tx $\longrightarrow$ hd'
      elif n == 1:
        la = 'tx $\longleftarrow$ hd'
      else:
        la = None
        
      t =  int((time_hd[n] - datetime(int(yy),int(mm),int(days[0]),0,0,0)).total_seconds())
      scatter(n, mean_hd[n], marker = m_hd, c = c_hd, linewidths = 1.5, label = la) #3.5

    plot(arange(0,32,1), mean_hd, 'k', alpha=0.5)
    title('hd ADCP ',fontsize=17)
    legend(fontsize=14)  
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    legend(fontsize=13)
    grid()
    
    if pl_vars=='EAST_VEL':
      xlabel('Number of transect',fontsize=17)
      ylabel('Mean eastward velocity ['+plv_units+']',fontsize=17)
      
      major_ticks_x = np.arange(0,32, 5)
      minor_ticks_x = np.arange(0, 32, 1)
      major_ticks_y = np.arange(-1, 1.05, 0.25)
      minor_ticks_y = np.arange(-1, 1.05, 0.05)  
      ax = plt.gca()
      ax.set_xticks(major_ticks_x)
      ax.set_xticks(minor_ticks_x, minor=True)
      ax.set_yticks(major_ticks_y)
      ax.set_yticks(minor_ticks_y, minor=True)
      ax.grid(which='both')
      ax.grid(which='minor', alpha=0.3)
      ax.grid(which='major', alpha=0.6)
      
    elif pl_vars=='NORTH_VEL':
      xlabel('Number of transect',fontsize=17)
      ylabel('Mean northward velocity ['+plv_units+']',fontsize=17)
    
      major_ticks_x = np.arange(0,32, 5)
      minor_ticks_x = np.arange(0, 32, 1)
      major_ticks_y = np.arange(-0.6, 0.65, 0.2)
      minor_ticks_y = np.arange(-0.6, 0.65, 0.05)  
      ax = plt.gca()
      ax.set_xticks(major_ticks_x)
      ax.set_xticks(minor_ticks_x, minor=True)
      ax.set_yticks(major_ticks_y)
      ax.set_yticks(minor_ticks_y, minor=True)
      ax.grid(which='both')
      ax.grid(which='minor', alpha=0.3)
      ax.grid(which='major', alpha=0.6)

    # ---------------------------------------- tx ------------------------------------------
    subplot(1,2,2)
    # Denote the two crossings with different colors 
    for n in range(0,32):
      if ferry_dir_hd[n] == 'right':
        m_hd = 'o'
        c_hd = 'royalblue'
      else:
        m_hd = 'o'
        c_hd = 'salmon'
      if ferry_dir_tx[n] == 'right':
        m_tx = 'o'
        c_tx = 'royalblue'
      else:
        m_tx = 'o'
        c_tx = 'salmon'
      if n == 0:
        la = 'tx $\longrightarrow$ hd'
      elif n == 1:
        la = 'tx $\longleftarrow$ hd'
      else:
        la = None

      t = int((time_hd[n] - datetime(int(yy),int(mm),int(days[0]),0,0,0)).total_seconds())
      scatter(n, mean_tx[n], marker = m_tx, c = c_tx, linewidths = 1.5, label = la)  

    plot(arange(0,32,1), mean_tx, 'k', alpha=0.5)
    title('tx ADCP',fontsize=17)
    legend(fontsize=13)  
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    grid()
    
    if pl_vars=='EAST_VEL':
      xlabel('Number of transect',fontsize=17)
      ylabel('Mean eastward velocity ['+plv_units+']',fontsize=17)
      
      major_ticks_x = np.arange(0,32, 5)
      minor_ticks_x = np.arange(0, 32, 1)
      major_ticks_y = np.arange(-1, 1.05, 0.25)
      minor_ticks_y = np.arange(-1, 1.05, 0.05)  
      ax = plt.gca()
      ax.set_xticks(major_ticks_x)
      ax.set_xticks(minor_ticks_x, minor=True)
      ax.set_yticks(major_ticks_y)
      ax.set_yticks(minor_ticks_y, minor=True)
      ax.grid(which='both')
      ax.grid(which='minor', alpha=0.3)
      ax.grid(which='major', alpha=0.6)
      
    elif pl_vars=='NORTH_VEL':
      xlabel('Number of transect',fontsize=17)
      ylabel('Mean northward velocity ['+plv_units+']',fontsize=17)
      
      major_ticks_x = np.arange(0,32, 5)
      minor_ticks_x = np.arange(0, 32, 1)
      major_ticks_y = np.arange(-0.6, 0.65, 0.2)
      minor_ticks_y = np.arange(-0.6, 0.65, 0.05)  
      ax = plt.gca()
      ax.set_xticks(major_ticks_x)
      ax.set_xticks(minor_ticks_x, minor=True)
      ax.set_yticks(major_ticks_y)
      ax.set_yticks(minor_ticks_y, minor=True)
      ax.grid(which='both')
      ax.grid(which='minor', alpha=0.3)
      ax.grid(which='major', alpha=0.6)

    
    if savefigg == 1:
      figfilename=pl_var+'_'+str(yy)+str(mm)+str(dd)+'_timeseries_mean_value_hd_tx'.zfill(2)+figtype
      savefig(figdir+'/'+figfilename,dpi=300)
      close(fignr)  
    
  return 
  
  
# -----------------------------------------------------------------------------------
def define_the_coefficients(t, vel):

  T1 = 0.518*24*3600 # period of M2 tide in seconds 
  omega1  = 2*pi/T1; omega2  = 2*2*pi/T1; omega3  = 3*2*pi/T1; omega4  = 4*2*pi/T1

  # Define the matrices for the harmonic analysis 
  S0=0; S1=0; S2=0; S3=0; S4=0; S5=0; S6=0; S7=0; S8=0;
  A0=0; A1=0; A2=0; A3=0; A4=0; A5=0; A6=0; A7=0; A8=0;  
  B0=0; B1=0; B2=0; B3=0; B4=0; B5=0; B6=0; B7=0; B8=0; 
  C0=0; C1=0; C2=0; C3=0; C4=0; C5=0; C6=0; C7=0; C8=0;
  D0=0; D1=0; D2=0; D3=0; D4=0; D5=0; D6=0; D7=0; D8=0;  
  E0=0; E1=0; E2=0; E3=0; E4=0; E5=0; E6=0; E7=0; E8=0;
  F0=0; F1=0; F2=0; F3=0; F4=0; F5=0; F6=0; F7=0; F8=0;  
  G0=0; G1=0; G2=0; G3=0; G4=0; G5=0; G6=0; G7=0; G8=0;
  H0=0; H1=0; H2=0; H3=0; H4=0; H5=0; H6=0; H7=0; H8=0;
  I0=0; I1=0; I2=0; I3=0; I4=0; I5=0; I6=0; I7=0; I8=0;
  
  for m in range(len(t)):
  
    S0 = 32; 
    valueS1 = cos(omega1*t[m]); S1 = S1 + valueS1
    valueS2 = sin(omega1*t[m]); S2 = S2 + valueS2
    valueS3 = cos(omega2*t[m]); S3 = S3 + valueS3
    valueS4 = sin(omega2*t[m]); S4 = S4 + valueS4
    valueS5 = cos(omega3*t[m]); S5 = S5 + valueS5
    valueS6 = sin(omega3*t[m]); S6 = S6 + valueS6
    valueS7 = cos(omega4*t[m]); S7 = S7 + valueS7
    valueS8 = sin(omega4*t[m]); S8 = S8 + valueS8
  
  
    valueA0 = cos(omega1*t[m]); A0 = A0 + valueA0
    valueA1 = cos(omega1*t[m])*cos(omega1*t[m]); A1 = A1 + valueA1
    valueA2 = cos(omega1*t[m])*sin(omega1*t[m]); A2 = A2 + valueA2
    valueA3 = cos(omega1*t[m])*cos(omega2*t[m]); A3 = A3 + valueA3
    valueA4 = cos(omega1*t[m])*sin(omega2*t[m]); A4 = A4 + valueA4
    valueA5 = cos(omega1*t[m])*cos(omega3*t[m]); A5 = A5 + valueA5
    valueA6 = cos(omega1*t[m])*sin(omega3*t[m]); A6 = A6 + valueA6
    valueA7 = cos(omega1*t[m])*cos(omega4*t[m]); A7 = A7 + valueA7
    valueA6 = cos(omega1*t[m])*sin(omega4*t[m]); A7 = A7 + valueA7
    
    
    valueB0 = sin(omega1*t[m]); B0 = B0 + valueB0
    valueB1 = sin(omega1*t[m])*cos(omega1*t[m]); B1 = B1 + valueB1
    valueB2 = sin(omega1*t[m])*sin(omega1*t[m]); B2 = B2 + valueB2
    valueB3 = sin(omega1*t[m])*cos(omega2*t[m]); B3 = B3 + valueB3
    valueB4 = sin(omega1*t[m])*sin(omega2*t[m]); B4 = B4 + valueB4
    valueB5 = sin(omega1*t[m])*cos(omega3*t[m]); B5 = B5 + valueB5
    valueB6 = sin(omega1*t[m])*sin(omega3*t[m]); B6 = B6 + valueB6
    valueB7 = sin(omega1*t[m])*cos(omega4*t[m]); B7 = B7 + valueB7
    valueB8 = sin(omega1*t[m])*sin(omega4*t[m]); B8 = B8 + valueB8
    
    
    valueC0 = cos(omega2*t[m]); C0 = C0 + valueC0
    valueC1 = cos(omega2*t[m])*cos(omega1*t[m]); C1 = C1 + valueC1
    valueC2 = cos(omega2*t[m])*sin(omega1*t[m]); C2 = C2 + valueC2
    valueC3 = cos(omega2*t[m])*cos(omega2*t[m]); C3 = C3 + valueC3
    valueC4 = cos(omega2*t[m])*sin(omega2*t[m]); C4 = C4 + valueC4
    valueC5 = cos(omega2*t[m])*cos(omega3*t[m]); C5 = C5 + valueC5
    valueC6 = cos(omega2*t[m])*sin(omega3*t[m]); C6 = C6 + valueC6
    valueC7 = cos(omega2*t[m])*cos(omega4*t[m]); C7 = C7 + valueC7
    valueC8 = cos(omega2*t[m])*sin(omega4*t[m]); C8 = C8 + valueC8
    
    
    valueD0 = sin(omega2*t[m]); D0 = D0 + valueD0
    valueD1 = sin(omega2*t[m])*cos(omega1*t[m]); D1 = D1 + valueD1
    valueD2 = sin(omega2*t[m])*sin(omega1*t[m]); D2 = D2 + valueD2
    valueD3 = sin(omega2*t[m])*cos(omega2*t[m]); D3 = D3 + valueD3
    valueD4 = sin(omega2*t[m])*sin(omega2*t[m]); D4 = D4 + valueD4
    valueD5 = sin(omega2*t[m])*cos(omega3*t[m]); D5 = D5 + valueD5
    valueD6 = sin(omega2*t[m])*sin(omega3*t[m]); D6 = D6 + valueD6
    valueD7 = sin(omega2*t[m])*cos(omega4*t[m]); D7 = D7 + valueD7
    valueD8 = sin(omega2*t[m])*sin(omega4*t[m]); D8 = D8 + valueD8
    
  
    valueF0 = cos(omega3*t[m]); F0 = F0 + valueF0
    valueF1 = cos(omega3*t[m])*cos(omega1*t[m]); F1 = F1 + valueF1
    valueF2 = cos(omega3*t[m])*sin(omega1*t[m]); F2 = F2 + valueF2
    valueF3 = cos(omega3*t[m])*cos(omega2*t[m]); F3 = F3 + valueF3
    valueF4 = cos(omega3*t[m])*sin(omega2*t[m]); F4 = F4 + valueF4
    valueF5 = cos(omega3*t[m])*cos(omega3*t[m]); F5 = F5 + valueF5
    valueF6 = cos(omega3*t[m])*sin(omega3*t[m]); F6 = F6 + valueF6
    valueF7 = cos(omega3*t[m])*cos(omega4*t[m]); F7 = F7 + valueF7
    valueF8 = cos(omega3*t[m])*sin(omega4*t[m]); F8 = F8 + valueF8
    
    
    valueG0 = sin(omega3*t[m]); G0 = G0 + valueG0
    valueG1 = sin(omega3*t[m])*cos(omega1*t[m]); G1 = G1 + valueG1
    valueG2 = sin(omega3*t[m])*sin(omega1*t[m]); G2 = G2 + valueG2
    valueG3 = sin(omega3*t[m])*cos(omega2*t[m]); G3 = G3 + valueG3
    valueG4 = sin(omega3*t[m])*sin(omega2*t[m]); G4 = G4 + valueG4
    valueG5 = sin(omega3*t[m])*cos(omega3*t[m]); G5 = G5 + valueG5
    valueG6 = sin(omega3*t[m])*sin(omega3*t[m]); G6 = G6 + valueG6
    valueG7 = sin(omega3*t[m])*cos(omega4*t[m]); G7 = G7 + valueG7
    valueG8 = sin(omega3*t[m])*sin(omega4*t[m]); G8 = G8 + valueG8
    
    
    valueH0 = cos(omega4*t[m]); H0 = H0 + valueH0
    valueH1 = cos(omega4*t[m])*cos(omega1*t[m]); H1 = H1 + valueH1
    valueH2 = cos(omega4*t[m])*sin(omega1*t[m]); H2 = H2 + valueH2
    valueH3 = cos(omega4*t[m])*cos(omega2*t[m]); H3 = H3 + valueH3
    valueH4 = cos(omega4*t[m])*sin(omega2*t[m]); H4 = H4 + valueH4
    valueH5 = cos(omega4*t[m])*cos(omega3*t[m]); H5 = H5 + valueH5
    valueH6 = cos(omega4*t[m])*sin(omega3*t[m]); H6 = H6 + valueH6
    valueH7 = cos(omega4*t[m])*cos(omega4*t[m]); H7 = H7 + valueH7
    valueH8 = cos(omega4*t[m])*sin(omega4*t[m]); H8 = H8 + valueH8
    
    
    valueI0 = sin(omega4*t[m]); I0 = I0 + valueI0
    valueI1 = sin(omega4*t[m])*cos(omega1*t[m]); I1 = I1 + valueI1
    valueI2 = sin(omega4*t[m])*sin(omega1*t[m]); I2 = I2 + valueI2
    valueI3 = sin(omega4*t[m])*cos(omega2*t[m]); I3 = I3 + valueI3
    valueI4 = sin(omega4*t[m])*sin(omega2*t[m]); I4 = I4 + valueI4
    valueI5 = sin(omega4*t[m])*cos(omega3*t[m]); I5 = I5 + valueI5
    valueI6 = sin(omega4*t[m])*sin(omega3*t[m]); I6 = I6 + valueI6
    valueI7 = sin(omega4*t[m])*cos(omega4*t[m]); I7 = I7 + valueI7
    valueI8 = sin(omega4*t[m])*sin(omega4*t[m]); I8 = I8 + valueI8
    
  
    valueE0 = vel[m]; E0 = E0 + valueE0
    valueE1 = vel[m]*cos(omega1*t[m]); E1 = E1 + valueE1
    valueE2 = vel[m]*sin(omega1*t[m]); E2 = E2 + valueE2
    valueE3 = vel[m]*cos(omega2*t[m]); E3 = E3 + valueE3
    valueE4 = vel[m]*sin(omega2*t[m]); E4 = E4 + valueE4
    valueE5 = vel[m]*cos(omega3*t[m]); E5 = E5 + valueE5
    valueE6 = vel[m]*sin(omega3*t[m]); E6 = E6 + valueE6
    valueE7 = vel[m]*cos(omega4*t[m]); E7 = E7 + valueE7
    valueE8 = vel[m]*sin(omega4*t[m]); E8 = E8 + valueE8
    
  
  matrixx = zeros([9,9])
  matrixx[0,0] = S0; matrixx[0,1] = S1; matrixx[0,2] = S2; matrixx[0,3] = S3; matrixx[0,4] = S4; matrixx[0,5] = S5; matrixx[0,6] = S6; matrixx[0,7] = S7; matrixx[0,8] = S8;
  matrixx[1,0] = A0; matrixx[1,1] = A1; matrixx[1,2] = A2; matrixx[1,3] = A3; matrixx[1,4] = A4; matrixx[1,5] = A5; matrixx[1,6] = A6; matrixx[1,7] = A7; matrixx[1,8] = A8;
  matrixx[2,0] = B0; matrixx[2,1] = B1; matrixx[2,2] = B2; matrixx[2,3] = B3; matrixx[2,4] = B4; matrixx[2,5] = B5; matrixx[2,6] = B6; matrixx[2,7] = B7; matrixx[2,8] = B8;
  matrixx[3,0] = C0; matrixx[3,1] = C1; matrixx[3,2] = C2; matrixx[3,3] = C3; matrixx[3,4] = C4; matrixx[3,5] = C5; matrixx[3,6] = C6; matrixx[3,7] = C7; matrixx[3,8] = C8;
  matrixx[4,0] = D0; matrixx[4,1] = D1; matrixx[4,2] = D2; matrixx[4,3] = D3; matrixx[4,4] = D4; matrixx[4,5] = D5; matrixx[4,6] = D6; matrixx[4,7] = D7; matrixx[4,8] = D8;
  matrixx[5,0] = F0; matrixx[5,1] = F1; matrixx[5,2] = F2; matrixx[5,3] = F3; matrixx[5,4] = F4; matrixx[5,5] = F5; matrixx[5,6] = F6; matrixx[5,7] = F7; matrixx[5,8] = F8;
  matrixx[6,0] = G0; matrixx[6,1] = G1; matrixx[6,2] = G2; matrixx[6,3] = G3; matrixx[6,4] = G4; matrixx[6,5] = G5; matrixx[6,6] = G6; matrixx[6,7] = G7; matrixx[6,8] = G8;
  matrixx[7,0] = H0; matrixx[7,1] = H1; matrixx[7,2] = H2; matrixx[7,3] = H3; matrixx[7,4] = H4; matrixx[7,5] = H5; matrixx[7,6] = H6; matrixx[7,7] = H7; matrixx[7,8] = H8;
  matrixx[8,0] = I0; matrixx[8,1] = I1; matrixx[8,2] = I2; matrixx[8,3] = I3; matrixx[8,4] = I4; matrixx[8,5] = I5; matrixx[8,6] = I6; matrixx[8,7] = I7; matrixx[8,8] = I8;
  
  data = zeros([9,1])
  data[0,0] = E0; data[1,0] = E1; data[2,0]=E2; data[3,0] = E3; data[4,0]=E4; data[5,0] = E5; data[6,0]=E6; data[7,0] = E7; data[8,0]=E8
  
  
  # Compute the matrix of the unknown coefficients 
  inv_matrixx = linalg.inv(matrixx)
  coeff = dot(inv_matrixx,data)
  alpha = coeff[0][0]; beta = coeff[1][0]; gamma = coeff[2][0]; delta = coeff[3][0]; epsilon = coeff[4][0]; zeta = coeff[5][0]; eta = coeff[6][0]; theta = coeff[7][0]; iota = coeff[8][0]
  
  return alpha, beta, gamma, delta, epsilon, zeta, eta, theta, iota
# -----------------------------------------------------------------------------------

def harmonic_analysis_of_the_northward_velocity(yy,mm,days,error_thresh, instance_1,instance_2,instance_step,bad_fraction,pl_var):
  
  T1 = 0.518*24*3600
  omega1  = 2*pi/T1; omega2  = 2*2*pi/T1; omega3  = 3*2*pi/T1; omega4  = 4*2*pi/T1
  
  for dd in days:
  
    fignr = 3
    figure(fignr,(22,7),None,facecolor='w',edgecolor='k')
    
    mean_front = []; mean_back = []; time_front = []; time_back = []; ferry_dir_front = []; ferry_dir_back = [];
    m_front, m_back, t_front, t_back, plv_units, arrow_front, arrow_back= mean_values(yy,mm,dd,instance_1,instance_2,error_thresh,pl_vars) 
    mean_front.extend(m_front); mean_back.extend(m_back); time_front.extend(t_front); time_back.extend(t_back); ferry_dir_front.extend(arrow_front); ferry_dir_back.extend(arrow_back)
    print(dd)

    t_front = []; t_back = []
    for n in range(0,32):
       t_front.append(int((time_front[n] - datetime(int(yy),int(mm),int(days[0]),0,0,0)).total_seconds()))
       t_back.append(int((time_back[n] - datetime(int(yy),int(mm),int(days[0]),0,0,0)).total_seconds()))
   
    # ---------------------------------------- front ADCP ------------------------------------------
    subplot(1,2,1)
    for n in range(0,32):
      if ferry_dir_front[n] == 'right':
        m_front = 'o'; c_front = 'royalblue'
      else:
        m_front = 'o'; c_front = 'salmon'
      if ferry_dir_back[n] == 'right':
        m_back = 'o'; c_back = 'royalblue'
      else:
        m_back = 'o'; c_back = 'salmon'
      if n == 0:
        la = 'tx $\longrightarrow$ hd'
      elif n == 1:
        la = 'tx $\longleftarrow$ hd'
      else:
        la = None
        
      t =  int((time_front[n] - datetime(int(yy),int(mm),int(days[0]),0,0,0)).total_seconds())
      scatter(t, mean_front[n], marker = m_front, c = c_front, linewidths = 1.5, label = la)
    
    # Coefficients
    alpha, beta, gamma, delta, epsilon, zeta, eta, theta, iota = define_the_coefficients(t_front, mean_front)
    alpha_front = alpha; beta_front = beta; gamma_front = gamma; delta_front = delta; epsilon_front = epsilon; zeta_front = zeta; eta_front = eta; theta_front = theta; iota_front = iota
    
    # Plot the fitted curve of the tical cycle
    x = arange(min(t_front),max(t_front),1)
    ynew = alpha + beta*cos(omega1*x) + gamma*sin(omega1*x) + delta*cos(omega2*x) + epsilon*sin(omega2*x) + zeta*cos(omega3*x) + eta*sin(omega3*x) + theta*cos(omega4*x) + iota*sin(omega4*x)
    plot(x,ynew, "k", linewidth=2, label='v = A + B cos($\omega$t) + C sin($\omega$t) + D cos(2$\omega$t) + E sin(2$\omega$t) + F cos(3$\omega$t) + G sin(3$\omega$t) + H cos(4$\omega$t) + I sin(4$\omega$t)' + '\nA = {:1.3f}'.format(alpha) + ', B = {:1.3f}'.format(beta)+', C = {:1.3f}'.format(gamma)+ ', D = {:1.3f}'.format(delta)+', E = {:1.3f}'.format(epsilon)+', F = {:1.3f}'.format(zeta)+', G = {:1.3f}'.format(eta)+', H = {:1.3f}'.format(theta)+', I = {:1.3f}'.format(iota))
    

    if pl_vars=='EAST_VEL':
      xlabel('Time since the beginning of the day [seconds]',fontsize=17)
      ylabel('Mean Eastward velocity ['+plv_units+']',fontsize=17)
    elif pl_vars=='NORTH_VEL':
      xlabel('Time since the beginning of the day [seconds]',fontsize=17)
      ylabel('Mean northward velocity ['+plv_units+']',fontsize=17)
    legend(loc='upper center', bbox_to_anchor= (0.5, -0.14),fontsize=10)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    title('Front ADCP',fontsize=17)
    lgd1 = legend(loc='upper center', bbox_to_anchor= (0.5, -0.14))
    
    major_ticks_x = np.arange(10000, 80000, 10000)
    minor_ticks_x = np.arange(10000, 80000, 2000)
    major_ticks_y = np.arange(-0.6, 0.62, 0.2)
    minor_ticks_y = np.arange(-0.6, 0.62, 0.05)  
    ax = plt.gca()
    ax.set_xticks(major_ticks_x)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(major_ticks_y)
    ax.set_yticks(minor_ticks_y, minor=True)
    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.3)
    ax.grid(which='major', alpha=0.6)
    
    
    # ----------------------------------------  rear ADCP ------------------------------------------
    subplot(1,2,2)
    for n in range(0,32):
      if ferry_dir_front[n] == 'right':
        m_front = 'o'; c_front = 'royalblue'
      else:
        m_front = 'o'; c_front = 'salmon'
      if ferry_dir_back[n] == 'right':
        m_back = 'o'; c_back = 'royalblue'
      else:
        m_back = 'o'; c_back = 'salmon'
      if n == 0:
        la = 'tx $\longrightarrow$ hd'
      elif n == 1:
        la = 'tx $\longleftarrow$ hd'
      else:
        la = None

      t = int((time_back[n] - datetime(int(yy),int(mm),int(days[0]),0,0,0)).total_seconds())
      scatter(t, mean_back[n], marker = m_back, c = c_back, linewidths = 1.5, label = la)  

    # Coefficients
    alpha, beta, gamma, delta, epsilon, zeta, eta, theta, iota = define_the_coefficients(t_back, mean_back)
    alpha_back = alpha; beta_back = beta; gamma_back = gamma; delta_back = delta; epsilon_back = epsilon; zeta_back = zeta; eta_back = eta; theta_back = theta; iota_back = iota
    
    # Plot the fitted curve of the tical cycle
    x = arange(min(t_back),max(t_back),1)
    ynew = alpha + beta*cos(omega1*x) + gamma*sin(omega1*x) + delta*cos(omega2*x) + epsilon*sin(omega2*x) + zeta*cos(omega3*x) + eta*sin(omega3*x) + theta*cos(omega4*x) + iota*sin(omega4*x)
    plot(x,ynew, "k", linewidth=2, label='v = A + B cos($\omega$t) + C sin($\omega$t) + D cos(2$\omega$t) + E sin(2$\omega$t) + F cos(3$\omega$t) + G sin(3$\omega$t) + H cos(4$\omega$t) + I sin(4$\omega$t)' + '\nA = {:1.3f}'.format(alpha) + ', B = {:1.3f}'.format(beta)+', C = {:1.3f}'.format(gamma)+ ', D = {:1.3f}'.format(delta)+', E = {:1.3f}'.format(epsilon)+', F = {:1.3f}'.format(zeta)+', G = {:1.3f}'.format(eta)+', H = {:1.3f}'.format(theta)+', I = {:1.3f}'.format(iota))
    

    if pl_vars=='EAST_VEL':
      xlabel('Time since the beginning of the day [seconds]',fontsize=13)
      ylabel('Mean Eastward velocity ['+plv_units+']',fontsize=13)
    elif pl_vars=='NORTH_VEL':
      xlabel('Time since the beginning of the day [seconds]',fontsize=13)
      ylabel('Mean northward velocity ['+plv_units+']',fontsize=13)
    title('Rear ADCP',fontsize=13)
    legend(loc='upper center', bbox_to_anchor= (0.5, -0.14),fontsize=10)
    lgd2 = legend(loc='upper center', bbox_to_anchor= (0.5, -0.14))
    
    major_ticks_x = np.arange(10000, 80000, 10000)
    minor_ticks_x = np.arange(10000, 80000, 2000)
    major_ticks_y = np.arange(-0.6, 0.62, 0.2)
    minor_ticks_y = np.arange(-0.6, 0.62, 0.05)  
    ax = plt.gca()
    ax.set_xticks(major_ticks_x)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(major_ticks_y)
    ax.set_yticks(minor_ticks_y, minor=True)
    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.3)
    ax.grid(which='major', alpha=0.6)
    
    
    if savefigg == 1:
      figfilename=pl_var+'_'+str(yy)+str(mm)+str(dd)+'_timeseries_mean_value_front_back'.zfill(2)+figtype
      savefig(figdir+'/'+figfilename, bbox_extra_artists=(lgd1,lgd2), bbox_inches='tight',dpi=300)
      #savefig(figdir+'/'+figfilename,dpi=300)
      close(fignr)
    
    
    # -------------------------------------------------------------------------------------------------------------------
    #----------------------------- Analyze the red and blue points seperately --------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    
    
    mean_front_right = []; mean_back_right = []; mean_front_left = []; mean_back_left = [];
    num_trans_front_right = [];num_trans_back_right = [];num_trans_front_left = [];num_trans_back_left= [];
    time_front_right = []; time_front_left =[]; time_back_right = []; time_back_left =[]; 
    
    for n in range(0,32):
      if ferry_dir_front[n] == 'right':
        mean_front_right.append(mean_front[n])
        num_trans_front_right.append(n)
        time_front_right.append(int((time_front[n] - datetime(int(yy),int(mm),int(days[0]),0,0,0)).total_seconds()))
      else:
        mean_front_left.append(mean_front[n])
        num_trans_front_left.append(n)
        time_front_left.append(int((time_front[n] - datetime(int(yy),int(mm),int(days[0]),0,0,0)).total_seconds()))
        
      if ferry_dir_back[n] == 'right':
        mean_back_right.append(mean_back[n])
        num_trans_back_right.append(n)
        time_back_right.append(int((time_back[n] - datetime(int(yy),int(mm),int(days[0]),0,0,0)).total_seconds()))
      else:
        mean_back_left.append(mean_back[n])
        num_trans_back_left.append(n)
        time_back_left.append(int((time_back[n] - datetime(int(yy),int(mm),int(days[0]),0,0,0)).total_seconds()))
      
   
    #--------------------------------------- Compute the difference -------------------------------------------
    
    # ------------- For the front ADCP ----------------------
    alpha_right = []
    for i in range(len(time_front_right)):
      y_front = alpha_front + beta_front*cos(omega1*time_front_right[i]) + gamma_front*sin(omega1*time_front_right[i]) + delta_front*cos(omega2*time_front_right[i]) + epsilon_front*sin(omega2*time_front_right[i]) + zeta_front*cos(omega3*time_front_right[i]) + eta_front*sin(omega3*time_front_right[i]) + theta_front*cos(omega4*time_front_right[i]) + iota_front*sin(omega4*time_front_right[i])
      alpha_right.append(mean_front_right[i] - y_front)
    mean_alpha_right_front = mean(alpha_right)
    
    alpha_left = []
    for i in range(len(time_front_left)):
      y_front = alpha_front + beta_front*cos(omega1*time_front_left[i]) + gamma_front*sin(omega1*time_front_left[i]) + delta_front*cos(omega2*time_front_left[i]) + epsilon_front*sin(omega2*time_front_left[i])+ zeta_front*cos(omega3*time_front_left[i]) + eta_front*sin(omega3*time_front_left[i]) + theta_front*cos(omega4*time_front_left[i]) + iota_front*sin(omega4*time_front_left[i])
      alpha_left.append(mean_front_left[i] - y_front)
    mean_alpha_left_front = mean(alpha_left)

    difference_front_right = alpha_right
    difference_front_left = alpha_left
    
    
   
    fignr=4  
    figure(fignr,(9,12),None,facecolor='w',edgecolor='k')
    subplot(2,1,1)
    plot(time_front_right,alpha_right,color = 'royalblue')
    plot(linspace(10000,78000,1000), mean_alpha_right_front*ones(1000), c = 'darkblue', linestyle = '--', label = 'mean offset = '+str(round(mean_alpha_right_front,3))+' m/s')
    ylim([-0.6,0.6])
    title('Front ADCP',fontsize=17)
    #xlabel('Time since the beginning of the day [seconds]',fontsize=13)
    ylabel('Deviation of the blue points [m/s]',fontsize=17)
    legend(fontsize=14)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    
    major_ticks_x = np.arange(10000, 81000, 10000)
    minor_ticks_x = np.arange(10000, 81000, 2000)
    major_ticks_y = np.arange(-0.6, 0.62, 0.2)
    minor_ticks_y = np.arange(-0.6, 0.62, 0.05)  
    ax = plt.gca()
    ax.set_xticks(major_ticks_x)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(major_ticks_y)
    ax.set_yticks(minor_ticks_y, minor=True)
    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.3)
    ax.grid(which='major', alpha=0.6)
    
    subplot(2,1,2)  
    plot(time_front_left,alpha_left,color = 'salmon')
    plot(linspace(10000,80000,1000), mean_alpha_left_front*ones(1000), c = 'darkred', linestyle = '--', label = 'mean offset = '+str(round(mean_alpha_left_front,3))+' m/s')
    ylim([-0.6,0.6])
    #title('front',fontsize=13)
    xlabel('Time since the beginning of the day [seconds]',fontsize=17)
    ylabel('Deviation of the red points [m/s]',fontsize=17)
    legend(fontsize=14)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    
    major_ticks_x = np.arange(10000, 81000, 10000)
    minor_ticks_x = np.arange(10000, 81000, 2000)
    major_ticks_y = np.arange(-0.6, 0.62, 0.2)
    minor_ticks_y = np.arange(-0.6, 0.62, 0.05)  
    ax = plt.gca()
    ax.set_xticks(major_ticks_x)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(major_ticks_y)
    ax.set_yticks(minor_ticks_y, minor=True)
    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.3)
    ax.grid(which='major', alpha=0.6)
   
    
    if savefigg == 1:
      figfilename=pl_var+'_'+str(yy)+str(mm)+str(dd)+'_difference_black_line_from_points_front'.zfill(2)+figtype
      savefig(figdir+'/'+figfilename,dpi=300)
      close(fignr)
    

    # ---------- For the rear ADCP ----------------------
    
    alpha_right = []
    for i in range(len(time_back_right)):
      y_back = alpha_back + beta_back*cos(omega1*time_back_right[i]) + gamma_back*sin(omega1*time_back_right[i]) + delta_back*cos(omega2*time_back_right[i]) + epsilon_back*sin(omega2*time_back_right[i]) + zeta_back*cos(omega3*time_back_right[i]) + eta_back*sin(omega3*time_back_right[i]) + theta_back*cos(omega4*time_back_right[i]) + iota_back*sin(omega4*time_back_right[i])
      alpha_right.append(mean_back_right[i] - y_back)
    mean_alpha_right_back = mean(alpha_right)
    
    alpha_left = []
    for i in range(len(time_back_left)):
      y_back = alpha_back + beta_back*cos(omega1*time_back_left[i]) + gamma_back*sin(omega1*time_back_left[i]) + delta_back*cos(omega2*time_back_left[i]) + epsilon_back*sin(omega2*time_back_left[i]) + zeta_back*cos(omega3*time_back_left[i]) + eta_back*sin(omega3*time_back_left[i]) + theta_back*cos(omega4*time_back_left[i]) + iota_back*sin(omega4*time_back_left[i])
      alpha_left.append(mean_back_left[i] - y_back)
    mean_alpha_left_back = mean(alpha_left)

    difference_back_right = alpha_right
    difference_back_left = alpha_left
   
    fignr=5  
    figure(fignr,(9,12),None,facecolor='w',edgecolor='k')
    subplot(2,1,1)
    plot(time_back_right,alpha_right,color = 'royalblue')
    plot(linspace(10000,80000,1000), mean_alpha_right_back*ones(1000), c = 'darkblue', linestyle = '--', label = 'mean value = '+str(round(mean_alpha_right_back,3))+' m/s')
    ylim([-0.6,0.6])
    title('Rear ADCP',fontsize=17)
    #xlabel('Time since the beginning of the day [seconds]',fontsize=13)
    ylabel('Deviation of the blue points [m/s]',fontsize=17)
    legend(fontsize=14)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    
    major_ticks_x = np.arange(10000, 81000, 10000)
    minor_ticks_x = np.arange(10000, 81000, 2000)
    major_ticks_y = np.arange(-0.6, 0.62, 0.2)
    minor_ticks_y = np.arange(-0.6, 0.62, 0.05)  
    ax = plt.gca()
    ax.set_xticks(major_ticks_x)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(major_ticks_y)
    ax.set_yticks(minor_ticks_y, minor=True)
    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.3)
    ax.grid(which='major', alpha=0.6)
    
    subplot(2,1,2)  
    plot(time_back_left,alpha_left,color = 'salmon')
    plot(linspace(10000,80000,1000), mean_alpha_left_back*ones(1000), c = 'darkred', linestyle = '--', label = 'mean value = '+str(round(mean_alpha_left_back,3))+' m/s')
    ylim([-0.6,0.6])
    xlabel('Time since the beginning of the day [seconds]',fontsize=17)
    ylabel('Deviation of the red points [m/s]',fontsize=17)
    legend(fontsize=14)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    
    major_ticks_x = np.arange(10000, 81000, 10000)
    minor_ticks_x = np.arange(10000, 81000, 2000)
    major_ticks_y = np.arange(-0.6, 0.62, 0.2)
    minor_ticks_y = np.arange(-0.6, 0.62, 0.05)  
    ax = plt.gca()
    ax.set_xticks(major_ticks_x)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(major_ticks_y)
    ax.set_yticks(minor_ticks_y, minor=True)
    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.3)
    ax.grid(which='major', alpha=0.6)
    
    if savefigg == 1:
      figfilename=pl_var+'_'+str(yy)+str(mm)+str(dd)+'_difference_black_line_from_points_back'.zfill(2)+figtype
      savefig(figdir+'/'+figfilename,dpi=300)
      close(fignr)
    
    
    # -------------------------------------------------------------------------------------------------------------------
    #----------------------------- Plot the sinusoidal curves for each crossing separately --------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    
    
    # ----------------For the front ADCP -------------------------------
    
    fignr=6  
    figure(fignr,(22,7),None,facecolor='w',edgecolor='k')
    subplot(1,2,1)
    
    x = arange(min(t_front),max(t_front),1)
    
    # For the crossing tx -> hd
    ynew = mean_alpha_right_front + alpha_front + beta_front*cos(omega1*x) + gamma_front*sin(omega1*x) + delta_front*cos(omega2*x) + epsilon_front*sin(omega2*x) + zeta_front*cos(omega3*x) + eta_front*sin(omega3*x) + theta_front*cos(omega4*x) + iota_front*sin(omega4*x)
    scatter(time_front_right, mean_front_right, marker = 'o', color='royalblue',linewidths = 3.5, label = 'tx $\longrightarrow$ hd')
    plot(x,ynew, "darkblue", linewidth=2, label='v = A + B cos($\omega$t) + C sin($\omega$t) + D cos(2$\omega$t) + E sin(2$\omega$t) + F cos(3$\omega$t) + G sin(3$\omega$t) + H cos(4$\omega$t) + I sin(4$\omega$t)' + '\nA = {:1.3f}'.format(mean_alpha_right_front + alpha_front) + ', B = {:1.3f}'.format(beta_front)+', C = {:1.3f}'.format(gamma_front)+ ', D = {:1.3f}'.format(delta_front)+', E = {:1.3f}'.format(epsilon_front)+', F = {:1.3f}'.format(zeta_front)+', G = {:1.3f}'.format(eta_front)+', H = {:1.3f}'.format(theta_front)+', I = {:1.3f}'.format(iota_front))
    
    # For the crossing tx <- hd
    ynew = mean_alpha_left_front + alpha_front + beta_front*cos(omega1*x) + gamma_front*sin(omega1*x) + delta_front*cos(omega2*x) + epsilon_front*sin(omega2*x) + zeta_front*cos(omega3*x) + eta_front*sin(omega3*x) + theta_front*cos(omega4*x) + iota_front*sin(omega4*x)
    scatter(time_front_left, mean_front_left, marker = 'o', color='salmon',linewidths = 3.5, label = 'tx$\longleftarrow$ hd')
    plot(x,ynew, "red", linewidth=2, label='A = {:1.3f}'.format( mean_alpha_left_front + alpha_front ) + ', B = {:1.3f}'.format(beta_front)+', C = {:1.3f}'.format(gamma_front)+ ', D = {:1.3f}'.format(delta_front)+', E = {:1.3f}'.format(epsilon_front)+', F = {:1.3f}'.format(zeta_front)+', G = {:1.3f}'.format(eta_front)+', H = {:1.3f}'.format(theta_front)+', I = {:1.3f}'.format(iota_front))

    if pl_vars=='EAST_VEL':
      xlabel('Time since the beginning of the day [seconds]',fontsize=13)
      ylabel('Mean Eastward velocity ['+plv_units+']',fontsize=13)
    elif pl_vars=='NORTH_VEL':
      xlabel('Time since the beginning of the day [seconds]',fontsize=13)
      ylabel('Mean northward velocity ['+plv_units+']',fontsize=13)
    legend(loc='upper center', bbox_to_anchor= (0.5, -0.14),fontsize=10)
    title('Front ADCP',fontsize=13)
   
    major_ticks_x = np.arange(10000, 80000, 10000)
    minor_ticks_x = np.arange(10000, 80000, 2000)
    major_ticks_y = np.arange(-0.6, 0.62, 0.2)
    minor_ticks_y = np.arange(-0.6, 0.62, 0.05)  
    ax = plt.gca()
    ax.set_xticks(major_ticks_x)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(major_ticks_y)
    ax.set_yticks(minor_ticks_y, minor=True)
    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.3)
    ax.grid(which='major', alpha=0.6)
    
    
    # ----------------For the rear ADCP -------------------------------
    
    subplot(1,2,2)
    
    x = arange(min(t_back),max(t_back),1)
    
    # For the crossing tx -> hd
    ynew = mean_alpha_right_back + alpha_back + beta_back*cos(omega1*x) + gamma_back*sin(omega1*x) + delta_back*cos(omega2*x) + epsilon_back*sin(omega2*x)+ zeta_back*cos(omega3*x) + eta_back*sin(omega3*x) + theta_back*cos(omega4*x) + iota_back*sin(omega4*x)
    scatter(time_back_right, mean_back_right, marker = 'o', color='royalblue',linewidths = 3.5, label = 'tx $\longrightarrow$ hd')
    plot(x,ynew, "darkblue", linewidth=2, label='v = A + B cos($\omega$t) + C sin($\omega$t) + D cos(2$\omega$t) + E sin(2$\omega$t) + F cos(3$\omega$t) + G sin(3$\omega$t) + H cos(4$\omega$t) + I sin(4$\omega$t)' + '\nA = {:1.3f}'.format( mean_alpha_right_back + alpha_back) + ', B = {:1.3f}'.format(beta_back)+', C = {:1.3f}'.format(gamma_back)+ ', D = {:1.3f}'.format(delta_back)+', E = {:1.3f}'.format(epsilon_back)+', F = {:1.3f}'.format(zeta_back)+', G = {:1.3f}'.format(eta_back)+', H = {:1.3f}'.format(theta_back)+', I = {:1.3f}'.format(iota_back))
    
    # For the crossing tx <- hd
    ynew = mean_alpha_left_back + alpha_back + beta_back*cos(omega1*x) + gamma_back*sin(omega1*x) + delta_back*cos(omega2*x) + epsilon_back*sin(omega2*x) + zeta_back*cos(omega3*x) + eta_back*sin(omega3*x) + theta_back*cos(omega4*x) + iota_back*sin(omega4*x)
    scatter(time_back_left, mean_back_left, marker = 'o', color='salmon',linewidths = 3.5, label = 'tx $\longleftarrow$ hd')
    plot(x,ynew, "red", linewidth=2, label='A = {:1.3f}'.format(mean_alpha_left_back + alpha_back) + ', B = {:1.3f}'.format(beta_back)+', C = {:1.3f}'.format(gamma_back)+ ', D = {:1.3f}'.format(delta_back)+', E = {:1.3f}'.format(epsilon_back)+', F = {:1.3f}'.format(zeta_back)+', G = {:1.3f}'.format(eta_back)+', H = {:1.3f}'.format(theta_back)+', I = {:1.3f}'.format(iota_back))

    if pl_vars=='EAST_VEL':
      xlabel('Time since the beginning of the day [seconds]',fontsize=13)
      ylabel('Mean Eastward velocity ['+plv_units+']',fontsize=13)
    elif pl_vars=='NORTH_VEL':
      xlabel('Time since the beginning of the day [seconds]',fontsize=13)
      ylabel('Mean northward velocity ['+plv_units+']',fontsize=13)
    legend(loc='upper center', bbox_to_anchor= (0.5, -0.14),fontsize=10)
    title('Rear ADCP',fontsize=13)
   
    major_ticks_x = np.arange(10000, 80000, 10000)
    minor_ticks_x = np.arange(10000, 80000, 2000)
    major_ticks_y = np.arange(-0.6, 0.6, 0.2)
    minor_ticks_y = np.arange(-0.6, 0.6, 0.05)  
    ax = plt.gca()
    ax.set_xticks(major_ticks_x)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(major_ticks_y)
    ax.set_yticks(minor_ticks_y, minor=True)
    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.3)
    ax.grid(which='major', alpha=0.6)
    
    if savefigg == 1:
      figfilename=pl_var+'_'+str(yy)+str(mm)+str(dd)+'_timeseries_subplots_front_back'.zfill(2)+figtype
      savefig(figdir+'/'+figfilename, bbox_extra_artists=(lgd1,lgd2), bbox_inches='tight',dpi=300)
      #savefig(figdir+'/'+figfilename,dpi=300)
      close(fignr)
    
  return mean_alpha_right_front, mean_alpha_left_front, mean_alpha_right_back, mean_alpha_left_back
  
  
  
#-----------------------------------------------------------------------

def plot_offset_for_many_days(yy,mm,days,error_thresh,instance_1,instance_2,instance_step,bad_fraction,pl_vars):  

  diff_right_front = []; diff_right_back = []; diff_left_front = []; diff_left_back = []; days_ar = []
  
  T1 = 0.518*24*3600
  omega1  = 2*pi/T1; omega2  = 2*2*pi/T1; omega3  = 3*2*pi/T1; omega4  = 4*2*pi/T1
  
  for dd in days:
  
    mean_front = []; mean_back = []; time_front = []; time_back = []; ferry_dir_front = []; ferry_dir_back = [];
    m_front, m_back, t_front, t_back, plv_units, arrow_front, arrow_back= mean_values(yy,mm,dd,instance_1,instance_2,error_thresh,pl_vars) 
    mean_front.extend(m_front); mean_back.extend(m_back); time_front.extend(t_front); time_back.extend(t_back); ferry_dir_front.extend(arrow_front); ferry_dir_back.extend(arrow_back)
    print(dd)

    t_front = []; t_back = []
    for n in range(0,32):
       t_front.append(int((time_front[n] - datetime(int(yy),int(mm),int(days[0]),0,0,0)).total_seconds()))
       t_back.append(int((time_back[n] - datetime(int(yy),int(mm),int(days[0]),0,0,0)).total_seconds()))
   
    alpha_front, beta_front, gamma_front, delta_front, epsilon_front, zeta_front, eta_front, theta_front, iota_front = define_the_coefficients(t_front, mean_front)
    alpha_back, beta_back, gamma_back, delta_back, epsilon_back, zeta_back, eta_back, theta_back, iota_back = define_the_coefficients(t_back, mean_back)
    
   
    
    # -------------------------------------------------------------------------------------------------------------------
    #----------------------------- Analyze the red and blue points seperately --------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    
    
    mean_front_right = []; mean_back_right = []; mean_front_left = []; mean_back_left = []; time_front_right = []; time_front_left =[]; time_back_right = []; time_back_left =[]; 
    
    for n in range(0,32):
      if ferry_dir_front[n] == 'right':
        mean_front_right.append(mean_front[n])
        time_front_right.append(int((time_front[n] - datetime(int(yy),int(mm),int(days[0]),0,0,0)).total_seconds()))
      else:
        mean_front_left.append(mean_front[n])
        time_front_left.append(int((time_front[n] - datetime(int(yy),int(mm),int(days[0]),0,0,0)).total_seconds()))
        
      if ferry_dir_back[n] == 'right':
        mean_back_right.append(mean_back[n])
        time_back_right.append(int((time_back[n] - datetime(int(yy),int(mm),int(days[0]),0,0,0)).total_seconds()))
      else:
        mean_back_left.append(mean_back[n])
        time_back_left.append(int((time_back[n] - datetime(int(yy),int(mm),int(days[0]),0,0,0)).total_seconds()))
      
   
    #--------------------------------------- Compute the difference -------------------------------------------
    
    # ------------- For the front ADCP ----------------------
    alpha_right = []
    for i in range(len(time_front_right)):
      y_front = alpha_front + beta_front*cos(omega1*time_front_right[i]) + gamma_front*sin(omega1*time_front_right[i]) + delta_front*cos(omega2*time_front_right[i]) + epsilon_front*sin(omega2*time_front_right[i]) + zeta_front*cos(omega3*time_front_right[i]) + eta_front*sin(omega3*time_front_right[i]) + theta_front*cos(omega4*time_front_right[i]) + iota_front*sin(omega4*time_front_right[i])
      alpha_right.append(mean_front_right[i] - y_front)
    mean_alpha_right_front = mean(alpha_right)
    
    alpha_left = []
    for i in range(len(time_front_left)):
      y_front = alpha_front + beta_front*cos(omega1*time_front_left[i]) + gamma_front*sin(omega1*time_front_left[i]) + delta_front*cos(omega2*time_front_left[i]) + epsilon_front*sin(omega2*time_front_left[i])+ zeta_front*cos(omega3*time_front_left[i]) + eta_front*sin(omega3*time_front_left[i]) + theta_front*cos(omega4*time_front_left[i]) + iota_front*sin(omega4*time_front_left[i])
      alpha_left.append(mean_front_left[i] - y_front)
    mean_alpha_left_front = mean(alpha_left)
    
    # ---------- For the rear ADCP ----------------------
    
    alpha_right = []
    for i in range(len(time_back_right)):
      y_back = alpha_back + beta_back*cos(omega1*time_back_right[i]) + gamma_back*sin(omega1*time_back_right[i]) + delta_back*cos(omega2*time_back_right[i]) + epsilon_back*sin(omega2*time_back_right[i]) + zeta_back*cos(omega3*time_back_right[i]) + eta_back*sin(omega3*time_back_right[i]) + theta_back*cos(omega4*time_back_right[i]) + iota_back*sin(omega4*time_back_right[i])
      alpha_right.append(mean_back_right[i] - y_back)
    mean_alpha_right_back = mean(alpha_right)
    
    alpha_left = []
    for i in range(len(time_back_left)):
      y_back = alpha_back + beta_back*cos(omega1*time_back_left[i]) + gamma_back*sin(omega1*time_back_left[i]) + delta_back*cos(omega2*time_back_left[i]) + epsilon_back*sin(omega2*time_back_left[i]) + zeta_back*cos(omega3*time_back_left[i]) + eta_back*sin(omega3*time_back_left[i]) + theta_back*cos(omega4*time_back_left[i]) + iota_back*sin(omega4*time_back_left[i])
      alpha_left.append(mean_back_left[i] - y_back)
    mean_alpha_left_back = mean(alpha_left)

    # Save the deviation of blue and red points for each day in a list
    diff_right_front.append(mean_alpha_right_front)
    diff_right_back.append(mean_alpha_right_back)
    diff_left_front.append(mean_alpha_left_front)
    diff_left_back.append(mean_alpha_left_back)
    days_ar.append(dd)
    
  
  fignr=7 
  figure(fignr,(20,7),None,facecolor='w',edgecolor='k')
  subplot(1,2,1)
  scatter(array(days), array(diff_right_front), s = 50,  color = 'mediumblue', label = '(tx $\longrightarrow$ hd)')
  scatter(array(days), array(diff_left_front), s = 50, color = 'firebrick', label = '(tx $\longleftarrow$ hd)')
  xlabel('days', fontsize = 15)
  ylabel('mean offset [m/s]', fontsize = 15)
  title('Front ADCP', fontsize = 15)
  ylim([-0.5,0.5])
  legend(fontsize = 14)
  plt.yticks(fontsize=15)
  plt.xticks(fontsize=13)
  grid()
  
  subplot(1,2,2)
  scatter(days, diff_right_back, s = 50, color = 'mediumblue', label = '(tx $\longrightarrow$ hd)')
  scatter(days, diff_left_back, s = 50, color = 'firebrick', label = '(tx $\longleftarrow$ hd)')
  xlabel('days', fontsize = 15)
  #ylabel('mean offset')
  title('Rear ADCP', fontsize = 15)
  ylim([-0.5,0.5])
  legend(fontsize = 14)
  plt.yticks(fontsize=15)
  plt.xticks(fontsize=13)
  grid()
  
  if savefigg == 1:
    figfilename=pl_vars+'_'+str(yy)+str(mm)+str(days)+'_offset_VS_days'.zfill(2)+figtype
    savefig(figdir+'/'+figfilename,dpi=300)
    close(fignr)
  
  
  return
  
############################### Main ##################################################3
indir='/home/jvandermolen/data_out/TESO/daily'
names=['hd','tx']
figdir='/home/akaraoli/python_scripts/figures/'
figtype='.jpg'
instance_1 = 0; instance_2 = 32; instance_step=1; bad_fraction=0.3
savefigg = 0 # 1: save figure, 0: not save the figure


# Date
yy='2021'; mm = '07'; dd = ['12'];  error_thresh = 0.22

#------------------------ Timeseries of the eastward velocity ---------------------------
pl_vars='EAST_VEL'; fignr = 1
velocity_timeseries(fignr,yy, mm, dd, instance_1,instance_2,instance_step,bad_fraction,pl_vars)

# ----------------------- Timeseries of the northward velocity ---------------------------
pl_vars='NORTH_VEL'; fignr = 2
velocity_timeseries(fignr,yy, mm, dd, instance_1,instance_2,instance_step,bad_fraction,pl_vars)


# ------------------------ Harmonic avalysis for the northward velocity ------------------------
pl_vars='NORTH_VEL'
harmonic_analysis_of_the_northward_velocity(yy,mm,dd,error_thresh, instance_1,instance_2,instance_step,bad_fraction,pl_vars) 


# --------------------Check the persistence of the forward flow over the time -------------------
yy='2021'; mm = '07'; dd = ['01','02','03','05','06','07','08','09','10','12','13','14','15','16','17', '19','20','21','22','23','24','26','27','28','29','30'];  error_thresh = 0.22
pl_vars='NORTH_VEL'
plot_offset_for_many_days(yy,mm,dd,error_thresh,instance_1,instance_2,instance_step,bad_fraction,pl_vars)

show()