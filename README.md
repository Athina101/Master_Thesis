# Master Thesis

The codes for my thesis are given. Scripts analyse the velocity data exctracted by the hull mounted ADCPs. We investigate the turbulent features in the Marsdiep inlet (defined as the standard deviation of the horizontal velocity difference) and also the influence of the ferry's motion on the velocity measurements. We search for the ship-induced backflow and we correct the vertical velocity for the ferry's motion.    

**Title:** Ferry-ADCP observations of tidal currents in the Marsdiep inlet and how they are affected by the motion of the ferry

**File descriptions:**

plot_data.py: A script to plot the profiles of the three components of the raw and cleaned-up flow's velocity.

phase_difference.py: A script to plot the timeseries of the water elevation and the tidal current (phase lag between water elevation and tidal current).

turbulence.py: A script to investigate the turbulent features in the Marsdiep inlet. We investigate how turbulence depends on the strength of the tidal current and the phase of the tide and we also construct the vertical structure of the turbulence over depth.

vertical_velocity_analysis.py: A script to compute the linear decrease of the vertical velocity and its relation with the the ferry's velocity.

vertical_velocity_correction_part1.py: A script to define the function which describes the correction of the vertical velocity for the ferry's motion. Furthermore, we fit this function on the measured vertical velocity. It also gives the smoothed corrected vertical velocity.  

vertical_velocity_correction_part2.py: This script gives the three corrections of the vertical velocity and the echo intensity. 

correl_vert_vel_topography.py: A script to search for topography-induced vertical velocity.

correction_ship_induced_flow: A script to plot the timeseries of the measured eastward and the measured northward tidal current. Furthermore, we try to compute the ship-induced backflow by subtracting two subsequent velocity profiles near the flood or ebb currents.

fitting_sinusoidal_tidal_cycle.py: A script to fit the tidal cycle of the northward velocity by using harmonic analysis. Observation of a forward flow instead of a backflow.

 
