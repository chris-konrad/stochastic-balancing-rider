# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 17:14:11 2025

Unscented Kalman Filter for fusing sensor data and smoothing measurements.

@author: Christoph M. Konrad
"""

import numpy as np
import matplotlib.pyplot as plt

from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from cyclistsocialforce.vehicle import BalancingRiderBicycle


def move_dynamic(x, t_s, bike, integration_method='euler'):
    """
    Predict the next step of the dynamic whipple-carvallo bicycle model
    assuming zero steer and roll torque. 
    
    The state vector is [p_x, p_y, psi, v, phi, delta, dpsi, dphi, ddelta, dv].
    
    Uses euler integration of the state-space formulation.

    Parameters
    ----------
    x : array-like
        Current state vector.
    t_s : float
        Time step.
    integration_method: str, optional
        The integration method for the integration of the system dynamics in 
        the prediction step. Can be 'backward euler' or 'midpoint'. The default
        is 'backward euler'.

    Returns
    -------
    x_pred : array-like
        Predicted state vector.

    """
    #extract for readibility
    p_x = x[0]
    p_y = x[1]
    psi = x[2]
    v = x[3]
    phi = x[4]
    delta = x[5]
    dpsi = x[6]
    dphi = x[7]
    ddelta = x[8]
    dv = x[9]
    
    #bicycle lateral dynamics
    x_lat = np.array([phi, delta, dphi, ddelta, psi])
    
    if integration_method == 'euler':
        
        #constant acceleration
        v_pred = dv * t_s + v
        dv_pred = dv

        #bicycle lateral dynamics
        A, B, C, D = bike.dynamics.get_statespace_matrices(v)
        x_lat_pred = x_lat + t_s * A @ x_lat
        A_pred, B, C, D = bike.dynamics.get_statespace_matrices(v_pred)
        dpsi_pred = (A_pred @ x_lat_pred)[-1]
        
        # forward dynamics
        p_x_pred =  v * np.cos(psi) * t_s + p_x
        p_y_pred =  v * np.sin(psi) * t_s + p_y
        
        #pack
        x_pred = [p_x_pred, p_y_pred, x_lat_pred[4], v_pred, 
                  x_lat_pred[0], x_lat_pred[1], dpsi_pred,
                  x_lat_pred[2], x_lat_pred[3], dv_pred]
        
    elif integration_method == 'midpoint':
        
        #constant acceleration
        v_pred = dv * t_s + v
        v_pred_h2 = dv * t_s / 2 + v
        dv_pred = dv
        
        #bicycle lateral dynamics
        A_h2, B, C, D = bike.dynamics.get_statespace_matrices(v_pred_h2)
        I = np.eye(x_lat.size)        
        x_lat_pred = np.linalg.inv(I - t_s/2 * A_h2) @ (I + t_s/2 * A_h2) @ x_lat
        psi_pred = x_lat_pred[-1]
        
        A_h, B, C, D = bike.dynamics.get_statespace_matrices(v_pred)
        dpsi_pred = (A_h @ x_lat_pred)[-1]

        # forward dynamics
        p_x_pred = ((v_pred + v) / 2) * np.cos((psi_pred + psi) / 2) * t_s + p_x
        p_y_pred = ((v_pred + v) / 2) * np.sin((psi_pred + psi) / 2) * t_s + p_y 
        
        x_pred = [p_x_pred, p_y_pred, x_lat_pred[4], v_pred, 
                  x_lat_pred[0], x_lat_pred[1], dpsi_pred,
                  x_lat_pred[2], x_lat_pred[3], dv_pred]
        
    else:
        raise ValueError("Invalid integration method!")
    
    return x_pred

def move_kinematic(x, t_s, dv=None, dpsi = None):
    """
    Kinematic constant acceleration and yaw rate model.
    
    The state vector is [p_x, p_y, psi, dpsi, v].

    Parameters
    ----------
    x : array-like
        Current state vector.
    t_s : float
        Time step.
    dv : float (optional)
        Acceleration input
    dpsi : float (optional)
        Yaw rate input
        
    Returns
    -------
    x_pred : array-like
        Predicted state vector.
    """
    
    #extract for readibility
    p_x = x[0]
    p_y = x[1]
    psi = x[2]
    v = x[4]
    
    if dv is None:
        dv = x[5]
    if dpsi is None:
        dpsi = x[3]
    
    #predict
    p_x_pred = v * np.cos(psi) * t_s + p_x
    p_y_pred = v * np.sin(psi) * t_s + p_y
             
    psi_pred = dpsi * t_s + psi
    dpsi_pred = dpsi 

    v_pred = dv * t_s + v
    
    dv_pred = dv 
    
    #pack
    x_pred = [p_x_pred, p_y_pred, psi_pred, dpsi_pred, v_pred, dv_pred]
    x_pred = [float(x_i) for x_i in x_pred]
    
    return x_pred

def measure_dynamic(x, gnss_available):
    """
    Extract the measured states of the dynamic whipple-carvallo model.These are
    - p_x: Measured by GNSS
    - p_y: Measured by GNSS
    - psi: Derived from GNSS x and y
    - dpsi: Yaw rate from IMU
    - v: speed from IMU
    - dv: acceleration from IMU
    - delta: steer angle from the steer encoder
    - ddelta: steer rate from the steer encoder
    - phi: roll angle from the IMU
    - dphi: roll rate from the IMU
    
    The state vector is [p_x, p_y, psi, v, phi, delta, dpsi, dphi, ddelta, dv].

    Parameters
    ----------
    x : array-like
        State vector.
    gnss_available : bool 
        Flag indicating if GNSS is available at the current time step. 

    """
    
    C = np.eye(10)
    if not gnss_available:
        C[:3,:3] = np.zeros((3,3))
    return (C @ x).flatten()

def measure_kinematic(x, gnss_available):
    """
    Extract the measured states. These are
    - p_x: Measured by GNSS
    - p_y: Measured by GNSS
    - psi: Derived from GNSS p_x and p_y
    - dpsi: Yaw rate from IMU
    - v: speed from IMU
    - dv: acceleration from IMU

    Parameters
    ----------
    x : array-like
        State vector [p_x, p_y, psi, dpsi, v].
    gnss_available : bool 
        Flag indicating if GNSS is available at the current time step. 

    """
    C = np.eye(6)
    if not gnss_available:
        C[:3,:3] = np.zeros(3)
    return (C @ x).flatten()


def filter_dynamic(measurements, R, Q, t_s=0.01, smooth = True, plot = False, 
                   integration_method = "backward euler", 
                   bicycleParameterDict=None):
    """
    Filter instrumented bicycle measurements from different sensors using
    an Unscented Kalman Filter.
    
    This requires N equally spaced samples of the following ten measurements:
        - p_x: Measured by GNSS
        - p_y: Measured by GNSS
        - psi: Derived from GNSS p_x and p_y
        - dpsi: Yaw rate from IMU
        - v: speed from IMU
        - dv: acceleration from IMU
        - delta: steer angle from the steer encoder
        - ddelta: steer rate from the steer encoder
        - phi: roll angle from the IMU
        - dphi: roll rate from the IMU
        
    NaN values indicate where the GNSS does not return a value due to lower
    sampling rate. 
        
    Based on the dynamic whipple-carvallo bicycle model with assuming zero 
    steer and roll torque.

    Parameters
    ----------
    measurements : array-like
        Measurement array of shape (N, 10).
    R : array-like
        Measurement noise matrix (10, 10).
    Q : array-like
        Process noise matrix (10, 10).
    t_s : float, optional
        Sample time. The default is 0.01.
    smooth : bool, optional
        Optionally smooth the filter result with a Rauch-Tung-Striebel smoother
        (recommended). The default is True.
    plot : bool, optional
        Plot the filter results. The default is False.
    integration_method: str, optional
        The integration method for the integration of the system dynamics in 
        the prediction step. Can be 'backward euler' or 'midpoint'. The default
        is 'backward euler'.
    bicycleParametersDict : dict
        A dictionary of bicycle parameters as returned by the bicycleparameters
        toolbox. Use this to customize the bike. 

    Returns
    -------
    states_filtered/smoothed : array-like
        Filtered (or smoothed) measurements (N, 10)

    """
    

    state_labels = ['p_x', 'p_y', 'psi', 'v', 'phi', 'delta', 'dpsi', 'dphi', 
                    'ddelta', 'dv']
    n_states = len(state_labels)
    n_samples = measurements.shape[0]-1
    n_measurements = n_states

    #make a time vector
    t = np.arange(0, n_samples+1) * t_s
    
    # intial conditions
    x0 = measurements[0,:]
    
    if bicycleParameterDict is None:
        bike = BalancingRiderBicycle([0,0,0,5,0,0,0,0])
    else:
        params = BalancingRiderBicycle.PARAMS_TYPE(bicycleParameterDict=
                                                    bicycleParameterDict)
        bike = BalancingRiderBicycle([0,0,0,5,0,0,0,0], params=params)
    
    # setup filter
    points = MerweScaledSigmaPoints(n_states, alpha=.1, beta=2., kappa=-1)
    
    def move(x, t_s):
        return move_dynamic(x, t_s, bike, integration_method=integration_method)

    ukf = UnscentedKalmanFilter(dim_x=n_states, dim_z=n_measurements, 
                                fx=move, hx=measure_dynamic, 
                                points=points, dt=t_s)
    
    ukf.x = x0
    ukf.P *= 0.2
    ukf.R = R
    ukf.Q = Q
    
    states_filtered = np.zeros((n_samples+1, n_states))
    states_filtered[0,:] = x0
    
    covs_filtered = np.zeros((n_samples+1, ukf.P.shape[0], ukf.P.shape[1]))
    covs_filtered[0,:,:] = ukf.P

    # filter measurements 
    for i in range(n_samples):
        #predict
        ukf.predict()
        
        #update
        m = measurements[i+1,:]

        gnss_available = np.all(np.isfinite(m))
        if not gnss_available:
            m[np.logical_not(np.isfinite(m))] = 0
            
        ukf.update(m, gnss_available = gnss_available)
        
        states_filtered[i+1,:] = ukf.x
        covs_filtered[i+1,:,:] = ukf.P

    states_filtered = np.array(states_filtered)
        
    # smooth filtered states using an Rauch-Tung-Striebel smoother. 
    if smooth:
        states_smoothed, covs_smoothed, K = ukf.rts_smoother(states_filtered, 
                                                             covs_filtered)
    
    #plot filter results
    if plot:
        
        #x-y plot
        fig0, ax0 = plt.subplots(1,1)
        ax0.set_aspect('equal')
        finite = np.logical_not(measurements[:,1]==0)
        ax0.plot(measurements[:,0][finite], measurements[:,1][finite], 
                 color='blue', label='measured')
        if smooth:
            ax0.plot(states_smoothed[:,0], states_smoothed[:,1], 
                     color = 'green', label='smoothed')
        
        #time plot
        fig1, axes1 = plt.subplots(n_states, 1, sharex=True)
        ax0.plot(states_filtered[:,0], states_filtered[:,1], 
                 color = 'orange', label='filtered')
            
        for i in range(n_states):
            axes1[i].set_ylabel(state_labels[i])
            
            m = measurements[:,i]
            finite = np.logical_not(m==0)
            axes1[i].plot(t[finite], m[finite], 
                          color = 'blue', label = 'data')
            axes1[i].plot(t, states_filtered[:,i], 
                          color = 'orange', label = 'filtered')
            if smooth:
                axes1[i].plot(t, states_smoothed[:,i], 
                              color = 'green', label = 'smoothed')
            
    if smooth:
        return states_smoothed
    else:
        return states_filtered
        

def filter_kinematic(measurements, R, Q, t_s=0.01):
    
    state_labels = ['p_x', 'p_y', 'psi', 'dpsi', 'v', 'dv']
    #noise_scale = 5 * np.array([0.01, 0.01, 0.01, 0.01, 0.1*t_s, 0.1])
    n_states = 6
    n_samples = measurements.shape[0]-1

    t = np.arange(0, (n_samples+1)*t_s, t_s)
    
    # figures 
    fig0, ax0 = plt.subplots(1,1)
    ax0.set_aspect('equal')
    finite = np.isfinite(measurements[:,0])
    ax0.plot(measurements[:,0][finite], measurements[:,1][finite], 
             color='blue', label='measured')
    
    fig1, axes1 = plt.subplots(n_states, 1, sharex=True)

    # inital state
    x0 = measurements[0,:]
    
    points = MerweScaledSigmaPoints(6, alpha=.1, beta=2., kappa=-1)

    ukf = UnscentedKalmanFilter(dim_x=6, dim_z=6, fx=move_kinematic, 
                                hx=measure_kinematic, points=points, dt=t_s)

    ukf.x = x0
    ukf.P *= 0.2
    ukf.R = R
    ukf.Q = Q        

    states_filtered = [x0]
    covs_filtered = ukf.P[np.newaxis,:,:]

    for i in range(n_samples):
        #predict
        ukf.predict()
        
        #update
        m = measurements[i+1,:]

        gnss_available = np.all(np.isfinite(m))
        if not gnss_available:
            m[np.logical_not(np.isfinite(m))] = 0
            
        ukf.update(m, gnss_available = gnss_available)
        
        states_filtered.append(ukf.x)
        covs_filtered = np.concatenate((covs_filtered, ukf.P[np.newaxis,:,:]), 
                                       axis=0)

    states_filtered = np.array(states_filtered)
    ax0.plot(states_filtered[:,0], states_filtered[:,1], 
             color = 'orange', label='filtered')
        
    for i in range(n_states):
        axes1[i].set_ylabel(state_labels[i])
        
        m = measurements[:,i]
        finite = np.logical_not(m==0)
        axes1[i].plot(t[finite], m[finite], 
                      color = 'blue', label = 'data')
        axes1[i].plot(t, states_filtered[:,i], 
                      color = 'orange', label = 'filtered')
        axes1[i].grid()
        
    #smooth

    states_smoothed, covs_smoothed, K = ukf.rts_smoother(states_filtered, 
                                                         covs_filtered)

    ax0.plot(states_smoothed[:,0], states_smoothed[:,1], 
             color = 'green', label='smoothed')
        
    for i in range(n_states):
        axes1[i].plot(t, states_smoothed[:,i], 
                      color = 'green', label = 'smoothed')