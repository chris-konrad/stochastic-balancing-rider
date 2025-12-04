# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:17:12 2024

Visualization helpers

@author: Christoph M. Konrad
"""

import numpy as np
import matplotlib.pyplot as plt
from pypaperutils.design import TUDcolors

def plot_results(exp_data_dict=None,
                 test_sim = None,
                 calib_states = None,
                 guess_states = None,
                 known_trajectory_map=None,
                 name=None,
                 note=None):
    
    fig, ax = plt.subplots(7,1, sharex=True, figsize = (10,8))
    
    title = f"Rider control identification results"
    if name is not None:
        title += f"\n{name}"
    if note is not None:
        title += f"\n{note}"
        
    ax[0].set_title(title)
    ax[0].set_ylabel('x [m]')
    ax[1].set_ylabel('y [m]')
    ax[2].set_ylabel('\psi [deg]')
    ax[3].set_ylabel('v [m/s]')
    ax[4].set_ylabel('\delta [deg]')
    ax[5].set_ylabel('\phi [deg]')
    ax[6].set_ylabel('y_c [m]')
    ax[6].set_xlabel('t [s]')
    
    maxmin = np.zeros((7,2))
    maxmin[:,0] = np.inf
    maxmin[:,1] = -np.inf
    
    def test_maxmin(data, i):
        
        datamin = np.amin(data)
        if np.isfinite(datamin):
            maxmin[i,0] = min(datamin, maxmin[i,0])
        
        datamax = np.amax(data)
        if np.isfinite(datamax):
            maxmin[i,1] = max(datamax, maxmin[i,1])
    
    if guess_states is not None:
        
        for a in ax:
            a.autoscale(False)
        
        t_s = 0.01
        t = t_s * np.arange(calib_states.shape[1])

        ax[0].plot(t, guess_states[0, :], color="gray")
        ax[1].plot(t, guess_states[1, :], color="gray")
        ax[2].plot(t, 180 * guess_states[2, :] / np.pi, color="gray")
        ax[4].plot(t, 180 * guess_states[4, :] / np.pi, color="gray")
        ax[5].plot(t, 180 * guess_states[5, :] / np.pi, label='initial guess', color="gray")
    
    if exp_data_dict is not None:
        
        
        i_finite = np.isfinite(exp_data_dict['p_x_m'])
        t = np.array(exp_data_dict['t_m'])
        
        t_finite = t[i_finite]
        
        ax[0].plot(t_finite, exp_data_dict['p_x_m'][i_finite])
        ax[1].plot(t_finite, exp_data_dict['p_y_m'][i_finite])
        ax[2].plot(t_finite, 180 * exp_data_dict['psi_m'][i_finite] / np.pi)
        ax[3].plot(t, exp_data_dict['v_m'][:t.size])
        ax[4].plot(t, 180 * exp_data_dict['delta_m'] / np.pi)
        ax[5].plot(t, 180 * exp_data_dict['phi_m'] / np.pi, 
                   label = 'measured')
        ax[6].plot(t, exp_data_dict['p_y_c_m'][:t.size])
        
        test_maxmin(exp_data_dict['p_x_m'][i_finite], 0)
        test_maxmin(exp_data_dict['p_y_m'][i_finite], 1)
        test_maxmin(180 * exp_data_dict['psi_m'][i_finite] / np.pi, 2)
        test_maxmin(exp_data_dict['v_m'][:t.size], 3)
        test_maxmin(180 * exp_data_dict['delta_m'] / np.pi, 4)
        test_maxmin(180 * exp_data_dict['phi_m'] / np.pi, 5)
        test_maxmin(exp_data_dict['p_y_c_m'][:t.size], 6)
        
    if calib_states is not None:

        t_s = 0.01
        t = t_s * np.arange(calib_states.shape[1])

        ax[0].plot(t, calib_states[0, :])
        ax[1].plot(t, calib_states[1, :])
        ax[2].plot(t, 180 * calib_states[2, :] / np.pi)
        ax[4].plot(t, 180 * calib_states[4, :] / np.pi)
        ax[5].plot(t, 180 * calib_states[5, :] / np.pi, label='calibrated')
        
        test_maxmin( calib_states[0, :], 0)
        test_maxmin( calib_states[1, :], 1)
        test_maxmin(180 * calib_states[2, :] / np.pi, 2)
        
        test_maxmin(180 * calib_states[4, :] / np.pi, 4)
        test_maxmin(180 * calib_states[5, :] / np.pi, 5)
        
        if known_trajectory_map is not None:
           ax[3].plot(t, known_trajectory_map[list(known_trajectory_map.keys())[2]]) 
           ax[6].plot(t, known_trajectory_map[list(known_trajectory_map.keys())[1]])
           test_maxmin(known_trajectory_map[list(known_trajectory_map.keys())[2]], 3)
    
    
    if test_sim is not None:
        assert not((calib_states is None) and (exp_data_dict is None)), ('One',
            'of calib_states or exp_data_dict must be not None')
        
        test_sim.bike.plot_states(axes=ax[0:6],
                                  t_end=t[-1],
                                  plot_over_time=True)
        
    ax[5].legend()
    for i in range(len(ax)):
        ax[i].set_ylim(maxmin[i,0]-.1*np.abs(maxmin[i,0]), maxmin[i,1]+.1*np.abs(maxmin[i,1]))
    ax[0].set_xlim(t[0], t[-1])    
    
    
    return fig, ax

    