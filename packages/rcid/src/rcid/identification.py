# -*- coding: utf-8 -*-
"""
Identify rider control parameters

@author: Christoph M. Konrad
"""

# base imports
import os
import numpy as np
import sympy as sm
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import permetrics as pm
from time import strftime, time

#helpers
from mypyutils.log import logger
from mypyutils.io import write_with_recovery, fileparts
from mypyutils.misc import s2hms

# opty
from opty.direct_collocation import Problem
from opty.utils import parse_free

# cyclistsocialforce
from cyclistsocialforce.dynamics import test_stability, from_gains

# zigzag experiment imports
from rcid.utils import apply_timeshift, read_yaml
from rcid.path_manager import PathManager
from rcid.data_processing import RCIDDataManager
from rcid.visualisation import plot_results
from rcid.simulation import FixedSpeedBalancingRiderBicycle, FixedSpeedPlanarPointBicycle, FixedInputZigZagTest, create_stable_gain_sample

# -------------- CONSTANTS --------------------------------------------------------------------
T_S = 0.01

# -------------- EXCEPTIONS -------------------------------------------------------------------
class NoForwardSimulationException(Exception):
    """ Raised when a cyclistsocialforce forward simulation fails. """
    pass

class UnknownBicycleModelException(Exception):
    """ Raised when an unknown bicycle model is given. """
    pass

# -------------- Classes -------------------------------------------------------------------

class ControlIdentifier:
    """ A class to identify the control parameters of a zigzag experiment run using opty. 
    
    Encapsulates an Opty Direct Collocation optimization problem and provides the necessary
    setup of the equations of motion, constrains, bounds, initial guesses and stability checks.

    Untested Features:
    The class contains a few features that where trialed but whose functionality was never conclusively
    confirmed. These are
        EOM process noise : 
        Process noise may be added to the EOMs and the EOMs may be repeated multiple times with different
        noise. This follows [cite] and is intended to generate more stable results. However, for the 
        present problem, no benefit was observed and this feature was not used to generate the final 
        results.
        Stability penalty : 
        A stability penalty may be applied. To determine the penalty, the EOM's are optoimized 
        for a duration significantly longer then the available data. Diverging trajectories at the
        end of the optimization period are penalized. 
    """

    DEFAULT_GAIN_BOUNDS = [-1000,1000]

    def __init__(self,
               id,
               data_dict,
               ipopt_config={},
               bike_class=FixedSpeedBalancingRiderBicycle,
               gain_bounds=None,
               slack_state_bounds = None,
               integration_method = "midpoint",
               error_weight = 0.05,
               known_gain_map = {},
               build_problem_on_init=True,
               add_process_noise=False,
               noise_level = 0.0025,
               n_reps=1,
               add_stability_penalty = False,
               sim_duration = 15,
               pen_duration = 5,):
        """ Create a control identifier object

        Parameters
        ----------

        id : str
            The id of the data sample. 
        data_dict : dict
            A dictionary containting the trajectories of the data sample.
        ipopt_config : dict, optional
            Keyword arguments passed to IPOPT.
        bike_class : cyclistsocialforce.vehicle, optional
            The class of the bicycle. Should be FixedSpeedBalancingRiderBicycle 
            or FixedSpeedPlanarPointBicycle.
        gain_bounds : dict, optional
            A dictionary of the gain bounds. I.e. dict(k_psi = [-100, 100], ...)
        slack_state_bounds : float, optional
            A factor by which the tracked states may exceed the measurements. If 
            None, no bounds on the states are used. Default is None.
        integration_method : str, optional  
            The integration method Opty uses to solve the dynamic constraints. 
            Must be 'backward euler' or 'midpoint'.
        error_weight : float, optional
            A weight factor for the angular arror terms of the optimisation criteria.
        known_gains_map : dict, optional
            A dictionary of known / fixed gain values. These are kept fixed and 
            not optimized.
        build_problem_on_init : bool, optional
            If False, building the problem upon initialization is omitted. Use for 
            debugging. Default is True.
        add_process_noise : bool, 
            If true, use the process noise feature (see above!). THIS FEATURE
            IS UNTESTED. RESULTS MAY BE WRONG. USE WITH CAUTION.
        noise_level : float, optional
            Process noise level. 
        n_reps : int, optional
            Number of EOM repetitions
        add_stability_penalty : bool, optional
            If true, use the stability penalty feature. THIS FEATURE IS UNTESTED.
            RESULATS MAY BE WRONG. USE WITH CAUTION.
        sim_duration : float
            Duration in s to simulate the system.
        pen_duration : float
            Duration in s at the end of the total duration for which the stability 
            penalty is applied. 
        """

        # ------ Setup --------------------------------------------------------        
        self.integration_method = integration_method
        self.error_weight = error_weight
        self.ipopt_config = ipopt_config
        
        self.data_dict = dict(data_dict)
        self.N_data = len(data_dict["p_x_c"])-1

        self.solution = None
            
        self.id = id
        self.known_gain_map = known_gain_map
        self.n_known_params = len(known_gain_map)
        
        # ------ EOM Repetitions ------------------------------------------
        if not (0 < n_reps <= 10):
            msg = (f"Must be at least one repetition of eoms and less then 10. Instead, n_reps was "
                   f" {n_reps}.")
            raise ValueError(msg)
        self.n_reps = n_reps
        self.add_process_noise = add_process_noise
        self.noise_level = noise_level
        
        # ------ Stability Penalty ------------------------------------------
        if bike_class is not FixedSpeedBalancingRiderBicycle:
            if add_stability_penalty:
                logger.warning(("Ignoring 'add_stability_penalty' for models "
                                "other then 'FixedSpeedBalancingRiderBicycle'"))
            add_stability_penalty = False
            
        self.add_stability_penalty = add_stability_penalty
        if self.add_stability_penalty:
            self.N = int(sim_duration/T_S)
            self.N_reg = int(pen_duration/T_S)
            if add_stability_penalty and (self.N_data + self.N_reg) > self.N:
                msg = (f"Data segement and stablity segement overlap! Choose the "
                       f"simulation duration and the penalty duration so that "
                       f"sim_duration - pen_duration >= data_duration. Instead "
                       f"it was {sim_duration:.2} s - {pen_duration:.2} s < "
                       f"{self.N_data * T_S:.2} s.")
                raise ValueError(msg)
            
        else:
            self.N = self.N_data
            self.N_reg = self.N_data
                
        # ------ Model definition ---------------------------------------------
        self._setup_bikemodel(bike_class)
        self._form_eoms()
        self._data_frame = 'E'
        self._transform_input_data('N')
    
        # ------ Constraints --------------------------------------------------
        self.constraints = (
            self.states[self.pos_state_ids[0]](0) - self.data_dict['p_x'][0],  # initial x location
            self.states[self.pos_state_ids[1]](0) - self.data_dict['p_y'][0],  # initial y location
            self.states[self.psi_state_id[0]](0) - self.data_dict['psi'][0])   # initial orientation
    
        # ------ Bounds -------------------------------------------------------
        self._make_bounds(gain_bounds, slack_state_bounds)
            
        # ------ Known Trajectories -------------------------------------------
        
        p_x_c_aug = self.data_dict['p_x'][-1] + 5000 * (self.data_dict['p_x'][-1] - self.data_dict['p_x'][-2])
        p_x_c = np.ones(max(self.N, self.N_data+1)) * p_x_c_aug
        p_x_c[:self.N_data+1] = self.data_dict['p_x_c']
        self.data_dict['p_x_c'] = p_x_c
        
        p_y_c_aug = self.data_dict['p_y'][-1] + 5000 * (self.data_dict['p_y'][-1] - self.data_dict['p_y'][-2])
        p_y_c = np.ones(max(self.N, self.N_data+1)) * p_y_c_aug
        p_y_c[:self.N_data+1] = self.data_dict['p_y_c']
        self.data_dict['p_y_c'] = p_y_c
        
        v = np.ones(max(self.N, self.N_data+1)) * self.data_dict['v'][-1]
        v[:self.N_data+1] = self.data_dict['v']
        self.data_dict['v'] = v
        
        self.known_trajectory_map = {self.input[0](self.t): self.data_dict['p_x_c'][:self.N], 
                                     self.input[1](self.t): self.data_dict['p_y_c'][:self.N], 
                                     self.specified_params[0](self.t): self.data_dict['v'][:self.N]}
        
        if self.add_process_noise:
            self._sample_process_noise()
        
        # ------ Build Problem -------------------------------------------
        if build_problem_on_init:
            self._build_problem()
        
            for s in self.state_names:
                state_ids_opty = self.get_opty_state_ids(s)
                state_ids = self.state_ids[s] 
                    
                assert np.all(state_ids == state_ids_opty), "State id error!"
            
            
    def _transform_input_data(self, to_frame):
        """The input data comes in the E frame. The balancingrider model
        operates in the N frame. The planarpoint model in the E frame. 
        
        This function transforms the data if necessary.
        """
        
        if self._data_frame == to_frame:
            return
        elif to_frame not in ['N', 'E']:
            msg = f"Unknown data reference frame requested: {to_frame}"
            raise ValueError(msg)
            
        if self.bike_class is FixedSpeedBalancingRiderBicycle:
            
            mirrored_states = ['psi', 'dpsi', 'delta', 'ddelta',
                               'p_y', 'p_y_c', 'psi_c']
            
            for k in mirrored_states:  
                self.data_dict[k] *= -1
                
            self.data_dict["target_locations"][:,1] *= -1
            
            if self._data_frame == 'E':
                self._data_frame = 'N'
            elif self._data_frame == 'N':
                self._data_frame = 'E'
            else:
                msg = f"Unknown data reference frame: {self._data_frame}"
                raise ValueError(msg)
       
        
    def _transform_initial_guess(self, to_frame):
        """The initial guess comes in the E frame. The balancingrider 
        model operates in the N frame.The planarpoint model in the E frame. 
        
        Checks the current frame and only transforms the initial guess if necessary.

        Paramters
        ---------
        to_frame : str
            The frame to transform to. Must be 'E' or 'N'.
        """
        self.initial_guess, self._initial_guess_frame = \
            self._transform_free(self.initial_guess, self._initial_guess_frame,
                                 to_frame)
        
        
    def _transform_solution(self, to_frame):
        """The solution in the E frame. The balancingrider 
        model operates in the N frame.The planarpoint model in the E frame. 
        
        Checks the current frame and only transforms the solution if necessary.

        Paramters
        ---------
        to_frame : str
            The frame to transform to. Must be 'E' or 'N'.
        """
        self.solution, self._solution_frame = self._transform_free(
            self.solution, self._solution_frame, to_frame)
        
        
    def _transform_free(self, free, frame, to_frame):
        """The output data comes is expected in the E. The balancingrider 
        model operates in the N frame.The planarpoint model in the E frame. 
        
        Paramters
        ---------
        free : array
            The free parameters to be transformed.
        frame : str
            The current frame of free. Must be 'E' or 'N'.
        to_frame : str
            The frame to transform to. Must be 'E' or 'N'.
        """
        
        if frame == to_frame:
            return free, frame
        elif to_frame not in ['N', 'E']:
            msg = f"Unknown data reference frame requested: {to_frame}"
            raise ValueError(msg)
        
        if self.bike_class is FixedSpeedBalancingRiderBicycle:
            
            mirrored_states = ['psi', 'delta', 'deltadot', 'p_y']
            
            for k in mirrored_states:
                state_id = self.state_ids[k]
                free[state_id*self.N:(state_id+1)*self.N] *= -1
                
            if frame == 'E':
                frame = 'N'
            elif frame == 'N':
                frame = 'E'
            else:
                msg = f"Unknown data reference frame: {frame}"
                raise ValueError(msg)
                
        return free, frame
        
        
    def _build_problem(self):
        """ Build the problem. """
        
        # ------ Objective Function -------------------------------------------
        
        # SSE between simulated and real position and yaw angle
        def obj(free):
    
            states, specified_values, constant_values = parse_free(free, self.n_states*self.n_reps, 
                                                                   0, self.N)
            
            sse = 0
            
            for obj_state in self.names_objective_states:
                state_ids = self.state_ids[obj_state] + np.arange(self.n_reps) * self.n_states
                error_s = ((states[state_ids,:self.N_data] - self.data_dict[obj_state][np.newaxis,:-1]) 
                                / self.err_scales[obj_state])
                
                sse += np.nansum(error_s**2)
                 
            if self.bike_class == FixedSpeedBalancingRiderBicycle and self.add_stability_penalty:
                
                n_reg = self.N_reg
                id_phi = self.state_ids['phi']
                stability_reg = states[id_phi, -int(0.5*n_reg):] / self.err_scales['phi_reg']
                sse += np.nansum(stability_reg**2)
            
            return sse
    
    
        def obj_grad(free):
    
            states, specified_values, constant_values = parse_free(free, self.n_states*self.n_reps, 
                                                                   0, self.N)
    
            dsse = np.zeros_like(free)
    
            for obj_state in self.names_objective_states:
                state_ids = self.state_ids[obj_state] + np.arange(self.n_reps) * self.n_states
            
                grad_ids = (np.tile(np.arange(0, self.N)[:,np.newaxis], [1,self.n_reps]) + state_ids*self.N).T.flatten()
                grad_ids = grad_ids[:self.N_data]
                
                dsse[grad_ids] = (2*(1/self.err_scales[obj_state])**2 
                                      * (states[state_ids,:self.N_data] - self.data_dict[obj_state+""][np.newaxis,:-1])).flatten() 
    
            dsse[np.isnan(dsse)] = 0
    
            if self.bike_class == FixedSpeedBalancingRiderBicycle and self.add_stability_penalty:
                
                n_reg =self.N_reg
                id_phi = self.state_ids['phi']
                stability_reg = states[id_phi, -n_reg:] / self.err_scales['phi_reg']
                
                dsse[self.N*(id_phi+1)-n_reg:self.N*(id_phi+1)] = \
                    2 * (1/self.err_scales['phi'])**2 * stability_reg
    
            return dsse
        
        # ------ Opty Problem -------------------------------------------
        state_functions = [s(self.t) for s in self.states]
        
        self.prob = Problem(
                    obj,
                    obj_grad,
                    self.f,
                    state_functions,
                    self.N,
                    T_S,
                    known_trajectory_map=self.known_trajectory_map,
                    instance_constraints=self.constraints,
                    bounds=self.bounds,
                    integration_method=self.integration_method,
                )
        
        self._apply_ipopt_config()
        
    def _sample_process_noise(self):
        """ Sample noise trajectories for the noise parameters. 
        
        Noise is normally distributed with mean = 0. The standard deviation 
        is derived from the customizable noise level self.noise_level and the bounds
        of each noisy state: std = self.noise_level * (upper_bound - lower_bound).
        """
        
        rng = np.random.default_rng()
        noisy_states = np.tile(self.noisy_states, self.n_reps)
        
        for n, sid in zip(self.noise, noisy_states):
            
            #derive noise level from the state bounds
            try:
                bounds = self.bounds[self.states[sid](self.t)]
                sigma = self.noise_level * (bounds[1] - bounds[0])
            except KeyError: 
                if self.state_names[sid][-3:] == 'dot':
                    statedot_m = np.diff(self.data_dict[self.state_names[sid][:-3]])/T_S
                    bounds = np.array((min(statedot_m[1:-1]), max(statedot_m[1:-1])))
                    sigma = self.noise_level * 2 * max(abs(bounds))
                else:
                    sigma = self.noise_level
            
            #sample a noise trajectory
            self.known_trajectory_map[n(self.t)] = rng.normal(0,sigma, self.N)
            
        test=0
        

    def _setup_bikemodel(self, bike_class):
        """ Sets up the RunControlIdentifier for different bicycle models.       
        """
        
        self.bike_class = bike_class
        
        # bikemodel-dependent setup
        if bike_class is FixedSpeedBalancingRiderBicycle:
            self._setup_balancingrider()
        elif bike_class is FixedSpeedPlanarPointBicycle:
            self._setup_planarpoint()
        else:
            msg = f"The bicycle model class '{bike_class}' is not supported!"
            raise UnknownBicycleModelException(msg)
        
    
    def _setup_balancingrider(self):
        """ Sets up the RunControlIdentifier for the Balancing Rider model. 
        """
        
        # create a cyclistsocialforce bike object for retrieving the dynamics. 
        s0 = (0,0,0,np.mean(self.data_dict["v"]),0,0,0,0)
        self.bike = self.bike_class(s0)
        
        # state and parameter names
        self.state_names = ["phi", "delta", "phidot", "deltadot", "psi", "p_x", "p_y"]
        self.param_names = ["k_phi", "k_delta", "k_phidot", "k_deltadot", "k_psi"]
        
        # ids of position and orientation states
        self.noisy_states = np.array([2,3,4,5,6])
        self.pos_state_ids = np.array([5,6])
        self.psi_state_id = np.array([4])
        self.bikerider_state_ids = np.array([0,1,2,3,4])
        self.Kx_param_ids = np.array([0,1,2,3,4])
        self.Ku_param_ids = np.array([4])
        
        # ------ Error scaling and objective features -------------------------
        self.names_objective_states = ['delta', 'phi', 'psi']
        
        self.err_scales = {}
        self.err_scales['delta'] = self.N_data * np.pi/(self.error_weight*180)
        self.err_scales['phi'] = self.N_data * np.pi/(self.error_weight*180)
        self.err_scales['psi'] = self.N_data * np.pi/(self.error_weight*180)
        self.err_scales['phi_reg'] = 0.5 * self.err_scales['phi'] / self.N_data * (self.N_reg)
        
    def _setup_planarpoint(self):
        """ Sets up the RunControlIdentifier for the Planar Point model. 
        """
        
        # create a cyclistsocialforce bike object for retrieving the dynamics. 
        s0 = (0,0,0,np.mean(self.data_dict["v"]))
        self.bike = self.bike_class(s0)
        
        # state and parameter names
        self.state_names = ["psi", "p_x", "p_y"]
        self.param_names = ["k_psi"]
        
        # ids of position and orientation states
        self.noisy_states = np.array([0,1,2])
        self.pos_state_ids = np.array([1,2])
        self.psi_state_id = np.array([0])
        
        # ------ Error scaling and objective features -------------------------
        self.names_objective_states = ['psi'] 
        
        self.err_scales = {}
        self.err_scales['psi'] = self.N * np.pi/(self.error_weight*180)
        #self.err_scales['p_x'] = self.N_data * 0.5
        #self.err_scales['p_y'] = self.N_data * 0.5
        
            
    def _apply_ipopt_config(self):
        """ Add the options of ipopt_config to the problem. """
        
        for key in self.ipopt_config:
            try:
                self.prob.add_option(key, self.ipopt_config[key])
            except Exception as e:
                print((f"Failed to assign '{key}' as an option to IPOPT with "
                       f"exeption {type(e).__name__}:  {str(e)}"))
                
    def _make_bounds(self, gain_bounds, slack_bounds):
        """ Make the bounds dictionary. Uses the given gain bounds and derives
        state bounds from  extrema of the data plus the specified slack. 
        States without measurement data are unbounded. Supply slack=None to 
        disable state bounds.

        Parameters
        ----------
        gain_bounds : dict
            Dictionary of bounds for all gains.
        slack_bounds : float
            Slack factor for state bounds. Supply None to deactivate state bounds. 
        """
             
        self.slack_bounds = slack_bounds
        self.bounds = {}
        
        # derive state bounds from data: slack * [min, max]
        if slack_bounds is not None:
            
            if not (0 <= slack_bounds <= 1):
                msg = (f"Slack of bounds has to be in [0,1], instead it was "
                       f"{slack_bounds}.")       
                raise ValueError(msg)
            
            for s in self.states:
                key = s.name[:-1]
                if key in self.data_dict:
                    
                   minimum = (1 - np.sign(min(self.data_dict[key])) * \
                              slack_bounds) * min(self.data_dict[key])
                   maximum = (1 + np.sign(max(self.data_dict[key])) * \
                              slack_bounds) * max(self.data_dict[key])
                   self.bounds[s(self.t)] = (minimum, maximum)
            
        # set gain bound or use defaults
        if gain_bounds is None:
            for g in self.params:
                if g not in self.known_gain_map:
                    self.bounds[g] = self.DEFAULT_GAIN_BOUNDS
        else:
            for g in self.params:
                if g not in self.known_gain_map:
                    self.bounds[g] = gain_bounds[g.name[:-1]] 
            
                
    def _form_eoms(self):
        """ Form the equations of motion. """
        
        # time
        self.t = sm.symbols("t")
    
        # specified inputs
        p_x_c, p_y_c, v = sm.symbols("p_x_c p_y_c v", cls=sm.Function)
        self.input = [p_x_c, p_y_c]
        self.specified_params = [v]
    
        # dynamically create symbols and functions
        def make_symbols(symbol_names, n_reps = 1, is_function=False):
            symbol_str = ""
            for i in range(n_reps):
                for s in symbol_names:
                    symbol_str += f" {s}{i}"
                    
            if is_function:
                symbols = sm.symbols(symbol_str, cls=sm.Function)
            else:
                symbols = sm.symbols(symbol_str)
                
            if not isinstance(symbols, tuple):
                symbols = (symbols,)
                
            symbol_ids = {symbol_names[j]: j for j in range(len(symbol_names))}
                
            return symbols, len(symbol_names), symbol_ids
        
        # make state functions
        self.states, self.n_states, self.state_ids = make_symbols(self.state_names, n_reps = self.n_reps, is_function=True)
   
        # make unknown parameter symbols
        self.params, self.n_params, self.param_ids = make_symbols(self.param_names)
        
        if self.add_process_noise:
            state_names = np.array(self.state_names)
            noise_names = [f'n_{s}' for s in state_names[self.noisy_states]]
            self.noise, self.n_noise, self.noise_ids = make_symbols(noise_names, n_reps = self.n_reps, is_function=True)
    
        # make n_reps repetitions of the equations of motions
        eoms = [None] * self.n_reps
        
        for i in range(self.n_reps):
            
            if self.bike_class is FixedSpeedBalancingRiderBicycle:
                eoms[i] = self._make_eoms_set_balancingrider(p_x_c, p_y_c, v, i)
            elif self.bike_class is FixedSpeedPlanarPointBicycle:
                eoms[i] = self._make_eom_set_planarpoint(p_x_c, p_y_c, v, i)
            else:
                raise UnknownBicycleModelException(f'Bike class {self.bike_class} not supported')
            
            # add process noise         
            if self.add_process_noise:
                f_n = [[0]] * self.n_states
                for n, n_id in zip(self.noise[i*self.n_noise:(i+1)*self.n_noise], self.noisy_states):
                    f_n[n_id] = [n(self.t)]
                f_n = sm.Matrix(f_n)
                
                eoms[i] += f_n
            
        # combine repitions
        if self.n_reps > 1:
            self.f = eoms[0].col_join(eoms[1])
            for i in range(2, self.n_reps):
                self.f = self.f.col_join(eoms[i])
        else:
            self.f = eoms[0]
            
        # substitute known gains
        known_gain_map = {}
        for k in self.known_gain_map:
            known_gain_map[self.params[self.param_names.index(k)]] = self.known_gain_map[k]
        self.known_gain_map = known_gain_map
        self.f = self.f.subs(known_gain_map)
        
        
    def _make_eoms_set_balancingrider(self, p_x_c, p_y_c, v, i):
        """ Make a set of EOMs for the balancing rider model.
        
        Parameters
        ----------
        
        p_x_c : sympy.Function
            Function symbol representing the commanded x location.
        p_x_y : sympy.Function
            Function symbol representing the commanded y location.
        v : sympy.Function
            Function symbol representing the speed.
        i : int
            The subscript of this EOM set.
        
        Returns
        -------
        
        eoms : sm.Matrix
            A vector of EOMs. 
        """
        
        # bike-rider states
        x_br = sm.Matrix([[self.states[j](self.t)] for j in self.bikerider_state_ids[:self.n_states]+i*self.n_states])
    
        # bike-rider state-space matrices
        A_br, B_br, Cbr, Dbr = self.bike.dynamics.get_symbolic_statespace_matrices(self.t)
        A_br = sm.Matrix(A_br)
        B_br = sm.Matrix(B_br[:, 1])  # only use steer torque input
    
        # gain matrices
        K_x = sm.Matrix([[self.params[j]] for j in self.Kx_param_ids]).T
        K_u = sm.Matrix([[self.params[j]] for j in self.Ku_param_ids])
        
        # bike-rider eoms
        pos_state_ids = np.array(self.pos_state_ids) + i * self.n_states
        f_br = x_br.diff() - ((A_br - B_br * K_x) * x_br + B_br * K_u * sm.atan((p_y_c(self.t) - self.states[pos_state_ids[1]](self.t)) / 
                                                                                (p_x_c(self.t) - self.states[pos_state_ids[0]](self.t))))
        
        # forward motion eoms 
        psi_state_id = self.psi_state_id[0] + i * self.n_states
        f_fw = sm.Matrix(
            [[self.states[pos_state_ids[0]](self.t).diff() - v(self.t) * sm.cos(self.states[psi_state_id](self.t))],   #xdot - v * cos(psi)
             [self.states[pos_state_ids[1]](self.t).diff() - v(self.t) * sm.sin(self.states[psi_state_id](self.t))]])  #ydot - v * sin(psi)
    
        # combine bike-rider eoms and forward eoms
        eoms = f_br.col_join(f_fw)
        
        return eoms
    
    
    def _make_eom_set_planarpoint(self, p_x_c, p_y_c, v, i):
        """ Make a set of EOMs for the balancing rider model.
        
        Parameters
        ----------
        
        p_x_c : sympy.Function
            Function symbol representing the commanded x location.
        p_x_y : sympy.Function
            Function symbol representing the commanded y location.
        v : sympy.Function
            Function symbol representing the speed.
        i : int
            The subscript of this EOM set.
        
        Returns
        -------
        
        eoms : sm.Matrix
            A vector of EOMs. 
        """

        # yaw tracking
        pos_state_ids = np.array(self.pos_state_ids) + i * self.n_states
        psi_state_id = self.psi_state_id[0] + i * self.n_states
    
        f_psi = sm.Matrix(
            [(self.states[psi_state_id](self.t).diff() + 
              self.params[0] * (self.states[psi_state_id](self.t) - 
                                sm.atan((p_y_c(self.t) - self.states[pos_state_ids[1]](self.t)) / 
                                        (p_x_c(self.t) - self.states[pos_state_ids[0]](self.t)))))])


        
        # forward motion eoms 
        f_fw = sm.Matrix(
            [[self.states[pos_state_ids[0]](self.t).diff() - v(self.t) * sm.cos(self.states[psi_state_id](self.t))],   #xdot - v * cos(psi)
             [self.states[pos_state_ids[1]](self.t).diff() - v(self.t) * sm.sin(self.states[psi_state_id](self.t))]])  #ydot - v * sin(psi)
    
        # combine yaw_tracking and forward eoms
        eoms = f_psi.col_join(f_fw)
        
        return eoms
    
                
    def _dict_to_opty(self, gain_dict):
        """Convert a gain dict to a gain array in 'opty' format
        
        Parameters
        ----------
        gain_dict : dict
            Dictionary containing gain key, value pairs. 

        Returns
        -------
        opty_gain_array : array
            Array of gains in the order of symbols defined in the Opty problem.  
        """
        opty_gain_array = np.array(
            [gain_dict[symbol.name[:-1]] for symbol in \
             self.prob.collocator.unknown_parameters]
        )
        return opty_gain_array
    

    def _opty_to_dict(self, opty_gain_array, add_known = False):
        """Convert a gain array in 'opty' format to a gain dictionary
        
        Parameters
        ----------
        opty_gain_array : array
            Array of gains in the order of symbols defined in the Opty problem.  

        Returns
        -------
        gain_dict : dict
            Dictionary containing gain key, value pairs. 
        """
        gain_dict = {}
        i = 0
        for symbol in self.prob.collocator.unknown_parameters:
            gain_dict[symbol.name[:-1]] = opty_gain_array[i]
            i+=1
            
        if add_known:
            for symbol, value in self.known_gain_map.items():
                gain_dict[symbol.name[:-1]] = value
        return gain_dict
    
    def _get_state_from_csf_traj(self, state_name, traj):
        """ Pick the trajectory of a given feature from a traj array in 
        cyclistsocialforce format. 

        Parameters
        ----------
        state_name : str
            The name of the feature/state to retrive.
        traj : array
            The array of state trajectories.
        """
        csf_traj_names = np.array(('p_x', 'p_y', 'psi', 'v', 'delta', 'phi'))
        
        if 'dot' in state_name:
            s = state_name[0:-3]
            derivative = True
        else:
            s = state_name
            derivative = False
            
        state_id = np.argwhere(csf_traj_names == s)[0,0]
        
        if derivative:
            traj_s = np.gradient(traj[state_id,:self.N])/T_S
            
        else:
            traj_s = traj[state_id,:self.N]
            
        return traj_s
        
        
    
    def get_opty_state_ids(self, states):
        """
        Return the index of a state in the opty output by name. 
        
        If the states are repeated (ie. delta0 ... deltaN), this only returns the first index. 
        
        Parameters
        ----------
        
        states : list(str)
            Opty state names without running number. E.g. deltadot (not deltadot0)

        """
        if type(states) is str:
            states = [states]
        opty_states = np.array([s.name for s in self.prob.collocator.state_symbols])
        
        indices = np.zeros(len(states), dtype=int)
        
        for s, i in zip(states, range(len(states))):
            try:
                indices[i] = np.argwhere(s+'0' == opty_states)[0][0]
            except IndexError:
                msg = f'{s} is not a state known to RunControlIdentifier!'
                raise KeyError(msg)
                
        if len(states) == 1:
            indices = indices[0]
                
        return indices
        
        
    
    def make_initial_guess(self, poles=None, gains=None, feasible=False):
        """ Make an initial guess for the solution from the data, based on a 
        given set of poles, or based on a given set of gains.

        Parameters
        ----------
        feasible : bool, optional
            If True, a feasible guess based on the given set of poles / gains is 
            created. If False, the data is used as initial guess. 
        poles : array
            Array of poles. If feasible is True, this is used to form an initial 
            guess.
        gains : dict
            Dictionary of gains. If feasible is True, this is used to form
            an initial guess. 

        Returns
        -------
        initial_guess_stable : array
            A stable initial guess for free.

        """

        assert ((poles is None) or (gains is None)), (f"Only one of poles or "
                                                       "gains can be not None")

        if feasible:
        # Simulate a feasible solution from the guessed gains
        
            #transform data into E frame if necessary
            self._transform_input_data('E')
            
            s0 = [self.data_dict["p_x"][0],
                  self.data_dict["p_y"][0],
                  self.data_dict["psi"][0],
                  self.data_dict["v"][0],
                  self.data_dict["delta"][0],
                  self.data_dict["phi"][0],
                  self.data_dict["ddelta"][0],
                  self.data_dict["dphi"][0],
                  ]
            
            if gains is None and poles is None:
            # if no explicit initial guess, simulate with default gains        
                test_sim = FixedInputZigZagTest(
                    s0, 
                    self.data_dict["p_x_c"], 
                    self.data_dict["p_y_c"], 
                    self.data_dict["v"], 
                    self.data_dict["target_locations"], 
                    cyclist_name="initialguess", 
                    bike_class=self.bike_class,
                    animate=False,
                )
                gain_array = self._dict_to_opty(
                    self.bike_class.DYNAMICS_TYPE.csf_to_dict(test_sim.bike.dynamics.gains[0]))
            else:
            # if explicit initial guess, use this to simulate
                test_sim = FixedInputZigZagTest(
                    s0, 
                    self.data_dict["p_x_c"], 
                    self.data_dict["p_y_c"], 
                    self.data_dict["v"], 
                    self.data_dict["target_locations"], 
                    cyclist_name="initialguess", 
                    bike_class=self.bike_class,
                    gains = self.bike_class.DYNAMICS_TYPE.dict_to_csf(gains),
                    poles = poles,
                    animate=False,
                )
                if gains is None:
                    gain_array = self._dict_to_opty(
                        self.bike_class.DYNAMICS_TYPE.csf_to_dict(test_sim.bike.dynamics.gains[0])) 
                else:
                    gain_array = self._dict_to_opty(gains)
            
            test_sim.animate = False
            test_sim.verbose = False
            try:
                test_sim.run()
            except Exception:
                raise NoForwardSimulationException()
            
            # build initial guess array
            self.initial_guess = np.zeros(self.N * self.n_states * self.n_reps + (self.n_params - self.n_known_params))
            self.initial_guess[-(self.n_params-self.n_known_params):] = gain_array
            for s in self.state_names:
                iopt = self.get_opty_state_ids(s)
                self.initial_guess[iopt*self.N:(iopt+1)*self.N] = \
                    self._get_state_from_csf_traj(s, test_sim.bike.traj)
                    
            # transform the input data back if necessary and transform the 
            # initial guess as well.
            self._initial_guess_frame = 'E'
            self._transform_input_data('N')
            self._transform_initial_guess('N')

        else:
            
            self._transform_input_data('N')
            
            if gains is None:
                gains = self._dict_to_opty(
                    self.bike_class.DYNAMICS_TYPE.csf_to_dict(self.bike.dynamics.gains[0])) 
                
            # Use the data as initial guess
            t_guess = np.arange(self.N)
            t_finite = np.where(np.isfinite(self.data_dict['p_x']))[0]
            finite = np.isfinite(self.data_dict['p_x'])
            
            gain_array = self._dict_to_opty(gains)
            
            # build initial guess array
            self.initial_guess = np.zeros(self.N * self.n_states * self.n_reps + (self.n_params - self.n_known_params))
            self.initial_guess[-(self.n_params-self.n_known_params):] = gain_array
            
            measurement_states = ('p_x', 'p_y', 'psi', 'phi', 'delta')
            
            for s in self.state_names:
                if s in measurement_states:
                    iopt = self.get_opty_state_ids(s)
                    data = np.interp(t_guess, t_finite, self.data_dict[s][finite])
                    self.initial_guess[iopt*self.N:(iopt+1)*self.N] = data
    
            self.initial_guess[np.isnan(self.initial_guess)] = 0
            
            self._initial_guess_frame = 'N'
            
        #repeat trajectories    
        if self.n_reps > 1:
            n_traj = self.n_states*self.N
            self.initial_guess = np.r_[np.tile(self.initial_guess[:n_traj],self.n_reps), gain_array]
            
        initial_guess_stable, poles = self._test_stability(gain_array)
        
        return initial_guess_stable            

        
    def fit(self, initial_guess = None, feasible = None):
        """ Fit the problem to the data. 

        Parameters
        ----------
        initial_guess : array, optional
            An initial guess of the gains in cyclistsocialforce format. 
        feasible : bool, optional
            If true, a feasible initial guess for the full solution based on the given gain guess is generated. 

        Returns
        -------
        solution : array
            The opty solution array
        info : dict
            The opty info dict
        gains : list
            A list of gains in the order defined by the opty problem. 
        poles : list 
            A list of poles corresponding to the gains.
        eval : dict
            The evaluation of the optimization result.
        """
        
        # ------ Initial guess ------------------------------------------------
        if initial_guess is None:
            igstable = self.make_initial_guess(feasible = feasible)
        else:
            if np.all(np.isreal(initial_guess)):
                igstable = self.make_initial_guess(
                                        gains=self.bike_class.DYNAMICS_TYPE.csf_to_dict(initial_guess), 
                                        feasible = feasible)
            else:
                igstable = self.make_initial_guess(poles = initial_guess, 
                                        feasible = feasible)
        
        # ------ Solution -----------------------------------------------------
        self._transform_input_data('N')
        self._transform_initial_guess('N')
        
        solution, info = self.prob.solve(self.initial_guess)
        self.solution = solution
        self._solution_frame = 'N'
        self.info = info
        
        self._transform_input_data('E')
        self._transform_solution('E')
        self._transform_initial_guess('E')
        
        # ------ Evaluate -----------------------------------------------------
        self._evaluate_fit(igstable)

        
        return self.solution, self.info, self.gains, self.poles, self.eval
    
    def get_solution_dict(self):
        """
        Returns a dictionary with the following content:
            solution
                states
                gains
            initial guess
                states
                gains
            n_states
            n_data
            n_pen
            t_s

        Returns
        -------
        solution_dict : dict
            The solution dictionary
        """
        
        if self.solution is None:
            raise RuntimeError(("Call the fit() function before requresting ",
                                "a solution."))
        
        solution_dict = {'n_states': self.n_states,
                         'n_sim': self.N,
                         'n_data': self.N_data,
                         'n_pen': self.N_reg,
                         't_s': T_S}
        
        for label, free in zip(('initial_guess', 'solution'), 
                               (self.initial_guess, self.solution)):
            
            states_dict = {}
            
            states, temp1, gains = parse_free(free, self.n_states*self.n_reps, 
                                              0, self.N)

            for i in range(self.n_states):
                state_name = self.state_names[i]
                state_index = self.get_opty_state_ids(state_name)
                
                states_dict[state_name] = list(states[state_index,:])
                
            label_dict = {'states': states_dict,
                          'gains': self._opty_to_dict(gains, add_known=True)}
            
            solution_dict[label] = label_dict
            
            
        return solution_dict
    
    
    def write_solution(self, filepath):
        """ Write the solution to a .pkl file.
        
        filenpath : str
            Path to the file (directory + filename). 
        """
        
        if filepath[-4:] != '.pkl':
            filepath += '.pkl'
        
        solution_dict = self.get_solution_dict()
        
        with open(filepath, 'wb') as file:
            pickle.dump(solution_dict, file)
        
    
    def _evaluate_fit(self, igstable):
        """ Evaluate the solution.

        Parameters
        ----------
        igstable : bool
            Indicate if the initial guess was stable. 
        """
        
        # ------ Evaluate -----------------------------------------------------
        states, temp1, gains = parse_free(self.solution, self.n_states*self.n_reps, 0, self.N)
        
        self.gains = list(gains)
        
        self.eval = {}
        self.eval['state_names'] = [self.state_names[i] for i in range(self.n_states)]
        self.eval['param_names'] = [p.name[:-1] for p in self.prob.collocator.unknown_parameters]
        self.eval['VAF'] = np.zeros(self.n_states)
        self.eval['MAE'] = np.zeros(self.n_states)
        
        #add known gains and values
        for k, v in self.known_gain_map.items():
            self.eval['param_names'].append(k.name[:-1])
            self.gains.append(v)
        
        self.eval['stability-initial-guess'] = igstable
        
        # VAF and MAE
        for i in range(self.n_states):
            state_name = self.state_names[i]
            state_index = self.get_opty_state_ids(state_name)
            if state_name in self.data_dict:
                finite = np.isfinite(self.data_dict[state_name][:-1])
                data_m = self.data_dict[state_name][:-1][finite]
                data_c = states[state_index,:self.N_data][finite]
                evaluator = pm.RegressionMetric(data_m, data_c)
                
                results = evaluator.get_metrics_by_list_names(["VAF", "MAE"])
                
                self.eval['VAF'][i] = results['VAF']
                self.eval['MAE'][i] = results['MAE']
            else:
                self.eval['VAF'][i] = None
                self.eval['MAE'][i] = None
            
        # euclidian error of the position
        pos_calib = states[self.get_opty_state_ids(['p_x','p_y']),:self.N_data].T
        pos_meas = np.c_[self.data_dict['p_x'][:-1], self.data_dict['p_y'][:-1]]
        
        finite = np.isfinite(self.data_dict['p_x'][:-1])
        pos_calib = pos_calib[finite,:]
        pos_meas = pos_meas[finite,:]
        
        self.eval['position_error'] = np.mean(np.sqrt(np.nansum((pos_calib - pos_meas)**2, axis=1)))
        
        # euclidian error of the pose
        r2d = 180 / np.pi
        angle_factor = 0.1  #10 deg error equals 1m error
        pose_calib = states[self.get_opty_state_ids(['p_x','p_y', 'psi']),:self.N_data].T
        pose_meas = np.c_[self.data_dict['p_x'][:-1], self.data_dict['p_y'][:-1], self.data_dict['psi'][:-1]]
        
        pose_calib = pose_calib[finite,:]
        pose_meas = pose_meas[finite,:]
        
        pose_calib *= np.array([1,1,angle_factor*r2d])
        pose_meas *= np.array([1,1,angle_factor*r2d])         
                              
        self.eval['pose_error'] = np.mean(np.sqrt(np.nansum((pose_calib - pose_meas)**2, axis=1)))
        
        # euclidian error of the lateral dynamics
        if 'phi' in self.state_names and 'psi' in self.state_names:
            r2d = 180 / np.pi
            lat_calib = states[self.get_opty_state_ids(['phi', 'delta']),:self.N_data].T
            lat_meas = np.c_[self.data_dict['phi'][:-1], self.data_dict['delta'][:-1]]
            
            lat_calib *= np.array([r2d, r2d])
            lat_meas *= np.array([r2d, r2d])         
            
            self.eval['lat_error'] = np.mean(np.sqrt(np.nansum((lat_calib - lat_meas)**2, axis=1)))
            
        else:
            self.eval['lat_error'] = None
       
        # stability and poles
        stable, poles = self._test_stability(gains)
        self.poles = poles
        self.eval['stability'] = stable
        
        # objective
        if self.add_stability_penalty:
            obj_full = self.prob.obj(self.solution)
            
            self.add_stability_penalty = False
            obj_data = self.prob.obj(self.solution)
            obj_stab = obj_full - obj_data
            self.add_stability_penalty = True
            
            self.eval['objective'] = obj_data
            self.eval['objective_stab'] = obj_stab
            self.eval['objective_sum'] = obj_full
            
        else:
            self.eval['objective'] = self.prob.obj(self.solution)
        
    
    def _test_stability(self, gains):
        """
        Test if the bicycle of this split is stable under the gains gains.

        Parameters
        ----------
        gains : array
            Gains to test stability for.

        Returns
        -------
        stable : bool
            True if stable.
        poles : array
            List of poles.

        """

        gains = self.bike_class.DYNAMICS_TYPE.dict_to_csf(self._opty_to_dict(gains, add_known=True))
        A, B, C, D = self.bike.dynamics.get_statespace_matrices(np.mean(self.data_dict["v"]))
        if isinstance(self.bike, FixedSpeedBalancingRiderBicycle):
            B = np.reshape(B[:,1], (5,1))   #drop roll torque input
            D = np.reshape(D[:,1], (1,1))   #drop roll torque input
        sys, gains = from_gains(A, B, C, D, gains)
        stable, poles = test_stability(sys, stability_type='marginal')
        
        return stable, poles
    
    
    def _make_csf_traj(self, state_array):
        """ Make a state array in cyclistsocialforce state order given an 
        array of states in opty order. 
        """
        
        csf_traj_names = ('p_x', 'p_y', 'psi', 'v', 'delta', 'phi')
        traj = np.ones((len(csf_traj_names), state_array.shape[1])) * np.nan
        
        for i in range(len(csf_traj_names)):
            name = csf_traj_names[i]
            if name in self.state_names:
                traj[i,:] = state_array[self.get_opty_state_ids(name)]
        
        return traj
        
        
    def plot(self, output_filepath=None):
        """ Plot the soluton.

        Parameters
        ----------
        output_filepath : str
            If supplied, the plot is saved using the supplied file path. 
        """
        
        self._transform_input_data('E')
        self._transform_solution('E')
        self._transform_initial_guess('E')
        
        states, temp1, temp2 = parse_free(self.solution, self.n_states, 0, self.N)
        states_guess, temp1, temp2 = parse_free(self.initial_guess, self.n_states, 0, self.N)
        
        fig, ax = plot_results(exp_data_dict=self.data_dict, 
                     calib_states=self._make_csf_traj(states),
                     guess_states=self._make_csf_traj(states_guess),
                     known_trajectory_map=self.known_trajectory_map,
                     name=self.id)
        
        if not output_filepath is None:
            fig.savefig(output_filepath, dpi=300)
        

    def print(self):
        """ Print the solution to terminal.
        """
        
        print(f"################## Solution ##################")
        print(self.id)
        
        print(self.info['status_msg'])

        print("")
        print("Identified gains from data")
        print("--------------------------")
        for symbol, val in zip(
            self.prob.collocator.unknown_parameters,
            self.solution[-len(self.prob.collocator.unknown_parameters):]):  
            print(f"{str(symbol).ljust(11)}: {val:.2f}")

        print("")
        print("Evaluation")
        print("----------")
        print(f"Objective:   {self.info['obj_val']:.8f}")
 
        
class IDSessionManager:
    """ Manage an ID session. Create gain guesses, run ID for multiple guesses and collect and
    store results.
    """
    
    MODEL_CLASSES = {'balancingrider': FixedSpeedBalancingRiderBicycle,
                     'planarpoint': FixedSpeedPlanarPointBicycle}
    
    def __init__(self, 
                 dir_data, 
                 sample_ids,
                 session_id='session', 
                 n_samples=np.inf, 
                 opty_config={}, 
                 ipopt_config={},
                 dir_results='results',
                 command_time_shift = None,
                 note=None, 
                 gain_guess_mode = None,
                 feasible_initial_guesses=True,
                 gain_search_limits = None,
                 model='balancingrider',
                 close_plots=False,
                 plot_mode= 'all',
                 tag=None):
        """
        Create an ID session manager. 

        Parameters
        ----------
        dir_data : str
            The root directory of the control ID data. 
        dir_results : str
            The desired root directory for the control ID results
        sample_ids : list
            A list of the id's of the data samples to be identified,
        n_samples : int, optional
            Number of samples from sample_indexs to identify. Use for debugging.
        session_id : str, optional
            An id for this session. The ID will appear in the output folder name.
            Should describe the samples in sample_index, for example the participant_id
            if all samples in sample_index are from the same participant.  
        opty_config : dict, optional
            Kwargs passed to the control identifier.
        ipopt_config : dict, optional
            Kwargs passed to IPOPT.
        command_time_shift : float, dict or None, opttional
            Apply a time shift to the command. Use this to compensate response times. 
            If float, the same command time shift is applied to all samples. If dict, 
            a separate time shift can be applied to each sample.
        note : str, optional
            A note to add to the logfile. 
        gain_guess_mode : int, str or None, optional
            If int, generate the given number of random initial gain guesses.
            If str, the parameter is interpreted as filepath for loading existing initial guesses from
            a .guess file. If None, use the default gains as initial guess. 
        feasible_initial_guesses : bool, optional
            Genearate feasible initial guesses for all states besed on the gain guess. 
            Default is true. 
        gain_search_limits : array, optional
            Gain search limits in cyclistsocialforce format. 
        model : str, optional
            The cyclist model to use for identification. Must be balancingrider or planarpoint
        close_plots : bool, optional
            Close plots after the identification of one sample is finished.
        plot_mode : str, optional
            Must be 'best', 'all', 'none'. If 'best', only the best results for each sample are plotted.
            If all, the results of all initial guesses per sample are plotted. If None, no 
            results are plotted.
        """
        self.paths = PathManager(data_dir=dir_data, id_result_dir=dir_results) 
        
        #set up model type
        self._parse_model(model)
        
        #give this session a tag
        self.session_id = session_id
        if 'add_process_noise' in opty_config:
            noisetag = '_n' 
        else:
            noisetag = ""
        self.tag = (f"{strftime('%y%m%d%H%M%S')}_rcid_{session_id}{noisetag}")
            
        if tag is not None:
            self.tag += f"_{tag}"
        
        #make files and directories
        self._make_directories(dir_results)
        self._init_output_files()
        
        #init logging
        self._init_logging()
        
        #set parameters
        self.dir_data = dir_data       
        self.sample_ids = sample_ids
        
        self.opty_config = opty_config
        self.ipopt_config = ipopt_config
        self.n_samples = n_samples
        self.n_samples = min(n_samples, len(sample_ids['splits']))
        self.note = note
        self.feasible_initial_guesses = feasible_initial_guesses
        
        self.gain_search_limits = gain_search_limits
        
        self.close_plots = close_plots
        self.plot_mode = plot_mode
        if self.plot_mode not in ('best', 'all', 'none'):
            msg = (f"plot_mode must be one of ('best', 'all', 'none')! "
                   f"Instead it was '{plot_mode}'.")
            raise ValueError(msg)
        
        #set up target location
        self.target_locations = self._get_target_locations()
        
        #command time shift
        self.command_time_shift = command_time_shift
        if command_time_shift == 0:
            command_time_shift = None
        self.max_offset = None

        self.gain_guess_mode = gain_guess_mode
        self.gain_limits = None
        
        # Add arguments to opty_config
        self.opty_config['bike_class'] = self.bike_class
        self.opty_config['ipopt_config'] = self.ipopt_config
 
        
    def _parse_model(self, model):
        """ Parse the cyclist model string.

        model : str
            Cyclist model string. Must be any of IDSessionManager.MODEL_CLASSES
        """
        
        assert model in IDSessionManager.MODEL_CLASSES, (f'Model type {model} not supported! Choose'
                                                         f'any of '
                                                         f'{IDSessionManager.MODEL_CLASSES.keys()}')
        
        self.bike_class = IDSessionManager.MODEL_CLASSES[model]


    def _get_target_locations(self):
        """ Get the target locations corresponding to the participant.
        """
        
        target_locations = read_yaml(self.paths.getfilepath_targetcalibration())
        part = self.sample_ids['participant_id']

        for testday in target_locations:
            if part in target_locations[testday]['participants']:
                return np.array(target_locations[testday]['target_locations'])
        
        raise ValueError(f"Couldn't find participant {part} in 'calibration.yaml'!")


    def _make_gain_guesses(self, v_minmax, sample_index):
        """ Create gain guesses.

        Samples the requrested number of random stable gains if 
        gain_guess_mode is an integer. If str, reuses old guesses, either from the
        previous run or load from file. 

        Parameters
        ----------
        v_minmax : tuple
            Minimum and maximum speed of the trajectory sample. Gain guesses are 
            checked for stability within this speed range. 
        sample_index : int
            The index of the current sample. 
        """

        #initial guesses
        if self.gain_guess_mode is None:
            #if none, use the default gains of the csf model
            self.initial_guesses = None
            self.n_guesses = 1
        elif type(self.gain_guess_mode) is int:
            #if int, generate a random sample of stable gains
        
            logger.info(f"Creating {self.gain_guess_mode} initial gain guesses.")
            
            def print_gain_limits(lim_type, lim_min, lim_max):
                msg = f'Gain {lim_type}: '
                for k in lim_min.keys():
                    msg += f"{k}: [{lim_min[k]:.1f}, {lim_max[k]:.1f}], "
                msg = msg[:-2]
                logger.info(msg)
                
            def csf_lims_to_dicts(csf_lims):
                lim_min = self.bike_class.DYNAMICS_TYPE.csf_to_dict(csf_lims[:,0])
                lim_max = self.bike_class.DYNAMICS_TYPE.csf_to_dict(csf_lims[:,1])
                return lim_min, lim_max
                
            #search limits
            if isinstance(self.gain_search_limits, dict):
                search_limits_csf = self.bike_class.DYNAMICS_TYPE.dict_to_csf(self.gain_search_limits)
            else:
                search_limits_csf = np.array(self.gain_search_limits)
            search_lim_min, search_lim_max = csf_lims_to_dicts(search_limits_csf)
            print_gain_limits("search limits", search_lim_min, search_lim_max)
            
            #create a bike for the stability test
            s0 = (0,0,0,np.mean(v_minmax),0,0,0,0)
            bike = self.bike_class(s0)
   
            #create a stable gain sample as initial guesses
            self.initial_guesses, stable_limits_csf = create_stable_gain_sample(bike, 
                                                                     n=self.gain_guess_mode, 
                                                                     vminmax=v_minmax, 
                                                                     limits=search_limits_csf)
            self.n_guesses = self.initial_guesses.shape[0]
            
            #stability limits
            stable_lim_min, stable_lim_max = csf_lims_to_dicts(stable_limits_csf)
            print_gain_limits("stability limits", stable_lim_min, stable_lim_max)
            
            #identification bounds (same as stability limits but with 0-bounds from search limits)
            bounds_limits_csf = stable_limits_csf
            bounds_limits_csf[search_limits_csf==0.0] = 0.0
            bounds_lim_min, bounds_lim_max = csf_lims_to_dicts(bounds_limits_csf)
            print_gain_limits("identification bounds", bounds_lim_min, bounds_lim_max)
            
            self.gain_bounds = {}
            for key in bounds_lim_min:
                self.gain_bounds[key] = (bounds_lim_min[key], bounds_lim_max[key])
            
            #write_to_file
            self._write_initial_guesses(sample_index, bike.dynamics.param_names)
            
        elif isinstance(self.gain_guess_mode, str):
            if self.gain_guess_mode == 'reuse_existing':
                #Reuse existing guesses if _identify_sample is called multiple times.
                self.n_guesses = self.initial_guesses.shape[0]
            elif self.gain_guess_mode[-6:] == '.guess':
                #Load from file
                self._load_initial_guesses(sample_index)
            
            
    def _make_directories(self, dir_results):
        """ Create the output directories and files.
        """

        #output directory
        self.dir_results = os.path.join(dir_results, self.tag)
        if not os.path.exists(self.dir_results):
            os.makedirs(self.dir_results)
            
        #result files
        self.resultfile = os.path.join(self.dir_results, "rcid.gains")
        self.evalfile = os.path.join(self.dir_results, "rcid.eval")
        self.guessfile = os.path.join(self.dir_results, "rcid.guess")
        self.logfile = os.path.join(self.dir_results, "rcid.log")
        
        #plot files
        self.plot_dir = os.path.join(self.dir_results, "plots")
        if not os.path.exists(self.plot_dir):
            os.mkdir(self.plot_dir)
            
        #solution files
        self.solution_dir = os.path.join(self.dir_results, "solutions")
        if not os.path.exists(self.solution_dir):
            os.mkdir(self.solution_dir)
        
        
    def _init_logging(self):
        """ Initialize the logger. """
        logger.init(dir_out = self.dir_results, filetag = 'rcid', ignore_write_errors=True)
        
        
    def _init_output_files(self):
        """ Initiaalize the output files"""
        #initialisation will be done upon first write to get the correct headers. 
        self.resultfile_header_written = False
        self.guessfile_header_written = False
        self.evalfile_header_written = False
        
        
    def _init_result_lists(self):
        """ Initialize lists to collect results"""
        self.gains = [None] * self.n_samples
        self.speeds = [None] * self.n_samples
        self.poles = [None] * self.n_samples
        self.stability = [None] * self.n_samples
        self.stability_initial_guesses = [None] * self.n_samples
        self.best_guesses = np.zeros(self.n_samples)
        

    def _write_results(self, sample_index, guessid, v_mean, info, gains, poles, evaluation):
        """ Write the results of an idenfication to file. 
        """

        sample_id = self.sample_ids['splits'][sample_index]
        
        # ----------- Write gains and poles -----------
        if not self.resultfile_header_written:
            self._write_resultfile_header(evaluation, poles)
            
        temp1, fname, ftype = fileparts(self.sample_ids['splits'][sample_index])
        fname += ftype

        
        stable = evaluation["stability"]
        igstable = evaluation["stability-initial-guess"]
        msg = f"{sample_index};{sample_id};{guessid};{v_mean};{int(stable)};{int(igstable)};"
        for g in gains:
            msg+=f"{g};"
        for p in poles:
            msg+=f"{p};"

        #remover trailing delimiter and add newline
        msg = msg[:-1]
        msg += '\n'
            
        write_with_recovery(self.resultfile, msg)
        
        # ----------- Write metrics and fit -----------
        #header
        if not self.evalfile_header_written:
            self._write_evalfile_header(evaluation)
            
        #objective value
        msg = (f"{sample_index};{sample_id};{guessid};{v_mean};{info['status']};"
               f"{evaluation['objective']};{np.mean(info['g'])};")
        if "add_stability_penalty" in self.opty_config:
            if self.opty_config["add_stability_penalty"]:
                msg = (f"{sample_index};{sample_id};{guessid};{v_mean};{info['status']};"
                    f"{evaluation['objective']};{evaluation['objective_stab']};"
                    f"{evaluation['objective_sum']};{np.mean(info['g'])};")
        
        #state metrics
        for m in ['VAF', 'MAE']:
            if m not in evaluation:
                continue
            for e in evaluation[m]:
                msg += f"{e};"  
            
        #distance metrics
        for m in ['position_error', 'pose_error', 'lat_error']:
            if m in evaluation:
                msg += f"{evaluation[m]};"
       
        #remover trailing delimiter and add newline
        msg = msg[:-1]
        msg += '\n'
        
        write_with_recovery(self.evalfile, msg)
        

    def _write_solution(self, sample_index, guessid, identifier):
        """ Write the solution of an identification to file. """
    
        output_file=os.path.join(self.solution_dir,
                                 f"sol_{sample_index}-{guessid}")
        
        identifier.write_solution(output_file)
            
    def _write_initial_guesses(self, sample_index, param_names):
        """ Write the initial guesses to file. """
        
        if not self.guessfile_header_written:
            self._write_guessfile_header(param_names)
        
        if self.initial_guesses is None:
            msg = "Using default initial guess."
        else:
            for i in range(self.initial_guesses.shape[0]):
                msg = f"{sample_index};{self.sample_ids["splits"][sample_index]};{i};"
                for j in range(self.initial_guesses.shape[1]):
                    msg += f"{self.initial_guesses[i,j]:.4f};"
                
                #remover trailing delimiter and add newline
                msg = msg[:-1]
                msg += '\n'
                
                write_with_recovery(self.guessfile, msg)
        
        
    def _write_evalfile_header(self, evaluation):
        """ Write the header of the evaluation result file."""
        
        header = "index;sample_id;guess;v_mean;ipopt_status;objective;constraints;"
        if "add_stability_penalty" in self.opty_config:
            if self.opty_config["add_stability_penalty"]:
                header = ("index;sample_id;guess_index;v_mean_ms;ipopt_status;objective;"
                        "objective_stab;objective_sum;constraints;")   
        
        #state metrics
        for m in ['VAF', 'MAE']:
            if m not in evaluation:
                continue
            for s in evaluation['state_names']:
                header += f"{m}_{s};"  
            
        #distance metrics
        for m in ['position_error', 'pose_error', 'lat_error']:
            if m in evaluation:
                header += f"{m};"
                
        #remover trailing delimiter and add newline
        header = header[:-1]
        header += '\n'
        
        write_with_recovery(self.evalfile, header, write_mode='w')
        self.evalfile_header_written = True
    
    
    def _write_resultfile_header(self, evaluation, poles): 
        """ Write the header of the identification result file."""
        
        header = f"index;sample_id;guess;v_mean;stability;guess_stability;"
        
        for p in evaluation['param_names']:
            header += f"{p};"
            
        for i in range(poles.size):
            header += f"pole{i};"
        
        #remover trailing delimiter and add newline
        header = header[:-1]
        header += '\n'
        

        write_with_recovery(self.resultfile, header, write_mode='w')
        
        self.resultfile_header_written = True
        
        
    def _write_guessfile_header(self, param_names):
        """ Write the header of the guess file"""
        
        header = "index;sample_id;guess;"
        
        for g in param_names:
            header += f"{g};"
        
        #remover trailing delimiter and add newline
        header = header[:-1]
        header += '\n'
        
        write_with_recovery(self.guessfile, header, write_mode='w')
        self.guessfile_header_written = True
        
        
    def _plot_sample_result(self, sample_index, guessid, identifier):
        """ Plot the identfication result of a single sample."""
        
        identifier.plot(
            output_file=os.path.join(self.plot_dir,
                                     f"traj_{sample_index}-{guessid}.png")
            )
        if self.close_plots:
            plt.close("all")
        
        identifier.print()
        
    def _plot_best(self, sample_index, data_dict, name,  mode='data', write=True):
        """ Plot the best identification result. """
        
        id_best_guess = int(self.best_guesses[sample_index])
        
        solution_file = os.path.join(self.solution_dir, 
                                     f"sol_{sample_index}-{id_best_guess}.pkl")
        

        solution_dict = load_solution_dict(solution_file)
        if mode == 'data':
            n = solution_dict['n_data']
        elif mode == 'all':
            n = solution_dict['n_all']
        else:
            raise ValueError((f"mode must be either 'data' or 'all'! Instead "
                              f"it was '{mode}'."))
        
        state_keys = ['p_x', 'p_y', 'psi', 'v', 'delta', 'phi', 'p_y_c']
        t = solution_dict['t_s'] * np.arange(n)
        
        fig, axes = plt.subplots(len(state_keys), 1, sharex=True)
        
        #track axis scaling
        maxmin = np.zeros((9,2))
        maxmin[:,0] = np.inf
        maxmin[:,1] = -np.inf        
        def test_maxmin(data, i):
            
            datamin = np.amin(data)
            if np.isfinite(datamin):
                maxmin[i,0] = min(datamin, maxmin[i,0])
            
            datamax = np.amax(data)
            if np.isfinite(datamax):
                maxmin[i,1] = max(datamax, maxmin[i,1])
        
        for i in range(len(state_keys)):
            
            skey = state_keys[i]
            axes[i].set_ylabel(skey)
            
            axes[i].plot(t, data_dict[skey][:n], label = 'measured')
            test_maxmin(data_dict[skey][:n], i)
            
            for key in ['initial_guess', 'solution']:
                if skey in solution_dict[key]['states']:
                    states = solution_dict[key]['states']
                
                    if key == 'initial_guess':
                        axes[i].plot(t, states[skey][:n], label = key, 
                                     color='gray')
                    else:
                        
                        data = states[skey][:n]
                        axes[i].plot(t, data, label = key)
                        test_maxmin(data, i)
                    
        #plot xy
        fig2, axes2 = plt.subplots(1,1)
        fig2.set_size_inches(13, 5)
        axes2.set_aspect('equal')
        axes2.set_ylabel('y')
        axes2.set_xlabel('x')
        
        axes2.plot(data_dict['p_x'][:n], 
                     data_dict['p_y'][:n], label = 'measured')
        test_maxmin(data_dict['p_x'][:n], 7)
        test_maxmin(data_dict['p_y'][:n], 8)
        
        
        for key in ['initial_guess', 'solution']:
            states = solution_dict[key]['states']
            
            if key == 'initial_guess':
                axes2.plot(states['p_x'][:n], states['p_y'][:n], 
                              label = key, color='gray')
            else:
                axes2.plot(states['p_x'][:n], states['p_y'][:n], label=key)
                test_maxmin(states['p_x'][:n], 7)
                test_maxmin(states['p_y'][:n], 8)
                
        axes2.legend()
        axes2.set_title((f'Best identification result {name}/'
                   f'{id_best_guess}'))
        axes[0].set_title((f'Best identification result {name}/'
                   f'{id_best_guess}'))
        
        #set axis scaling
        for i in range(len(state_keys)):
            axes[i].set_ylim(maxmin[i,0]-.1*np.abs(maxmin[i,0]), 
                             maxmin[i,1]+.1*np.abs(maxmin[i,1]))
            axes[i].set_xlim(t[0], t[-1])    
            
        axes2.set_ylim(maxmin[8,0]-.1*np.abs(maxmin[8,0]), 
                          maxmin[8,1]+.1*np.abs(maxmin[8,1]))
        axes2.set_xlim(maxmin[7,0]-.1*np.abs(maxmin[7,0]), 
                          maxmin[7,1]+.1*np.abs(maxmin[7,1]))
        
        if write:
            output_file_1=os.path.join(self.plot_dir,
                                     f"trajt_{sample_index}-{id_best_guess}.png")
            output_file_2=os.path.join(self.plot_dir,
                                     f"trajxy_{sample_index}-{id_best_guess}.png")
            fig.savefig(output_file_1, dpi=300)
            fig2.savefig(output_file_2, dpi=300)
        

    def _plot_gains(self, param_names):
        """ Plot the identified gains
        """
        
        l = None
        
        fig, axes = plt.subplots(1, len(param_names), sharex=True,sharey=True,figsize=(15,10))
        
        plt.suptitle((f'Rider control identification\n{self.tag}'))
        
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        
        for ax, pname in zip(axes, param_names):
            ax.set_ylabel(pname)
            ax.grid()
        
        for i in range(self.n_samples):
            for j in range(len(param_names)):
                gains_plot = self.gains[i][:,j]
                axes[j].plot(self.speeds[i], self.gains[i][int(self.best_guesses[i]),j], 
                           linestyle='None', marker='o', markeredgecolor='red',
                           markerfacecolor='none', markersize = 10)
                
                for k in range(self.n_guesses):
                    if k == 0:
                        markerstyle = {
                            'marker': 'o',
                            'linestyle': 'None',
                            'markersize': 5,
                            }
                    else:
                        markerstyle = {
                            'marker': 'o',
                            'linestyle': 'None',
                            'markersize': 5,
                            'color': l[-1].get_color()
                            }
                        
                    if self.stability[i][k]:
                        markerstyle['fillstyle'] = 'full'
                    else:
                        markerstyle['fillstyle'] = 'none'
                    
                    l = axes[j].plot(self.speeds[i], gains_plot[k], **markerstyle)
                    
                    if k == 0 and j==0:
                        l = axes[j].plot(self.speeds[i], gains_plot[k], **markerstyle, label=f"split#{i}")
                    else:
                        l = axes[j].plot(self.speeds[i], gains_plot[k], **markerstyle)
                        

        fig.legend()
        fig.savefig(os.path.join(self.plot_dir,"gains.png"), dpi=300)
        
        if self.close_plots:
            plt.close(fig)


    def _plot_poles(self):
        """ Plot the identified poles
        """
        
        l = None
        
        fig, ax = plt.subplots(1,1,figsize=(15,10))
        ax.set_ylabel("Im")
        ax.set_xlabel("Re")
        ax.set_title((f'Identified poles\n{self.tag}'))
        ax.grid()
        
        for i in range(self.n_samples):
            
            for j in range(self.n_guesses):
                
                if j == 0:
                    markerstyle = {
                        'marker': 'o',
                        'linestyle': 'None',
                        'markersize': 5,
                        }
                else:
                    markerstyle = {
                        'marker': 'o',
                        'linestyle': 'None',
                        'markersize': 5,
                        'color': l[-1].get_color()
                        }
                
                if self.stability[i][j]:
                    markerstyle['fillstyle'] = 'full'
                else:
                    markerstyle['fillstyle'] = 'none'
                
                p = self.poles[i][j,:]
                if j==0:
                    l = ax.plot(np.real(p), np.imag(p), **markerstyle, label=f"split#{i}")
                else:
                    l = ax.plot(np.real(p), np.imag(p), **markerstyle)
            
            best_poles = self.poles[i][int(self.best_guesses[i]),:]
            ax.plot(np.real(best_poles), np.imag(best_poles), 
                    linestyle='None', marker='o', markeredgecolor='red',
                    markerfacecolor='none', markersize = 10)
            
        fig.legend()
        fig.savefig(os.path.join(self.plot_dir,"poles.png"), dpi=300)
        
        if self.close_plots:
            plt.close(fig)
            
    def _load_data(self, sample_index):
        """ Load the trajectory data of the current sample. Apply timeshift. 
        """

        dataman = RCIDDataManager(self.paths.getdir_data_processed())
        trk = dataman.load_split(self.sample_ids['splits'][sample_index], subset='steps')
        data_dict = trk.to_dict()
        data_dict['t'] = np.array(trk.get_relative_time())
    
        data_dict['target_locations'] = self.target_locations
            
        if not self.command_time_shift is None:
            
            if isinstance(self.command_time_shift, (list, tuple, np.ndarray)):
                shift = self.command_time_shift[sample_index]
            else:
                shift = self.command_time_shift
            
            if self.max_offset is not None:
                n_samples_max = len(data_dict['t']) - self.max_offset 
            else:
                n_samples_max = None
            logger.warning(f"command-time-shift: {shift} "
                           f"timesteps / {shift * T_S:.2f} s") 
            data_dict = apply_timeshift(shift,
                                        data_dict,
                                        ('p_x_c', 'p_y_c'),
                                        n_samples_max)
        else:
            shift = None
        
        v_mean = np.mean(data_dict['v'])
        v_minmax = [np.min(data_dict['v']), np.max(data_dict['v'])]
        
        
        self.speeds[sample_index] = v_mean
        logger.info(f"number of timesteps in data: {len(data_dict['t'])}")
        
        return data_dict, v_minmax, v_mean, shift
    
    
    def _load_initial_guesses(self, sample_index):
        """ Load an initial guess from file."""
        
        df = pd.read_csv(self.guessfile, sep=';')
        
        gain_dict = df[df["split"]==sample_index]
        
        self.initial_guesses = self.bike_class.DYNAMICS_TYPE.dict_to_csf(gain_dict)
        self.n_guesses = self.initial_guesses.shape[0]
        
        logger.info(f'loaded initial guesses from {self.guessfile}')
        
        
    def _identify_sample(self, sample_index):
        """ Run the identification for a single sample. """
        
        t0_split = time()
        plt.close("all")
        
        # ------ Get split data -----------------------------------------------
        data_dict, v_minmax, v_mean, shift = self._load_data(sample_index)
        
        
        # ------ Identify -----------------------------------------------------
        
        self._make_gain_guesses(v_minmax, sample_index)
        self.opty_config['gain_bounds'] = self.gain_bounds
        
        if shift is not None:
            name = (f'{self.tag}_shift{shift}/{sample_index}')
        else:
            name = (f'{self.tag}/{sample_index}')
            
        #logger.info(f"Identifying split {sample_index}")
        identifier = ControlIdentifier(name, data_dict, **self.opty_config)   
                
        self.gains[sample_index] = np.zeros((self.n_guesses, 5))
        self.poles[sample_index] = np.zeros((self.n_guesses, 5), dtype=complex)
        self.stability[sample_index] = np.zeros((self.n_guesses),dtype=bool)
        self.stability_initial_guesses[sample_index] = np.zeros((self.n_guesses),dtype=bool)
        self.error_array = np.zeros((self.n_guesses))
        self.constraint_array = np.zeros((self.n_guesses))
        self.status_array = np.zeros((self.n_guesses))
        
        for guessid in range(self.n_guesses):
            t0_guess = time()
            try: 
                if self.initial_guesses is not None:
                    
                    #log the current initial guess
                    initial_guess_dict = self.bike_class.DYNAMICS_TYPE.csf_to_dict(
                        self.initial_guesses[guessid,:])
                    msg = f"Using initial guess {guessid}: "
                    for k, v in initial_guess_dict.items():
                        msg += f"{k}: {v:.1f}, " 
                    logger.info(msg[:-2])
                                
                    #fit
                    solution, info, gains, poles, evaluation = identifier.fit(
                        initial_guess=self.initial_guesses[guessid,:],
                        feasible=self.feasible_initial_guesses)
                else:
                    #fit
                    solution, info, gains, poles, evaluation = identifier.fit(
                        feasible=self.feasible_initial_guesses)
                
                self.param_names = evaluation['param_names']
                self.gains[sample_index][guessid,:] = gains
                self.poles[sample_index][guessid,:] = poles
                self.stability[sample_index][guessid] = evaluation['stability']
                self.stability_initial_guesses[sample_index][guessid] = evaluation['stability-initial-guess']
                self.error_array[guessid] = evaluation['objective']
                self.constraint_array[guessid] = np.mean(info['g'])
                self.status_array = info['status']
                
                self._write_results(sample_index, 
                                    guessid, 
                                    v_mean, 
                                    info, 
                                    gains,
                                    poles,
                                    evaluation)
                
                if self.plot_mode == 'all':
                    self._plot_sample_result(sample_index, guessid, v_mean, 
                                            identifier, gains)
                    
                self._write_solution(sample_index, guessid, identifier)
                logger.info((f"Finished identifying split {sample_index}, guess {guessid}: status = "
                                 f"{info['status']}, duration = {time()-t0_guess:.2f} s"))
                logger.info(info['status_msg'].decode("utf-8"))
            except NoForwardSimulationException:
                logger.info(f"Bad initial guess: {guessid}")
            
        #Best guess
        self._find_best(sample_index)
        
        if self.plot_mode == 'best':
            self._plot_best(sample_index, data_dict, name)
            
        #log time
        t = time() - t0_split
        logger.info(f"Identifying split {sample_index} took {s2hms(t)} ({t/self.n_guesses:.2f} s per guess)")
        
        
    def _find_best(self, sample_index):
        """ Find the gain guess leading to the best result."""
        
        order = np.argsort(self.error_array)
        stability_sorted = self.stability[sample_index][order]
        
        if np.any(stability_sorted):
            best_guess = order[np.min(np.argwhere(stability_sorted))]
            logger.info((f"Best stable guess: guess {best_guess} with obj = "
                         f"{self.error_array[best_guess]}"))
        else:
            best_guess = np.argmin(self.error_array)
            logger.info((f"All guesses unstable! Best is guess {best_guess} "
                         "with obj = {self.error_array[best_guess]}"))
        
        self.best_guesses[sample_index] = best_guess
        
        
    def identify(self):
        
        t0 = time()
        
        #log run identification
        logger.info("Starting rider control identification")
        logger.info(f"dir_data: {self.dir_data}")
        logger.info(f"session_id: {self.session_id}")
        logger.info(f"resultfile: {self.resultfile}")
        logger.info(f"evalfile: {self.evalfile}")
        logger.info(f"guessfile: {self.guessfile}")
        if self.note is not None:
            logger.info(f"note: {self.note}")
        
        logger.info(f"command time shift: {self.command_time_shift}")
        logger.info(f"feasible initial guesses: {self.feasible_initial_guesses}")
        logger.info(f"Cyclist model: {self.bike_class.__name__}")
        
        for key in self.opty_config:
            logger.info(f"{key}: {self.opty_config[key]}")
            
            
        ## LOAD EXPERIMENT DATA
        self._init_result_lists()   

        ## IDENTIFY SPLITS
        for split in range(self.n_samples):
            logger.info(f'Identifying split {self.sample_ids['splits'][split]} ({split+1}/{self.n_samples}) ')
            
            self._identify_sample(split)
            
        logger.info(f"Finished identifying all splits of this run.")
        #log time
        t = time() - t0
        logger.info((f"Identifying {self.n_samples} splits took {s2hms(t)} "
                     f"({s2hms(t/self.n_samples)} per split)"))
        
        ## Plot gains and poles
        if self.plot_mode != 'none':
            self._plot_gains(self.param_names)
            self._plot_poles()
        

class ReactionTimeIDSessionManager(IDSessionManager):
    """ Manage an ID session. Create gain guesses, run ID for multiple guesses and collect and
    store results. Repeat identification for multiple command time shifts to find best
    response time.

    This approach was dropped after the results proved to be not satisfactory. 
    Use with caution. Code may contain errors. 
    """
    
    def __init__(self, response_times, *args, **kwargs):
        """
        Replaces command_time_shift of the super constructor with an response
        time array. 
        """
        
        if 'command_time_shift' in kwargs.keys():
            if kwargs['command_time_shift'] is not None:  
                kwargs['command_time_shift'] = None
                warn_invalid_arg_cts = True
        else:
            warn_invalid_arg_cts = False
        
        super().__init__(*args, **kwargs)
        
        self.response_times = response_times
        self.base_tag = self.tag
        self.n_gains = 5
        
        if warn_invalid_arg_cts:
            logger.warning("Ignoring unused argument 'command_time_shift'")
            
    def _parse_calibration(self, calibration_dict):
        """ Additionally parse sample time """
        try:
            super()._parse_calibration(calibration_dict)
        except KeyError:
            msg = (f"The calibration dict must have the keys 'sample_time', 'target_locations', and"
                   f" 'steer_bias_correction'. Instead it had {calibration_dict.keys()}")
            raise KeyError(msg)
            
    
    def _make_directories(self, dir_results):
        """
        Overwrite _make_directories to only create toplevel directory without result files
        """
        
        #toplevel output directory
        self.base_dir_results = os.path.join(dir_results, self.tag)
        self.dir_results = self.base_dir_results
        if not os.path.exists(self.base_dir_results):
            os.makedirs(self.base_dir_results)
        
    def _make_subdirectory(self, t_r):
        """
        Use super()._make_directories to create subdirectories for control id with individual
        discrete response times.
        
        The subdirectories are named trXXXXX, where XXXXX is the response time in ms. 

        """
        self.tag = f'tr{int(t_r*1000):05}'
        if self.rtid > 0:
            guessfile = self.guessfile
        super()._make_directories(self.base_dir_results)
        if self.rtid >0:
            self.guessfile = guessfile
        
    def _load_data(self, sample_index):
        """ Load the trajectory data of the current sample. Apply timeshift. 
        """
        
        dataman = RCIDDataManager(self.dir_data)
        data_dict = dataman.load_track(self.sample_ids['splits'][sample_index])
        
        data_dict['delta'] = data_dict['delta'] + self.steer_bias_correction
        logger.warning(f"Corrected delta_m by {self.steer_bias_correction * 180/np.pi}")
    
        data_dict['target_locations'] = self.target_locations
            

        i_commands = np.argwhere(np.diff(data_dict['p_y_c'])).flatten()
        if i_commands.size > 1:
            i_2ndcommand = i_commands[1]
        else:
            i_2ndcommand = data_dict['t'].size
            
        n_samples_max = i_2ndcommand - self.max_offset 
        n_samples_max = None

        logger.warning(f"command-time-shift: {self.command_time_shift} "
                       "timesteps")
        data_dict = apply_timeshift(self.command_time_shift,
                                    data_dict,
                                    ('p_x_c', 'p_y_c'),
                                    n_samples_max)
        
        v_mean = np.mean(data_dict['v'])
        v_minmax = [np.min(data_dict['v']), np.max(data_dict['v'])]
        
        
        self.speeds[sample_index] = v_mean
        logger.info(f"number of timesteps in data: {len(data_dict['t'])}")
        
        return data_dict, v_minmax, v_mean, None
        
    def _identify_sample(self, sample_index):
        """
        Add recording objective values to _identify_sample()

        """
        
        # reuse the existing guesses from the previous identification round 
        if self.rtid > 0:
            self.gain_guess_mode = 'reuse_existing'
            self.initial_guesses = self.initial_guesses_all_splits[sample_index]
        
        super()._identify_sample(sample_index)
        self.rt_errors[:,sample_index,self.rtid] = self.error_array
        self.rt_gains[:,:,sample_index, self.rtid] = self.gains[sample_index]
    
        # store initial guesses for later identification rounds
        if self.rtid == 0.0:
            self.initial_guesses_all_splits.append(self.initial_guesses)

        

    def identify(self):
        """ Run the identification session.
        """
        
        assert isinstance(self.gain_guess_mode, int), 'Explicit initial guesses not supported!'
        self.n_guesses = self.gain_guess_mode
        self.initial_guesses_all_splits = []
        self.rtid = 0
        self.max_offset = int(np.round(self.response_times[-1]/T_S))
    
        self.rt_errors = np.zeros((self.n_guesses, self.n_samples, len(self.response_times)))
        self.rt_gains = np.zeros((self.n_guesses, self.n_gains, self.n_samples, len(self.response_times)))
        
        for t_r in self.response_times:
                
            self._make_subdirectory(t_r)
            self.command_time_shift = int(np.round(t_r/T_S))
            self.note = (f"Identification for response time sample r_t ="
                         f"{t_r:.4f} s ({self.rtid+1}/{len(self.response_times)})")
            super().identify()
            
            
            
            self.resultfile_header_written = False
            self.evalfile_header_written = False
            
            self.rtid+=1
    
    
# -------------- Module functions ------------------------------------------------------------------

def load_solution_dict(path):
    """
    Loads the dictionary generated by ControlIdentifier.write_solution()
    """
    
    with open(path, 'rb') as file:
        solution_dict = pickle.load(file)
        
    return solution_dict
    

def dict_to_csf(gain_dict):
    """
    Convert a gain dict to the gain array convention of cyclist social forces.

    Parameters
    ----------
    gain_dict : dictionary
        Gain dictionary with the keys k_phi, k_delta, k_phidot, k_deltadot
        and k_psi

    Returns
    -------
    csf_gain_array : array
        1 x 5 gain array in the order k_phi, k_delta, k_phidot, k_deltadot, 
        k_psi.

    """
    if gain_dict is None:
        return None
    csf_gain_array = np.array([gain_dict["k_phi"], 
                           gain_dict["k_delta"], 
                           gain_dict["k_phidot"], 
                           gain_dict["k_deltadot"],
                           gain_dict["k_psi"]])
    csf_gain_array = np.reshape(csf_gain_array, (1,5))
    return csf_gain_array


def csf_to_dict(csf_gain_array):
    """
    Convert the gain array convention of cyclist social forces to a gain dict.

    Parameters
    ----------
    csf_gain_array : array
        1 x 5 gain array in the order k_phi, k_delta, k_phidot, k_deltadot, 
        k_psi.

    Returns
    -------
    gain_dict : dictionary
        Gain dictionary with the keys k_phi, k_delta, k_phidot, k_deltadot
        and k_psi

    """
    csf_gain_array = csf_gain_array.flatten()
    gain_dict = {
        "k_phi": csf_gain_array[0],
        "k_delta": csf_gain_array[1],
        "k_phidot": csf_gain_array[2],
        "k_deltadot": csf_gain_array[3],
        "k_psi": csf_gain_array[4],
    }
    return gain_dict