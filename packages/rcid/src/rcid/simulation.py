# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 15:32:19 2024

Simulate results for the rider control identification ZigZag experiments for
measurement data validation and control identification validation. 

Overwrites some cyclistsocialforce classes with custom functionalities for this
application. 

@author: Christoph M. Konrad
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Circle
from shapely.plotting import plot_polygon

from cyclistsocialforce.vehicle import BalancingRiderBicycle, PlanarPointBicycle, UncontrolledVehicle, calc_direct_approach_dest_force
from cyclistsocialforce.intersection import SocialForceIntersection
from cyclistsocialforce.scenario import Scenario
from cyclistsocialforce.dynamics import BalancingRiderDynamics, PlanarPointDynamics
from cyclistsocialforce.dynamics import from_gains, test_stability
from cyclistsocialforce.utils import FIFOBuffer
from cyclistsocialforce.parameters import BalancingRiderBicycleParameters
from rcid.params.balanceassist_bikeparams import balanceassistv1_with_averagerider

from pytrafficutils.agents import RectObstacle, Bicycle
from pytrafficutils.ssm import SurrogateSafetyMetricProcessor

from mypyutils.misc import none_switch

from PIL import Image, ImageDraw

from pypaperutils.design import TUDcolors
tudcolors = TUDcolors()
cmap = tudcolors.colormap()

T_S = 0.01

class StopSimulationInterrupt(Exception):
    """Use this exception as an interrupt to stop simulation at a certain condition"""
    def __init__(self, reason):
        msg = f"Simulation stopped with reason: {reason}"
        super().__init__(msg)

class Collision(Exception):
    pass

# ------------------------------------------------------------------------------------
#                               DYNAMICS CLASSES
# ------------------------------------------------------------------------------------  
class FixedSpeedBalancingRiderDynamics(BalancingRiderDynamics):
    """ Overwrites speed control with the desired speed. The bicycle always travels at
    the currently desired speed. To have the bike follow a given speed trajectory, 
    update the desired speed in 'bicycle.params.v_desired_default'.

    The dynamics keep a fixed gain set, even though the speed is changing. If poles are 
    provided, make sure that bicycle.params.v_desired_default corresponds to the speed
    associated with the desired poles/gains (i.e. the mean speed of the sample.)
    """

    def __init__(self, bicycle):
        """ Initialize a FixedSpeedBalancingRiderDynamics object.

        Fixes the gains at the the values associated with the speed saved in bicycle.params.v_desired_default.
        """

        self.fix_gains = False
        super().__init__(bicycle)
        self.gains = self._get_gains(bicycle.params.v_desired_default)
        self.fix_gains = True


    def _get_gains(self, v):
        """ The identification result has a speed trajectory (different speeds at every time step), but 
        a single gain set associated with the trajectory. After getting the gains for the mean speed of 
        the identification sample, the gains should be fixed and not updated with speed. 

        This function overwrites BalancingRiderDynamics._get_gains(v) to ensure that the gains are fixed.
        """

        if self.fix_gains:
            return self.gains
        else:
            return super()._get_gains(v)

    def _step_speed(self, bicycle, Fx, Fy):
        """ Overwrite the speed control. Set speed to currently desired speed. 
        """
        return bicycle.params.v_desired_default


    def dict_to_csf(gain_dict):
        """
        Convert an identification gain dict to the gain array convention of cyclist social forces.

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
        if csf_gain_array.ndim == 1:
            csf_gain_array = np.reshape(csf_gain_array, (1,5))
        return csf_gain_array


    def csf_to_dict(csf_gain_array):
        """
        Convert the gain array convention of cyclist social forces to a identification gain dict.

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
        
        
class FixedSpeedPlanarPointDynamics(PlanarPointDynamics):
    """ Overwrites speed control with the desired speed. The bicycle always travels at
    the currently desired speed. To have the bike follow a given speed trajectory, 
    update the desired speed in 'bicycle.params.v_desired_default'.
    """

    def _step_speed(self, bicycle, Fx, Fy):
        """ Overwrite the speed control. Set speed to currently desired speed. 
        """
        return bicycle.params.v_desired_default
        
        
    def dict_to_csf(gain_dict):
        """
        Convert a gain dict to the gain array convention of cyclist social forces.

        Parameters
        ----------
        gain_dict : dictionary
            Gain dictionary with the keys k_psi

        Returns
        -------
        csf_gain_array : array
            1 x 1 gain array in the order k_psi.

        """
        if gain_dict is None:
            return None
        csf_gain_array = np.array([gain_dict["k_psi"]])
        if csf_gain_array.ndim == 1:
            csf_gain_array = np.reshape(csf_gain_array, (1,1))
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
        gain_dict = {"k_psi": csf_gain_array[0]}
        return gain_dict

# ------------------------------------------------------------------------------------
#                               PARAMETER CLASSES
# ------------------------------------------------------------------------------------
class Balanceassistv1BalancingRiderParameters(BalancingRiderBicycleParameters):
    """ Overwrite default BalancingRiderParameters class to set bicycle parameters
    to the paramters of the Balanceassist bike. 
    """
    def __init__(self, **kwargs):
        kwargs['bicycleParameterDict'] = balanceassistv1_with_averagerider
        super().__init__(**kwargs)



# ------------------------------------------------------------------------------------
#                               VEHICLE CLASSES
# ------------------------------------------------------------------------------------
class FixedSpeedBalancingRiderBicycle(BalancingRiderBicycle):
    """ Balancing Rider Bicycle with fixed speed and balanceassistv1 parameters.
    """
    DYNAMICS_TYPE = FixedSpeedBalancingRiderDynamics
    PARAMS_TYPE = Balanceassistv1BalancingRiderParameters



class FixedSpeedPlanarPointBicycle(PlanarPointBicycle):
    """ Planar Point Bicycle with fixed speed.
    """
    DYNAMICS_TYPE = FixedSpeedPlanarPointDynamics



class FixedResponseTimePlanarPointBicycle(FixedSpeedPlanarPointBicycle):
    """ Planar Point Bicycle with fixed speed and delayed responses to desired destination changes. 
    """
    
    def __init__(self, *args, response_time=0.0, **kwargs):
        """ Initialize a Particle Bicycle with delayed responses to desired destination changes.

        Paramters
        ---------
        response_time : float, optional
            The destination response delay time in seconds. Default is 0.0.
        """
        super().__init__(*args, **kwargs)
        self.response_time = response_time
        self._initialize_response_buffer(self.dest, response_time)


    def _initialize_response_buffer(self, dest, response_time):
        if response_time > 0:
            n = int(round(response_time/self.params.t_s))
            destlist = []
            for i in range(n):
                destlist.append(dest)
            self.dest_response_buffer = FIFOBuffer(destlist)
        else:
            self.dest_response_buffer = None


    def updateDestination(self):
        """ Update the current destination.

        Delays destination updates by the fixed destination 
        """
        super().updateDestination()
        if self.dest_response_buffer is not None:
            self.dest = self.dest_response_buffer.next(self.dest)
    
    
    def setDestinations(self, x, y, stop=None, reset=False):
        super().setDestinations(x, y, stop, reset)

        if reset:
            if stop is None:
                stop = np.zeros_like(x)
            self._initialize_response_buffer(np.array([x[0], y[0], stop[0]]), self.response_time)


# ------------------------------------------------------------------------------------
#                            SIMULATION SCENARIO CLASSES
# ------------------------------------------------------------------------------------    

class FixedSpeedStaticObstacleAvoidance(Scenario):
    """ The obstacle avoidance test scenario.
    """

    INITIAL_CONDITION = np.zeros(8)

    OBSTACLE_PARAMS = dict(dist=15, width=1, depth=1.5)

    BIKEMODELS = {
        'balancingrider': FixedSpeedBalancingRiderBicycle,
        'planarpoint': FixedResponseTimePlanarPointBicycle
    }

    def __init__(self, bikemodel, speed, poles, id, response_time=0.0, **scenario_kwargs):

        self.speed = speed
        self.destinations = self.get_destinations()
        
        # bikemodel
        if bikemodel in self.BIKEMODELS:
            self.bikemodel = bikemodel
        else:
            raise ValueError (f"Unknown bikemodel {bikemodel}! Choose any of {list(self.BIKEMODELS.keys())}")
        
        bike_type = self.BIKEMODELS[bikemodel]

        # bike arguments
        bike_params_args = dict(v_desired_default=speed, poles=poles)
        if bike_type is FixedSpeedBalancingRiderBicycle:
            s0 = self.INITIAL_CONDITION
            s0[3] = speed
            params = bike_type.PARAMS_TYPE(**bike_params_args)
            bike_kwargs = dict(id=f"pred-{id}", params=params)
        else:
            s0 = np.zeros(4)
            s0[:3] = self.INITIAL_CONDITION[:3] 
            s0[3] = speed
            params = bike_type.PARAMS_TYPE(**bike_params_args)
            bike_kwargs = dict(id=f"pred-{id}", params=params, response_time=response_time)
        
        # make bike
        self.bike = bike_type(s0, **bike_kwargs)
        self.bike.dest_force_func = calc_direct_approach_dest_force
        self.bike.setDestinations(self.destinations[:, 0], self.destinations[:, 1], reset=True)
            
        #define interaction scenatio 
        self.t_s = self.bike.params.t_s
        if not "t_r" in scenario_kwargs:
            scenario_kwargs["t_r"] = 0
        super().__init__(self._step_func, **scenario_kwargs)

        #response time queue
        
        self.response_time_queue = FIFOBuffer(self.bike.dest)
        

    @classmethod
    def get_destinations(cls):

        p0 = cls.INITIAL_CONDITION[:3]
        
        destinations = np.array(
            [[p0[0]+5, p0[1]],
             [p0[0]+cls.OBSTACLE_PARAMS['dist'] + 0.5 * cls.OBSTACLE_PARAMS['depth'], p0[1]-cls.OBSTACLE_PARAMS['width']-1.5],
             [p0[0]+cls.OBSTACLE_PARAMS['dist'] + cls.OBSTACLE_PARAMS['depth'] + 35, p0[1]]])
        
        return destinations
    

    @classmethod
    def get_obstacle(cls):

        p0 = cls.INITIAL_CONDITION[:3]

        obstacle = np.array(
            [[p0[0]+cls.OBSTACLE_PARAMS['dist'], p0[1]-cls.OBSTACLE_PARAMS['width']],
             [p0[0]+cls.OBSTACLE_PARAMS['dist'], p0[1]+cls.OBSTACLE_PARAMS['width']],
             [p0[0]+cls.OBSTACLE_PARAMS['dist'] + cls.OBSTACLE_PARAMS['depth'], p0[1]+cls.OBSTACLE_PARAMS['width']],
             [p0[0]+cls.OBSTACLE_PARAMS['dist'] + cls.OBSTACLE_PARAMS['depth'], p0[1]-cls.OBSTACLE_PARAMS['width']]])

        return obstacle
    
    @classmethod
    def make_scenario_plot(cls, model, speed, radius=2, ax=None, draw_labels=True, footprint_kwargs={}):

        default_footprint_kwargs = dict(add_points=False, edgecolor=None, facecolor=tudcolors.get('blauw'), ax=ax)
        footprint_kwargs = default_footprint_kwargs | footprint_kwargs
        
        if ax is None:
            _, ax = plt.subplots(1,1, layout='constrained')

        ax.set_aspect("equal")
        if draw_labels:
            ax.set_ylabel("y")
            ax.set_xlabel("x")
            ax.set_title(f"Obstacle avoidance test for model {model}, v={speed*3.6:.1f}")

        obstacle = cls.get_obstacle()
        destinations = cls.get_destinations()

        polygon = Polygon(obstacle, closed=True, facecolor='dimgray', edgecolor='black')
        ax.add_patch(polygon)

        ax.scatter(destinations[:, 0], destinations[:, 1], color='lightgray', marker='x', zorder=3)

        for (x, y) in destinations:
            circle = Circle((x, y), radius=radius, edgecolor='lightgray', facecolor='none', linestyle='--')
            ax.add_patch(circle)

        bicycle = Bicycle([0, 0.015, 0.01], cls.INITIAL_CONDITION[:4].reshape(1,4))
        plot_polygon(bicycle.footprint.get_polygons_at(cls.INITIAL_CONDITION[:3].reshape(1,3))[0], **footprint_kwargs)
        
        return ax

    def _step_func(self):
        """
        A custom step function for this scenario class. Overwrites
        Scenario._step_func().

        Propagates the bicycle.
        """
        Fx, Fy = self.bike.calcDestinationForce()
        self.bike.step(Fx,Fy)

    def run(self, t=None):

        if t is None:
            # calculate simulation time from approximate travel distance and speed. 
            p0 = self.INITIAL_CONDITION[:2].reshape(1,2)    
            keypoints = np.concatenate((p0, self.destinations), axis=0)
            total_distance = np.sum(np.linalg.norm(np.diff(keypoints, axis=0), axis=1))
            t = round(0.9 * total_distance / self.speed, 2)

        N = int(round(t/self.t_s))
        
        self.ttc = np.zeros(N)
        self.dist = np.zeros(N)
        
        try:
            super().run(t)
        except Collision as col:
            print(col)

    def analyze_TTC(self, ax=None, ttc_marker_kwargs={}, footprint_kwargs={}):

        #plotting configuration
        default_footprint_kwargs = dict(add_points=False, ax=ax, edgecolor=None, facecolor=tudcolors.get('blauw'), alpha=0.1)
        footprint_kwargs = default_footprint_kwargs | footprint_kwargs

        default_ttc_marker_kwargs = dict(color=tudcolors.get('rood'), linestyle='none', marker=".", zorder=10000)
        ttc_marker_kwargs = default_ttc_marker_kwargs | ttc_marker_kwargs

        t = [0, self.i*T_S, T_S]

        p0 = self.INITIAL_CONDITION[:2]
        s_obstacle = [p0[0]+self.OBSTACLE_PARAMS['dist'], p0[1], 0]
        
        obstacle = RectObstacle(t, s_obstacle, length=self.OBSTACLE_PARAMS['depth'], width=2*self.OBSTACLE_PARAMS['width'], anchor=[-self.OBSTACLE_PARAMS['depth']/2, 0])
        bicycle = Bicycle(t, self.bike.traj[:4,:self.i].T)

        ssm = SurrogateSafetyMetricProcessor([obstacle, bicycle], t_s=T_S, verbose=False)
        ttc_results, _ = ssm.eval_TTC(T_extrapolation=5)

        ttc = ttc_results['ttc_min'].iloc[0]
        ttc_mean = ttc_results['ttc_mean'].iloc[0]
        t_ttc_min = ttc_results['t_ttc_min'].iloc[0]

        if not (ax is None):

            x = ttc_results['x2_ttc_min'].iloc[0]
            y = ttc_results['y2_ttc_min'].iloc[0]

            if ttc == 0.0:
                psi = ttc_results['psi2_ttc_min'].iloc[0]
                s = np.array([[x, y, psi]])
                plot_polygon(ssm.agents[1].footprint.get_polygons_at(s)[0], **footprint_kwargs)
       
            p_marker = ax.plot(x, y, **ttc_marker_kwargs)  

        return ttc, t_ttc_min, ttc_mean, ax


    def plot(self, ax, **plot_kwargs):

        if not 'linewidth' in plot_kwargs:
            plot_kwargs['linewidth'] = 1
        if not 'alpha' in plot_kwargs:
            plot_kwargs['alpha'] = 0.1
        if not 'color' in plot_kwargs:
            plot_kwargs['color']=tudcolors.get("blauw")
        if not 'label' in plot_kwargs:
            plot_kwargs['label'] = self.bike.id

        xy = self.bike.traj[0:2, :self.i]
        p = ax.plot(xy[0,:], xy[1,:], **plot_kwargs), 

        return ax, p


class FixedInputZigZagTest(Scenario):
    """
    A specialized simulation scenario that simulates a ZigZag test for a
    fixed, given input command sequence.
    """

    def __init__(
        self,
        s0,
        p_x_c,
        p_y_c,
        v,
        target_locations,
        cyclist_name="cyclist",
        ax=None,
        bike_class=FixedSpeedBalancingRiderBicycle,
        seed = 42,
        animate = True,
        **kwargs
    ):
        """
        Create a test scenario

        Parameters
        ----------
        s0 : array-like
            Initial state of the test scenario bike given as (x0, y0, psi0,
            v0, delta0, phi0).
        p_x_c : array-like
            Trajectory of x-components of the commanded destinations.
        p_y_c : array-like
            Trajectory of y-components of the commanded destinations.
        v : loat or array-like
            If a single value, this is the fixed default desired speed of the
            bicycle. If an array-, this is the trajectory of desired speeds.
        target_locations : array-like
            Locations of the command targets. Shaped (n_targets, 2).
        cyclist_name : string, optional
            Name of the cyclist for labels in plots. The default is 'cyclist'.
        ax : TYPE, optional
            Axes for the simulation. If None, a new set of axes is created.
            The default is None.
        bike_class : cyclistsocialforce.vehicle.Vehicle, optional
            Bicycle class of the simulated bike. The default is
            WhippleCarvalloBicycle.
        bike_dynamics_class : cyclistsocialforce.dynamics.VehicleDynamics, opt.
            Dynamics class of the simulated bike. The default is
            FixedSpeedBalancingRiderDynamics.
        seed : int, optional
            Seed used for all random operations performed by this class. 
            Default is 42.
        **kwargs Any keyword arguments passed to the bicycle params class of
            the bicycle. 
        """
        
        self.seed = seed

        #increase the sample length of the command by one to be able to take derivatives later on
        self.p_x_c = np.r_[p_x_c, p_x_c[-1]]
        self.p_y_c = np.r_[p_y_c, p_y_c[-1]]

        self.target_locations = target_locations
            
        if animate:
            t_r = 0.01  #simulation speed
            verbose = True
            if ax is None:
                ax = self.create_axes()
        else:
            t_r = 0
            verbose=False
            
        Scenario.__init__(self, self.step_func, animate=animate, axes=ax, t_r=t_r, verbose=verbose)

        # make bike object
        if bike_class is FixedSpeedBalancingRiderBicycle:
            kwargs['v_desired_default'] = np.mean(v)
            params = bike_class.PARAMS_TYPE(**kwargs)

            self.bike = bike_class(s0, id=cyclist_name, saveForces=True, params=params)

        elif bike_class is FixedSpeedPlanarPointBicycle:
            params = bike_class.PARAMS_TYPE(**kwargs)
            self.bike = bike_class(s0, id=cyclist_name, saveForces=True, params=params)

        else:
            raise NotImplementedError(f"Bicycle type '{bike_class}' not implemented!")

        if isinstance(v, float):
            self.has_vd_trajectory = False
            self.bike.params.v_desired_default = v
        else:
            self.v = np.r_[v, v[-1]]
            self.has_vd_trajectory = True

        if animate:
            self.bike.add_drawing(ax, draw_force_destination=True)
        self.bike.dest_force_func = calc_direct_approach_dest_force

        self.i = 1

    def step_func(self):
        """
        A custom step function for this scenario class. Overwrites
        Scenario.step_func().

        Calculates the destination force for the current commanded
        destination and propagates the bicycle.
        """

        self.bike.setDestinations(
            self.p_x_c[self.i], self.p_y_c[self.i], reset=True
        )

        # set the desired speed to the current speed in the trajectoy
        if self.has_vd_trajectory:
            self.bike.params.v_desired_default = self.v[self.i]

        Fx, Fy = self.bike.calcDestinationForce()

        self.bike.step(Fx, Fy)

        self.bike.update_drawing(Fdest=(Fx, Fy))

    def run(self):
        """
        A custom run function for this scenario class. Overwrites
        Scenario.run().

        Automatically calculates the duration of this run from the length
        of the provided command sequence and the sampling time.
        """
        t_end = self.t_s * (len(self.p_x_c))
        Scenario.run(self, t_end)

        if self.animate:
            self.bike.drawing.set_animated(False)

    def create_axes(self):
        """
        Create an axes object for this simulation.

        Returns
        -------
        ax : axes
            The axes this simulation is animated in.

        """
        fig, ax = plt.subplots(1, 1)
        ax.set_xlim(-20, 30)
        ax.set_ylim(5, 20)
        ax.set_aspect("equal")

        ax.scatter(
            self.target_locations[:, 0],
            self.target_locations[:, 1],
            color="gray",
        )

        return ax
    
    def get_control_id_data(self, T = None, noise = (0,0,0,0,0,0,0)):
        """
        Return the data required for rider control id.

        Parameters
        ----------
        T : float, optional
            Duration of the sample in seconds. The default is the full time 
            the simulation was run. 
        noise : array_like, optional
            White gaussian noise added to the clean simulation data give as an
            array of variances:
                (command_time_delay_variance,
                 x_noise_variance
                 y_noise_variance
                 psi_noise_variance)
            The default is (0,0,0,0) which results in no added noise.

        Returns
        -------
        p_x_c_m : numpy.ndarray
            X component of the commanded destination. A non-zero noise 
            will introduce a time delay rather then an error to the 
            x-value.
        p_y_c_m : numpy.ndarray
            Y component of the commanded destination. A non-zero noise 
            will introduce a time delay rather then an error to the 
            y-value.
        p_x_m : numpy.ndarray
            X component of the simulated bike trajectory. A non-zero noise will
            distort the x value. 
        p_y_m : numpy.ndarray
            Y component of the simulated bike trajectory. A non-zero noise will
            distort the y value. 
        psi_m : numpy.ndarray
            Psi component of the simulated bike trajectory. A non-zero noise 
            will distort the psi value. 

        """
        
        if T is None:
            N = self.i
        else:
            N = int(T/self.t_s)
        
        rng = np.random.default_rng(self.seed)
        
        if noise[0] > 0:
            di_noise = int(abs(rng.normal(0, noise[0]))/self.t_s)
            
            p_x_c_m = np.ones(N) * self.p_x_c[0]
            p_y_c_m = np.ones(N) * self.p_y_c[0]
            
            p_x_c_m[di_noise:] = self.p_x_c[0:N-di_noise]
            p_y_c_m[di_noise:] = self.p_y_c[0:N-di_noise]
            
        else:
            p_x_c_m = self.p_x_c[0:N]
            p_y_c_m = self.p_y_c[0:N]
        
        p_x_m = rng.normal(self.bike.traj[0, 0:N], noise[1])
        p_y_m = rng.normal(self.bike.traj[1, 0:N], noise[2])
        psi_m = rng.normal(self.bike.traj[2, 0:N], noise[3])
        delta_m = rng.normal(self.bike.traj[4, 0:N], noise[4])
        phi_m = rng.normal(self.bike.traj[5, 0:N], noise[5])
        
        return p_x_c_m, p_y_c_m, p_x_m, p_y_m, psi_m, delta_m, phi_m


class ZigZagTest(FixedInputZigZagTest):
    """ Simulate runs across the experimental using asynthetic input signal.
    """
    
    TMAX = 20
    
    def __init__(self,
        command_frequency = 0.33,
        command_time_jitter = 0.5,
        evaluation_area_size = [28,10],
        v = 3,
        t_s = 0.01,
        seed = 42,
        **kwargs):
        
        self.command_frequency = command_frequency
        self.command_time_jitter = command_time_jitter
        self.evaluation_area_size = evaluation_area_size
        
        self.target_locations = np.array([[37, 2],[37, 4],[37,6],[37,8],[37,10]])
        
        self.t_s = t_s
        kwargs["t_s"] = t_s
        self.seed = seed
        kwargs["seed"] = seed
        self._make_command_signal()
        
        v = self._parse_speed(v)
        self.initial_state = np.array([0,self.target_locations[2,1],0,v[0],0,0], dtype=float)
        
        super().__init__(self.initial_state, self.p_x_c, self.p_y_c, v, self.target_locations,
                         **kwargs)
        
    def _parse_speed(self, v):
        
        if isinstance(v, (int, float)):
            v = v * np.ones_like(self.p_x_c, dtype=float)
        elif v.size != self.p_x_c.size:
            raise ValueError(f'v must be size {self.p_x_c.size}')
            
        return v
        
    def _make_command_signal(self):
        
        i_max = int(self.TMAX/self.t_s)+1
        
        rng = np.random.default_rng(self.seed)
        
        # draw command times and command light indices
        time_to_signal = rng.uniform(1/self.command_frequency - self.command_time_jitter,   
                                     1/self.command_frequency + self.command_time_jitter, (15,))
        steps_per_command = (time_to_signal/self.t_s).astype(int)
        
        command_light_idx = [rng.integers(0,4)]
        for i in range(1,15):
            idx = rng.integers(0,4)
            while idx == command_light_idx[i-1]:
                idx = rng.integers(0,4)
            command_light_idx.append(idx)
        
        #create data array
        commands = np.repeat(command_light_idx, steps_per_command)
        assert commands.size >= i_max, 'Not enough commands!'
        commands = commands[:i_max]
        
        self.p_x_c = self.target_locations[:,0][commands]
        self.p_y_c = self.target_locations[:,1][commands]
        
    def step_func(self):
        """Add interrupt if outside simulation area to the parent step function """
        super().step_func()
        
        outside = self.bike.s[0] > self.evaluation_area_size[0] or \
                  self.bike.s[1] < self.target_locations[2,1] - self.evaluation_area_size[1]/2 or \
                  self.bike.s[1] > self.target_locations[2,1] + self.evaluation_area_size[1]/2 
                  
        if outside:
            raise StopSimulationInterrupt(f"Bicycle at [{self.bike.s[0]:.2f}, {self.bike.s[1]:.2f}]"
                                          f" outside experiment area!")
        
    def run(self):
        """Add interrupt if outside simulation area to the parent run function"""
        try:
            super().run()
        except StopSimulationInterrupt as e:
            print(e)
        
    def create_axes(self):
        """
        Create an axes object for this simulation. Overwrites the parent method.

        Returns
        -------
        ax : axes
            The axes this simulation is animated in.

        """
        fig, ax = plt.subplots(1, 1)
        ax.set_xlim(-5, 40)
        ax.set_ylim(0, 12)
        ax.set_aspect("equal")

        ax.scatter(
            self.target_locations[:, 0],
            self.target_locations[:, 1],
            color="gray",
        )

        return ax


class FixedSpeedStepResponses():
    """ Simualte the response to a heading step command at fixed speed. 
    """

    BIKEMODELS = {
        'balancingrider': FixedSpeedBalancingRiderBicycle,
        'planarpoint': FixedSpeedPlanarPointBicycle
    }

    def __init__(self, poles, speeds, T=5, psi_c_deg=10, bikemodel='balancingrider'):

        # bike model
        self.bikemodel = self._check_bikemodel(bikemodel)

        # simulation time
        self.t = T_S * np.arange(int(T/T_S))

        # make target
        self.psi_c = np.deg2rad(psi_c_deg)
        target_dist = 100
        self.target_locs = np.array([[target_dist,0], [target_dist * np.cos(self.psi_c), target_dist* np.sin(self.psi_c)]])
        self.p_x_c = np.ones_like(self.t) * target_dist * np.cos(self.psi_c)
        self.p_y_c = np.ones_like(self.t) * target_dist * np.sin(self.psi_c)

        self.p_x_c[0] = target_dist * np.cos(0)
        self.p_y_c[0] = target_dist * np.sin(0)

        self.poles = poles
        self.speeds = speeds


    def get_ideal_xy_response(self, speed=4):

        x_ideal = self.t * speed * np.cos(self.psi_c)
        y_ideal = self.t * speed * np.sin(self.psi_c)
        psi_ideal = np.ones_like(self.t) * self.psi_c

        return x_ideal, y_ideal, psi_ideal


    def _check_bikemodel(self, bikemodel):

        if bikemodel not in self.BIKEMODELS:
            raise ValueError(f"'bikemodel' must be one of {self.BIKEMODELS}. Instead it was '{bikemodel}'.")
        
        return bikemodel


    def simulate(self, animate=False, verbose=False, plot=True):
        """Simulate the trajectories of a fixed-speed command step response"""

        trajs = []
        n_samples = self.poles.shape[0]

        print(f"Simulate {n_samples} step responses for step height psi_c = {np.rad2deg(self.psi_c)} deg:")

        for i in range(n_samples):

            poles_i = self.poles[i,:]

            if np.any(np.real(poles_i)>0):
                print(f"Skipping ustable prediction {i}")
                continue

            print(f"Simulating pole {i}/{n_samples}")

            if self.bikemodel == 'balancingrider':
                s0 = np.array([0, 0, 0, self.speeds[i], 0, 0, 0, 0])
            else:
                s0 = np.array([0, 0, 0, self.speeds[i]])

            test_sim = FixedInputZigZagTest(
                s0, 
                self.p_x_c, 
                self.p_y_c, 
                s0[3] * np.ones_like(self.p_x_c), 
                self.target_locs, 
                cyclist_name=str(i), 
                bike_class=self.BIKEMODELS[self.bikemodel],
                poles=poles_i,
                animate=animate, 
                verbose=verbose)
            test_sim.run()

            trajs.append(test_sim.bike.traj[:,:int(test_sim.t/T_S)])

        return trajs

class StableGainFinder:
    """ Search for stable gain combinations and randomly sample.
    """
    
    SEARCH_LIMITS = (-401, 400)
    SEARCH_STEP = 200
    
    def __init__(self, bike, 
                 vminmax=None, 
                 search_limits=None, 
                 search_step=None):

        self.search_limits = np.array(none_switch(search_limits, 
                                              StableGainFinder.SEARCH_LIMITS))
        
        if self.search_limits.size == 2:
           self.search_limits = np.tile(np.reshape(self.search_limits, (2,1)), 5).T
    
        self.search_step = none_switch(search_step,
                                            StableGainFinder.SEARCH_STEP)
        
        self.bike = bike
        
        if vminmax is None:
            self.minmax=False
            v = bike.s[3]
        
        
            A, B, C, D = bike.dynamics.get_statespace_matrices(v)
            B = np.reshape(B[:,1], (5,1))
            D = np.reshape(D[:,1], (1,1))
            
            self.A = A
            self.B = B
            self.C = C
            self.D = D
            
        else:
            self.minmax=True
            
            A, B, C, D = bike.dynamics.get_statespace_matrices(min(vminmax))
            B = np.reshape(B[:,1], (5,1))
            D = np.reshape(D[:,1], (1,1))
            
            self.Amin = A
            self.Bmin = B
            self.Cmin = C
            self.Dmin = D
            
            A, B, C, D = bike.dynamics.get_statespace_matrices(max(vminmax))
            B = np.reshape(B[:,1], (5,1))
            D = np.reshape(D[:,1], (1,1))
            
            self.Amax = A
            self.Bmax = B
            self.Cmax = C
            self.Dmax = D
            
    def test_stability(self, gains_test):
        if self.minmax:
            sys, gains = from_gains(self.Amin, self.Bmin, self.Cmin, self.Dmin, gains_test[np.newaxis,:])
            stbl_min, pls = test_stability(sys, stability_type='marginal')
            
            sys, gains = from_gains(self.Amax, self.Bmax, self.Cmax, self.Dmax, gains_test[np.newaxis,:])
            stbl_max, pls = test_stability(sys, stability_type='marginal')
            
            stbl = stbl_min & stbl_max
            
        else:         
            sys, gains = from_gains(self.A, self.B, self.C, self.D, gains_test[np.newaxis,:])
            stbl, pls = test_stability(sys, stability_type='marginal')
                
        return stbl, pls
        

    def find_stable_gains(self):
        
        gain_limits = self.search_limits
        
        n_gains = gain_limits.shape[0]
        n_samples = n_gains * 8
        total_range = np.sum(np.abs(np.diff(gain_limits)))
        step = total_range // n_samples
        
        has_sign_change = np.logical_and(gain_limits[:,0]<0, gain_limits[:,1]>0)
        
        n_samples_per_gain = np.floor(((np.diff(gain_limits)/total_range)*n_samples)).astype(int)
        n_samples_per_gain = n_samples_per_gain.flatten()
        n_samples_per_gain[n_samples_per_gain < 2 ] = 2
        
        gains_range = []
        for i in range(n_gains):
            
            if gain_limits[i,0] < 0 and gain_limits[i,1] > 0:
                
                n_samples_pos = int(n_samples_per_gain[i] * np.abs((gain_limits[i,1]/(gain_limits[i,1]-gain_limits[i,0]))))
                n_samples_neg = int(n_samples_per_gain[i] * np.abs((gain_limits[i,0]/(gain_limits[i,1]-gain_limits[i,0]))))
                
                n_samples_pos = max(n_samples_pos, 2)
                n_samples_neg = max(n_samples_neg, 2)
                
                rng = list(np.linspace(min(gain_limits[i,0], -0.1), -0.1, n_samples_neg)) + \
                      [0.0] + \
                      list(np.linspace(0.1, max(gain_limits[i,1], 0.1), n_samples_pos))
            elif gain_limits[i,0] == gain_limits[i,1]:
                rng = [gain_limits[i,0]]
            else:
                if gain_limits[i,1] <= 0:
                    rng = list(np.linspace(gain_limits[i,0], min(-0.1, gain_limits[i,1]), n_samples_per_gain[i]))
                else:
                    rng = list(np.linspace(max(0.1, gain_limits[i,0]), gain_limits[i,1], n_samples_per_gain[i]))
                if np.any(gain_limits[i,1]==0):
                    rng += [0.0]
                
            gains_range.append(rng)
        
        
        N = np.prod([len(grange) for grange in gains_range])    
        gains_stable = np.zeros((N,n_gains))
    
        gains_test = np.meshgrid(*gains_range)
        gains_test = np.array([gtest.flatten() for gtest in gains_test]).T
        
        i_stable = 0
        for i in range(N):
            
            stbl, pls = self.test_stability(gains_test[i,:])
            
            if stbl:
                gains_stable[i_stable,:] = gains_test[i,:]
                i_stable+=1
                
        gains_stable = gains_stable[:i_stable,:]
        
        limits_stable = np.c_[np.min(gains_stable,0), np.max(gains_stable,0)]
         
        return gains_stable, limits_stable

    
    def find_stable_gains_alternative(self, limit, n_samples_per_gain):
        
        #### NOT YET FULLY IMPLEMENTED. DOES NOT WORK
        
        glim = np.tile(np.array([[-401], [400]]), 5).T
        
        power = 2
        gains_test = np.array(([0]))
        n_samples=len(gains_test)
        j = 0
        
        step = 10
        while n_samples < n_samples_per_gain:
            i = 2
            gains_test = np.array(([0]))
            while gains_test[-1] < limit:
                gains_test = np.cumsum(np.arange(start=0.1**(1/power),stop=i*step+0.1, step=step))**power
                i+=1
            step = step/2
            n_samples = len(gains_test)
        
            j +=1
            if j == 3000:
                break
        
    
    def test_zero_limit(self,):
        
        zero_limit = (-1,1)
        step = 0.2
        test_range = np.arange(zero_limit[0], zero_limit[1]+step, step)
        
        limits = []
        for i in range(self.stable_gains.shape[1]):
            limits.append((min(self.stable_gains[:,i]), max(self.stable_gains[:,i])))
        limits = np.array(limits)
        
        stable_zero = []
        for i in range(self.stable_gains.shape[1]):
            for g in test_range:
                gain = np.array(self.stable_gains)
                gain[:,i] = g
                
                for j in range(gain.shape[0]):
                    stbl, pls = self.test_stability(gain[j,:])
                    
                if stbl:
                    limits[i,: ] = [max(limits[i,0], g), min(limits[i,1], g)]
                        
        return limits
    
    def sample_extremes(self, initial_guess, n_extremes = 2):
        
        rng = np.random.default_rng() 
        n_gains = initial_guess.shape[1]
        extremes = np.zeros((n_gains*2*n_extremes, n_gains))
        
        for i in range(n_gains):
            
            candidates = np.where(initial_guess[:,i] == np.min(initial_guess[:,i]))
            random_extreme = rng.choice(candidates[0], size=n_extremes)
            extremes[(i*n_extremes):(i*n_extremes)+n_extremes,:] = initial_guess[random_extreme,:]
            
            candidates = np.where(initial_guess[:,i] == np.max(initial_guess[:,i]))
            random_extreme = rng.choice(candidates[0], size=n_extremes)
            extremes[(i*n_extremes)+(n_extremes*n_gains):(i*n_extremes)+n_extremes+(n_extremes*n_gains)] = initial_guess[random_extreme,:]
            
        extremes = np.unique(extremes, axis=0)
        
        rng = np.random.default_rng()
        rng.shuffle(extremes, axis=0)
        
        return extremes

    def sample(self, initial_guess, n = 30, force_extremes=True, n_extremes=1):
        
        #sample extremes
        if force_extremes:
            extremes = self.sample_extremes(initial_guess, n_extremes=n_extremes)
            extremes = extremes[:min(n,extremes.shape[0]),:]
            n_extremes = extremes.shape[0]
        else:
            n_extremes = 0
            
        
        #sample
        rng = np.random.default_rng()
        rows = np.arange(0, initial_guess.shape[0])
        rows_sampled = rng.choice(rows, size=n-n_extremes)
        
        #join and find unique
        if force_extremes:
            sampled = np.r_[extremes, initial_guess[rows_sampled, :]]
        else:
            sampled = initial_guess[rows_sampled, :]
        sampled = np.unique(sampled, axis=0)
        
        #fill missig gains if sample had duplicates
        attempts = 0
        while sampled.shape[0] < n:
            n_more = n-sampled.shape[0]
            rows_sampled = rng.choice(rows, size=n_more)
            
            sampled = np.r_[sampled, initial_guess[rows_sampled, :]]
            sampled = np.unique(sampled, axis=0)
            
            #remove all zero gains
            nonzero = np.any(sampled, axis=1)
            sampled = sampled[nonzero,:]
            
            attempts += 1
            if attempts > 30:
                break
        return sampled


def create_stable_gain_sample(bike, n=30, vminmax=None, limits=None, step=None):
    
    if isinstance(bike.dynamics, FixedSpeedBalancingRiderDynamics):
        gain_finder = StableGainFinder(bike, vminmax=vminmax, 
                                       search_limits=limits, search_step=step)
        
        stable_gains, limits = gain_finder.find_stable_gains()
        sampled_stable_gains = gain_finder.sample(stable_gains, n=n)
        
        rng = np.random.default_rng()
        rng.shuffle(sampled_stable_gains, axis=0)
        
    elif isinstance(bike.dynamics, FixedSpeedPlanarPointDynamics):
        limits = none_switch(limits, StableGainFinder.SEARCH_LIMITS)
        limits = np.maximum(limits, 0)
        limits = np.reshape(limits, (1,2))
        
        gains_highmag = np.linspace(limits[0,0], limits[0,1], int(n*0.7))
        gains_lowmag = np.linspace(0, gains_highmag[2], n - int(n*0.7)+3)
        
        sampled_stable_gains = np.r_[gains_lowmag[1:-1], gains_highmag[1:]]
        sampled_stable_gains = np.reshape(sampled_stable_gains, (n, 1))
    else:
        raise ValueError(f'Bike type {type(bike)} not supported!')
    
    return sampled_stable_gains, limits

def simulated_input(samples, f, y_vals=None, t_s=0.01):
    """
    Simulate an zigzagtest input signal. Random sequence of jumps between the discrete y_coordinates of the 
    light positions. The jump times are uniformly distributed in f +/- 500 ms. Consecutive jumps
    must have different values. 
    """
    
    rng = np.random.default_rng()
    
    T = ((np.array([- 0.5, 0.5]) + 1/f) / t_s).astype(int)
    jump_times = rng.integers(T[0], T[1], samples)
    jump_times = np.cumsum(jump_times)
    jump_times = jump_times[jump_times < samples]
    jump_times = np.r_[[0], jump_times, samples]
    
    val = 0
    simulated_input = np.zeros(samples)
    for i in range(1, jump_times.size):
        val = rng.choice(y_vals[np.logical_not(y_vals==val)])
        simulated_input[jump_times[i-1]:jump_times[i]] = val
        
    simulated_input -= np.mean(simulated_input)
        
    return simulated_input