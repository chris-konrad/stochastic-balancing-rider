# -*- coding: utf-8 -*-
"""
Process the data of the zigzag experiment. 

@author: Christoph M. Konrad

"""
import warnings
import re
import copy
import os
import yaml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
import sympy as sm
#import asammdf
import sympy.physics.mechanics as me
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
from scipy.signal import correlate
from trajdatamanager.datamanager import (
    DataManager,
    RTKLibGNSSManager,
    Track,
    Sequence,
)

from cyclistsocialforce.utils import cart2polar
from cyclistsocialforce.vehicle import BalancingRiderBicycle, PlanarPointBicycle

from mypyutils.io import fileparts

from rcid.path_manager import PathManager
from rcid.ukf import filter_dynamic
from rcid.utils import read_yaml

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

class UnexpectedStringArgError(ValueError):
    """ An error raise if a String argument does not have the expected format.
    """
    def __init__(self, arg_name, arg, desired_arg_format):
        """ Raise an UnexpectedStringArgError
        
        Parameters
        ----------
        arg_name : str
            The name of the wrong argument/variable/parameter.
        arg : str
            The received value of the argument/variable/parameter.
        desired_arg_format : str
            A str describing the desired format of the arg/var/param.
        """
        
        self.arg_name = arg_name
        self.arg = arg
        self.desired_arg_format = desired_arg_format
    
    def __str__(self):
        msg = (f"The given {self.arg_name} '{self.arg}' does not match the " 
               f"expected format {self.desired_arg_format}.")
        return repr(msg)


class BalanceAssistLogDataManager(DataManager):
    """
    Manage the CAN bus log data from the Balance Assist Bikes.
    """

    NUM_SKIP_ROWS = 6  # Number of rows to skip at the bginning of a .txt file
    GPS_LEAP_SECONDS = 18  # GPS leap seconds for GPS time to UTC conversion
    G = 9.81  # gravity
    STEER_ANGLE_BIAS = 32 - 12.5

    def __init__(self, path_can_log, path_gnss_report, bike_geometry, dbc_file=None):
        super().__init__(path_can_log)
        self.dir_gnss_report = path_gnss_report
        self.bike_geom = bike_geometry

        self.dbc_file = dbc_file

    # def load_sequence(self):
    #    for path, folders, files in os.walk(self.dir):
    #        test = 100

    def load_tracks(self, filename_can, filename_bike_gnss):

        # extract CAN data
        if filename_can.endswith('.mf4'):
            if self.dbc_file is None:
                raise ValueError("Supply .dbc file to decode .mf4 CAN logs.")

            df = decode_CANedge(
                [os.path.join(self.dir, filename_can)],
                {"LIN": [(self.dbc_file, 0)], "CAN": [(self.dbc_file, 0)]},
            )
        elif filename_can.endswith('.parquet'):
            df = decode_parquet(os.path.join(self.dir, filename_can))

        t_can = np.array(df.index)
        t_can = np.array([(ti - t_can[0]).total_seconds() for ti in t_can])

        a_can = (
            np.sqrt(
                df["accel_x"] ** 2 + df["accel_y"] ** 2 + df["accel_z"] ** 2
            )
            - self.G
        )
        a_can = np.array(a_can)
        
        # roll and yaw and linear acceleration
        yaw, dyaw, roll, droll, accel = self.bike_geom.transform_imu2bike(
            df["yaw"],
            df["roll"],
            df["gyro_x"],
            df["gyro_z"],
            df["accel_x"],
            )

        
        # speed
        #   Derived from the rear wheelspeed assuming a circumfence of 221cm
        #   With rider (approx 70kg) and 3.0 bar tirepressure, the effective
        #   circumfence reduces to 219.25cm
        speed = np.array(df["ws_rear"]) * (219.25/221)
        
        # steer angle
        #   The steer assist biccle assumes a wrong sensor bias. This is 
        #   corrected here. Additionally direction definitions varies.
        #   Definition on the balance-assist bikes: right+, left-
        #   Definition in cyclistsocialforces: right-, left+
        steer = -(
            np.pi * (np.array(df["LWS_ANGLE"]) - self.STEER_ANGLE_BIAS) / 180
        )
        dsteer = -(np.pi * np.array(df["LWS_SPEED"]) / 180)

        # extract GNSS IMU data for time synchronisation of the CAN data
        t_gnss_global, a_gnss, path_timesync_source = self.load_gnss_imu(
            filename_bike_gnss
        )
        t_gnss_global_begin = t_gnss_global[0]
        t_gnss_local = np.array(
            [(ti - t_gnss_global[0]).total_seconds() for ti in t_gnss_global]
        )

        #plot for validation
        #fig0, ax0 = plt.subplots(2,1, sharex=True)
        #ax0[0].plot(t_gnss_local, a_gnss)
        #ax0[1].plot(t_can, a_can)

        # identify time offset between local times of CAN and GNSS data
        t_ss = 0.001
        t_gnss_100 = np.arange(0, t_gnss_local[-1], t_ss)
        t_can_100 = np.arange(0, t_can[-1], t_ss)

        a_can_interp = np.interp(t_can_100, t_can, a_can)
        a_gnss_interp = np.interp(t_gnss_100, t_gnss_local, a_gnss)


        a_corr = correlate(a_can_interp, a_gnss_interp, mode="full")
        n_offset = (np.argmax(a_corr) - len(a_gnss_interp) + 1)
        t_offset = n_offset * t_ss
            #t_offset = -np.argmax(a_corr) * 0.005
        print(f"Time offset: {t_offset} s ... ", end="")

        #plot for validation
        #fig, ax = plt.subplots(1,1)
        #if n_offset < 0: 
        #    ax.plot(a_gnss_interp[abs(n_offset):])
        #    ax.plot(a_can_interp)
        #else:
        #    ax.plot(a_gnss_interp)
        #    ax.plot(a_can_interp[abs(n_offset):])
        
        #identify drift
        if n_offset < 0: 
            get_drift = find_drift(a_gnss_interp[abs(n_offset):], a_can_interp, t_ss)
            drift = get_drift(t_can[:,np.newaxis])
        else:
            get_drift = find_drift(a_gnss_interp, a_can_interp[abs(n_offset):], t_ss)
            drift = get_drift(t_can[:,np.newaxis]-t_offset)
            
        # plot for validation
        # fig2, ax2 = plt.subplots(1,1)
        # ax2.plot(a_corr)

        # derive timstamps for CAN data from GNSS time, offset and drift
        t_can_global = t_gnss_global_begin + dt.timedelta(seconds=1) * (t_can - (t_offset - drift)) 

        # plot for validation
        fig3, ax3 = plt.subplots(1, 1, layout='constrained')
        ax3.set_xlabel("time")
        ax3.set_ylabel("total acceleration [m/s^2]")
        ax3.set_ylim(-20,20)
        ax3.plot(t_gnss_global, a_gnss)
        ax3.plot(t_can_global, a_can)

        metadata = {
            "track_type": "BalanceAssistLogData",
            "source": os.path.join(self.dir, filename_can),
            "source_timesync": path_timesync_source,
            "dbc_file": self.dbc_file,
            "time_offset": t_offset,
        }

        # create a track with the CAN data
        trk = Track(
            filename_can,
            2,
            t_can_global,
            np.c_[steer, dsteer, roll, droll, yaw, dyaw, speed, accel],
            data_feature_keys=["delta", "ddelta", "phi", "dphi", "psi", "dpsi", "v", "a"],
            metadata=metadata,
        )

        return trk

    def load_gnss_imu(self, session_name):

        if not isinstance(session_name, list):
            session_name = [session_name]

        t_out = np.array([])
        a_out = np.array([])
        gyro_z_out = np.array([])

        for rname in session_name:
            if rname[-4] == ".":
                rname = rname[:-4]

            path_timesync_source = os.path.join(
                self.dir_gnss_report, rname, rname + "-ins.csv"
            )

            df = pd.read_table(
                path_timesync_source,
                sep=",",
            )

            gps_epoch = dt.datetime(1980, 1, 6, tzinfo=dt.UTC)
            gps_weeks = df["GPS Week"]
            gps_tow = df["GPS TOW [s]"]

            t = []
            valid_times = np.ones(len(gps_tow), dtype=bool)
            i = -1
            for si, wi in zip(gps_tow, gps_weeks):
                i += 1
                if (np.isnan(si) or np.isnan(wi)) or (si == 0 or wi == 0):
                    valid_times[i] = False
                    continue
                t.append(
                    gps_epoch
                    + dt.timedelta(
                        weeks=wi, seconds=si - self.GPS_LEAP_SECONDS
                    )
                )

            t = np.array(t)

            a = np.sqrt(
                df["Acc X [g]"][valid_times] ** 2
                + df["Acc Y [g]"][valid_times] ** 2
                + df["Acc Z [g]"][valid_times] ** 2
            )
            a = (a * self.G) - self.G
            a = np.array(a)
            
            gyro_z = df['Gyr Z [deg/s]']

            valid_accels = np.logical_not(np.isnan(a))
            a = a[valid_accels]
            t = t[valid_accels]

            t_out = np.r_[t_out, t]
            a_out = np.r_[a_out, a]

        return t_out, a_out, path_timesync_source
        

class CommandLightDatamanager(DataManager):
    """
    Manage command light data created from the light controller
    Arduino.
    """

    NUM_SKIP_ROWS = 6  # Number of rows to skip at the bginning of a .txt file
    T_S = 0.01

    def __init__(
        self,
        path_name,
    ):
        super().__init__(path_name)

    def preview_times(self, filename):

        df = pd.read_table(
            os.path.join(self.dir, filename),
            sep=";",
            parse_dates={"Timestamp": [0]},
            date_format="%Y%m%d_%H%M%S%f",
            skiprows=self.NUM_SKIP_ROWS,
            header=0,
            names=["", "light_command"],
        )

        t = df["Timestamp"]
        t_begin = t.iloc[0].to_pydatetime().replace(tzinfo=dt.UTC)
        t_end = t.iloc[-1].to_pydatetime().replace(tzinfo=dt.UTC)

        return t_begin, t_end

    def load_tracks(self, filename):

        df = pd.read_table(
            os.path.join(self.dir, filename),
            sep=";",
            parse_dates={"Timestamp": [0]},
            date_format="%Y%m%d_%H%M%S%f",
            skiprows=self.NUM_SKIP_ROWS,
            header=0,
            names=["", "light_command"],
        )

        t = df["Timestamp"]
        t = np.array([ti.to_pydatetime().replace(tzinfo=dt.UTC) for ti in t])

        data = np.array(df["light_command"])[:, np.newaxis]

        # interpolate
        t_full = [t[0]]
        while t_full[-1] < t[-1]:
            t_full.append(t_full[-1] + dt.timedelta(seconds=self.T_S))
        t_full = np.array(t_full[:-1])

        data_full = np.zeros((len(t_full), 1))
        data_full[0, 0] = data[0]
        for i in range(1, len(t_full)):
            j = max(np.argwhere(t < t_full[i]))
            data_full[i, 0] = data[j]

        keys = ["n_c"]

        metadata = {
            "track_type": "CommandLightData",
            "source": os.path.join(self.dir, filename),
        }

        trk = Track(
            filename,
            2,
            t_full,
            data_full,
            yaw_feature_index=None,
            data_feature_keys=keys,
            metadata=metadata,
        )
        return trk


class RawDataProcessor:
    """ A class to process raw data of a session of the zigzag experiment. 

    A session a consecutive set of measurements and typically equals one block
    of one participant.
    """

    TZ = "UTC"

    def __init__(
        self,
        dir_data,
        experiment_name,
        experiment_date,
        participant_id,
        session_name,
        filename_can,
        filename_lights_gnss="0001-00000.pos",
        subdir_target_calibration=None,
        subdir_bike_gnss_solution=None,
        subdir_bike_gnss_report=None,
        subdir_cmds=None,
        subdir_bike_can=None,
        filename_dbc=None,
        rotation=66.0,
        reference_location=[51.999370, 4.370451],
        flip_target_locs=False,
        get_return_trajectories = False,
        filter_settings = None,
        save_reports=True,
        plot=True,
    ):
        """ Create a ZigZagExperimentData object to process the data 
        of one session.

        Required file system.

        raw
        |---experiment_name
            |---light-gnss
            |   |---solution
            |---bike-gnss
            |   |---solution
            |   |---report
            |---ligh-controller
            |   |---experiment_date
            |       |---participant_id
            |           |---session_name.txt
            |---can-logger
                |---filename_can
        """
        self.plot = plot
        self.save_reports = save_reports

        # set up paths and directories
        self.paths = PathManager(dir_data)
        self.paths.getdir_data_processed(new=True)
        self.paths.getdir_data_processed_reports(new=True)

        if subdir_target_calibration is None:
            subdir_target_calibration = os.path.join(
                experiment_name, "light-gnss", "solution"
            )

        if subdir_bike_gnss_solution is None:
            subdir_bike_gnss_solution = os.path.join(
                experiment_name, "bike-gnss", "solution"
            )

        if subdir_bike_gnss_report is None:
            subdir_bike_gnss_report = os.path.join(
                experiment_name, "bike-gnss", "report"
            )

        if subdir_cmds is None:
            subdir_cmds = os.path.join(
                experiment_name,
                "light-controller",
                experiment_date,
                participant_id,
            )

        if subdir_bike_can is None:
            subdir_bike_can = os.path.join(experiment_name, "can-logger")

        self.dir_base = self.paths.getdir_data_raw()
        self.experiment_name = experiment_name
        self.participant_id = participant_id
        self.session_name = session_name
        self.filename_can = filename_can
        self.subdir_target_calibration = subdir_target_calibration
        self.subdir_bike_gnss_solution = subdir_bike_gnss_solution
        self.subdir_bike_gnss_report = subdir_bike_gnss_report
        self.subdir_cmds = subdir_cmds
        self.subdir_bike_can = subdir_bike_can
        self.filename_dbc = filename_dbc
        self.filename_can = filename_can
        self.filename_lights_gnss = filename_lights_gnss

        # trial date
        self.experiment_date = dt.datetime.strptime(
            experiment_date + self.TZ, "%Y%m%d%Z"
        )

        # rotation and reference location
        self.rotation = rotation
        self.reference_location = reference_location
        
        # bicycle geometry and parameters
        sensor_geom_params = read_yaml(self.paths.getfilepath_data_sensorgeometry())
        self.bike_geom = InstrumentedBikeGeometry(sensor_geom_params)

        filepath_parset = self.paths.getfilepath_data_bicycleparset()
        with open(filepath_parset, 'r') as f:
            self.bicycleParameterDict = yaml.safe_load(f)['values']
        
        # flipping of target locations for inverse light positioning
        self.flip_target_locs = flip_target_locs

        # extract return trajctories
        self.get_return_trajectories = get_return_trajectories

        # run times
        self.t_runs = []

        # data
        self.bike_gnss_data = None
        self.bike_dynamic_data = None
        self.rider_cmd_data = None
        self.target_locations = None
    
        # measurement filter
        self.filter_settings = filter_settings

        #feature map
        self.feature_map = get_feature_map()

        
    def _filter(self, data_dict, 
                measurement_noise_scale=None, 
                process_noise_scale=None,
                integration_method=None):
        """Run a UKF on the given data.
        """
        
        if measurement_noise_scale is None:
            measurement_noise_scale = self.filter_settings['measurement_noise_scale']
        
        if process_noise_scale is None:
            process_noise_scale = self.filter_settings['process_noise_scale']
            
        if integration_method is None:
            integration_method = self.filter_settings['integration_method']
        
        keys = ["p_x", "p_y", "psi", "v", "phi", "delta", "dpsi",
                "dphi", "ddelta", "a"]
        
        #the measurements are in the E frame. The bike model of the kalman
        #filter operates in the N frame -> transform
        data_dict = self.bike_geom.transform_E2N(data_dict)
        
        measurements = np.zeros((len(data_dict["t"]), len(keys)))
        for i, k in enumerate(keys):
            measurements[:,i] = data_dict[k]
        
        R = np.diag(measurement_noise_scale**2)
        Q = np.diag(process_noise_scale**2) 
        
        filtered_measurements = filter_dynamic(measurements, R, Q, 
                                               integration_method=integration_method,
                                               bicycleParameterDict=self.bicycleParameterDict)
        
        data_dict_filtered = {k:v for k,v in data_dict.items() if k not in keys}
        for i, k in enumerate(keys):
            data_dict_filtered[k] = filtered_measurements[:,i]
          
        #the smoothed results are in the N frame. The measurements
        #are assumed to be in the E frame -> transform
        data_dict_filtered = self.bike_geom.transform_N2E(data_dict_filtered)
        data_dict = self.bike_geom.transform_N2E(data_dict)
            
        data_dict_filtered['psi_c'] = estimate_yaw_command(data_dict_filtered['p_x'], 
                                                           data_dict_filtered['p_y'],
                                                           data_dict_filtered['p_x_c'],
                                                           data_dict_filtered['p_y_c'])
    
        return data_dict_filtered


    def _load_bikedynamics(self):
        """ Load the bicycle kinematics measured by the IMU in the bicycle on-board computer
        """

        # make paths and directorys
        dir_can_log = os.path.join(self.dir_base, self.subdir_bike_can)
        dir_bike_gnss_report = os.path.join(
            self.dir_base, self.subdir_bike_gnss_report
        )

        if self.filename_dbc is not None:
            path_dbc = os.path.join(dir_can_log, self.filename_dbc)
        else:
            path_dbc=None

        # create datamanager
        dataman = BalanceAssistLogDataManager(
            dir_can_log, dir_bike_gnss_report, self.bike_geom, dbc_file=path_dbc
        )

        # load parquets instead of mf4s
        filename_can = list(fileparts(self.filename_can))
        filename_can[1] += ".parquet"
        filename_can = os.path.join(*filename_can[:2])
        

        # load the full track and segment into runs
        trk = dataman.load_tracks(filename_can, self.filenames_bike_gnss)
        self.bike_dynamic_data = self._segment_runs(trk, "bikedyn")

        # sample data at common time frame
        for trk_dyn, trk_cmd in zip(self.bike_dynamic_data, self.command_data):
            trk_dyn.sample_at_times(np.array(trk_cmd.t))
            
        # correct yaw angle offset
        for trk_dyn, trk_gnss in zip(self.bike_dynamic_data, self.bike_gnss_data):
            id_dyn_psi = trk_dyn.data_feature_keys.index('psi')
            
            psi_gnss = expand_timebase(trk_gnss['psi'][:,np.newaxis], 
                                       trk_gnss.t, trk_dyn.t)
            
            offset = np.nanmean(trk_dyn['psi']-psi_gnss)
            
            trk_dyn.data[:,id_dyn_psi] -= offset

        return self.bike_dynamic_data

    def _load_times(self):
        """
        Get the begin and end time of this experiment.

        Returns
        -------
        t_begin : datetime
            Begin time.
        t_end : datetime
            End time.

        """
        dir_cmds = os.path.join(self.dir_base, self.subdir_cmds)

        dataman = CommandLightDatamanager(dir_cmds)

        return dataman.preview_times(self.session_name + ".txt")

    def _load_cmds(self):
        """
        Load the command light data and derive the commanded yaw angle for
        every run.

        Requires to run _load_runs_gnss() first

        Parameters
        ----------
        path_cmd_data : string
            Path to the diectory with the command data files.
        run_filename : string
            File name of the run to be analyzed.

        Returns
        -------
        Sequence
            Sequence of tracks of the commanded yaw angle and the command
            lights

        """

        assert self.bike_gnss_data is not None, "First load bike GNSS data!"
        assert (
            self.target_locations is not None
        ), "First load target locations!"

        dir_cmds = os.path.join(self.dir_base, self.subdir_cmds)

        dataman = CommandLightDatamanager(dir_cmds)

        # load full track and segment into runs
        trk = dataman.load_tracks(self.session_name + ".txt")
        self.command_data = self._segment_runs(trk, "cmd")

        # derive commanded yaw angle and add to gnss data.
        for trk_gnss, trk_cmd in zip(self.bike_gnss_data, self.command_data):
            psi_c = np.zeros([trk_gnss.data.shape[0], 1])

            trk_cmd_sampled = copy.copy(trk_cmd)
            trk_cmd_sampled.sample_at_times(trk_gnss.t)

            # interpolation is based on cubic splines but led numbers have to
            # be integer indices.
            trk_cmd_sampled.data = np.floor(trk_cmd_sampled.data).astype(int)

            if len(trk_gnss.t) != len(trk_cmd_sampled.t):
                raise Exception(
                    (
                        f"Run {trk_gnss.track_id}: GNSS data and "
                        f"command data do not fully overlap"
                    )
                )

            def lightnum2targetloc(trk):
                x_cmd = np.zeros_like(trk.data, dtype=float)
                y_cmd = np.zeros_like(trk.data, dtype=float)

                for i in range(5):
                    x_cmd[trk.data == i] = self.target_locations[4 - i, 0]
                    y_cmd[trk.data == i] = self.target_locations[4 - i, 1]

                return x_cmd, y_cmd

            x_cmd, y_cmd = lightnum2targetloc(trk_cmd)
            x_cmd_sampled, y_cmd_sampled = lightnum2targetloc(trk_cmd_sampled)
            
            psi_c = estimate_yaw_command(trk_gnss["x"], trk_gnss["y"], 
                                         x_cmd_sampled, y_cmd_sampled)

            trk_cmd.add_features(np.c_[x_cmd, y_cmd], ["p_x_c", "p_y_c"])
            trk_gnss.add_features(psi_c[:, np.newaxis], ["psi_c"])

        # resample command data to match the exact gnss timestamps
        for i in range(0, len(self.bike_gnss_data.tracks)):
            t_sample = [self.bike_gnss_data[i].t_begin]
            while t_sample[-1] <= self.bike_gnss_data[i].t_end:
                t_sample.append(
                    t_sample[-1]
                    + dt.timedelta(seconds=self.command_data[i].t_s)
                )
            t_sample = np.array(t_sample[:-1])
            
            assert np.all([ti in t_sample for ti in self.bike_gnss_data[i].t])
            
            self.command_data[i].sample_at_times(t_sample)

        return self.command_data

    def _load_runs_gnss(self, x_limits=None, plot_results=True):
        """
        Load the gnss data of individual runs of a zigzag trial.

        Parameters
        ----------
        path : String
            Path to the gnss data (The directory of the .pos files of the RTK
            solution).
        run_filename : String
            Name of the .pos file of the run to be analyzed.
        x_limits : list, optional
            Minimum and maximum x value to segment the full track into individual
            runs. The default is [-18, 14].

        Returns
        -------
        trajdatamanager.datamanager.Sequence
            Sequence of tracks representing each individual run.

        """
        
        #Derive limits of the experiment area from measurements relative to the
        #target lights.
        if x_limits is None:
            x_limits = np.mean(self.target_locations[:,0]) - np.array([28+8.7, 8.7])

        if plot_results:
            fig, ax = plt.subplots(1, 1, layout='constrained')
            ax.grid()
            ax.set_xlabel('x [m]')
            ax.set_ylabel('x [m]')
            ax.set_title(f"Full GNSS data and extracted runs (before filtering): {self.experiment_name}, {self.participant_id}, {self.session_name}")
            plt.plot([x_limits[0], x_limits[0]], [-50, 50], color="red")
            plt.plot([x_limits[1], x_limits[1]], [-50, 50], color="red")

        # built paths and directories
        dir_bike_gnss_sol = os.path.join(
            self.dir_base, self.subdir_bike_gnss_solution
        )

        # create a datamanager instance
        dataman = RTKLibGNSSManager(
            dir_bike_gnss_sol, reference_location=self.reference_location
        )

        # load the track
        seq = dataman.load_sequence()
        seq = seq.reduce_to_timespan(
            self.t_begin, self.t_end, criterion="overlap"
        )
        self.filenames_bike_gnss = [trk.track_id for trk in seq.tracks]
        trk = seq.serialize()
        trk.crop_to_timespan(self.t_begin, self.t_end)

        # rotate the track for convenience and plot again
        trk.rotate_xy(self.rotation, deg=True)
        if plot_results:
            trk.plot_xy(ax=ax, color="gray")

        self.trk_gnss = trk
        # segment the track
        seqs = trk.segment_by_geofencing(x_limits)

        if self.get_return_trajectories:
            # filter by orientation to extract only the tracks back from the lights
            # (from right to left)
            seqs[1] = seqs[1].filter_by_feature("psi", -np.pi / 2, np.pi / 2, ret='outside')
            
            if plot_results:
                seqs[1].plot_xy(ax=ax, colors=colors[0 : len(seqs[1].tracks)])
                ax.set_ylim(min(trk.data[:, 1]) * 1.1, max(trk.data[:, 1]) * 1.1)
                plt.legend()
            
        else:
            # filter by orientation to extract only the tracks towards the lights
            # (from left to right)
            seqs[1] = seqs[1].filter_by_feature("psi", -np.pi / 2, np.pi / 2)
            if plot_results:
                seqs[1].plot_xy(ax=ax, colors=colors[0 : len(seqs[1].tracks)])
                ax.set_ylim(min(trk.data[:, 1]) * 1.1, max(trk.data[:, 1]) * 1.1)
                #plt.legend()

        if self.save_reports:
            figpath = os.path.join(self.paths.getdir_data_processed_reports(),
                                   f"fig_raw-runs_{self.experiment_name}-{self.participant_id}-{self.session_name}.png")
            fig.set_size_inches(10, 4)
            fig.savefig(figpath)

        self.t_runs = []
        self.id_runs = []
        for i in range(len(seqs[1].tracks)):
            self.t_runs.append([seqs[1][i].t_begin, seqs[1][i].t_end])
            seqs[1][i].track_id = self.session_name + "_" + f"gnss_split#{i}"

        self.bike_gnss_data = seqs[1]
        print(f"Average run time: {(self.t_end-self.t_begin)/len(seqs[1].tracks)} ... ", end="")


        return self.bike_gnss_data

    def _segment_runs(self, trk, name=""):
        """
        Chop a track into a sequence of tracks corresponding the the begin
        and end times of the GNSS runs.

        Parameters
        ----------
        trk : trajdatamanager.datamanager.Track
            The track to be chopped up.

        Returns
        -------
        trajdatamanager.datamanager.Sequence
            The sequence of tracks.

        """

        assert self.t_runs is not None, "Load the GNSS position data first!"

        tracks = []

        # created copys of the track and crop them to the run begin and end
        # times
        i = 0
        for ti in self.t_runs:

            trk_new = copy.copy(trk)
            trk_new.track_id = self.session_name + "_" + name + f"_split#{i}"
            trk_new.metadata["split_nr"] = i
            trk_new.crop_to_timespan(
                ti[0] - dt.timedelta(seconds=1),
                ti[1] + dt.timedelta(seconds=1),
            )
            tracks.append(trk_new)

            i += 1

        return Sequence(tracks)

    def _calibrate_target_locations(
        self,
        ymin=24.8,
        clusters_init=np.array([[25, 25, 25, 25, 25], [8, 10, 12, 14, 16]]).T
    ):
        """ Find positons of the command lights from GNSS measurements. 
        """

        #attempt to read
        try: 
            target_calib = read_yaml(self.paths.getfilepath_targetcalibration())
            if self.experiment_name in target_calib:
                self.target_locations = np.array(target_calib[self.experiment_name]['target_locations'])
                print("loading from file ... ", end="")
                return self.target_locations
        except FileNotFoundError:
            target_calib = {}

        dir_target_calibration = os.path.join(
            self.dir_base, self.subdir_target_calibration
        )

        dataman = RTKLibGNSSManager(
            dir_target_calibration, reference_location=self.reference_location
        )
        # load data
        seq = dataman.load_sequence()

        # rotate to align to y-axis
        seq.rotate_xy(self.rotation, deg=True)

        # prefilter by removing points that are not at the fence
        points = seq.tracks[0].data[:, 0:2]
        points = points[points[:, 0] > ymin, :]

        # DBSCAN to identify clusters
        eps = np.arange(0.01, 0.1, 0.01)
        for e in eps:
            dbscan = DBSCAN(eps=e, min_samples=150)
            clustering = dbscan.fit(points)
            labels = clustering.labels_

        # target locations are the mean cluster position
        self.target_locations = np.zeros((5, 2))
        for i in range(5):
            self.target_locations[i, :] = np.mean(points[labels == i], 0)

        # plot
        if self.plot:
            fig, ax = plt.subplots(1, 1, layout='constrained')
            ax.scatter(
                seq.tracks[0].data[:, 0],
                seq.tracks[0].data[:, 1],
                color="gray",
                label="all points",
                s=3,
            )
            for i in range(max(labels) + 1):
                ax.scatter(
                    points[labels == i, 0],
                    points[labels == i, 1],
                    color=colors[i],
                    label=f"cluster#{i}",
                )
            ax.scatter(
                points[:, 0],
                points[:, 1],
                color="black",
                label="used for clustering",
                s=5,
            )
            ax.scatter(
                self.target_locations[:, 0],
                self.target_locations[:, 1],
                label="cluster_centers",
                marker="x",
                s=20,
                color="orange",
            )
            for i in range(5):
                sigma = np.var(points[labels == i, :], axis=0)
                ax.annotate(
                    (
                        f"({self.target_locations[i, 0]:.2f}+/-{sigma[0]:.2f}, "
                        f"{self.target_locations[i, 1]:.2f}+/-{sigma[1]:.2f})"
                    ),
                    xy=self.target_locations[i, :],
                )
            ax.set_aspect("equal")
            ax.set_title(f"Target location calibration: {self.experiment_name}")
            ax.set_xlim([np.mean(self.target_locations[:, 0])-5, np.mean(self.target_locations[:, 0])+5])
            ax.set_ylim([np.min(self.target_locations[:, 1])-2, np.max(self.target_locations[:, 1])+2])
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            plt.legend(loc='upper left')

            if self.save_reports:
                filepath = os.path.join(self.paths.getdir_data_processed_reports(), 
                                        f"fig_target-loc-calibration_{self.experiment_name}.png")
                fig.set_size_inches(7,9)
                fig.savefig(filepath)

        if max(labels) != 4:
            raise ValueError("Could not identify five clusters!")
            
        if self.flip_target_locs:
            self.target_locations = np.flip(self.target_locations,0)

        if self.experiment_name not in target_calib:
            parts = self._get_testday_participants(self.experiment_name)
            data = [self.target_locations[i,:].tolist() for i in range(self.target_locations.shape[0])]
            target_calib[self.experiment_name] = {'target_locations': data,
                                                  'participants': parts}
            with open(self.paths.getfilepath_targetcalibration(new=True), "w") as f:
                yaml.dump(target_calib, f) 

        return self.target_locations
    

    def _get_testday_participants(self, testday):
        """ Get all participants of a certain testday.
        """
        filepath_exp_index = self.paths.getfilepath_data_experimentindex()
        exp_index = read_yaml(filepath_exp_index)

        for testday in exp_index['testdays']:
            testday_participants = []
            for sess in exp_index['testdays'][testday]['sessions']:
                part = exp_index['testdays'][testday]['sessions'][sess]['participant_id']
                testday_participants.append(str(part))
            if str(self.participant_id) in testday_participants:
                break
        testday_participants = np.unique(testday_participants).tolist()
            
        return testday_participants
                

    def _transform_gnss_to_rearwheel(self):    
        """ Transform the GNSS measurements to the position of the rear wheel. 
        """    
       
        for trk_gnss, trk_dyn, trk_cmd in zip(self.bike_gnss_data, 
                                     self.bike_dynamic_data,
                                     self.command_data):
            
            gnss_data = expand_timebase(trk_gnss.data, 
                                        trk_gnss.t,
                                        trk_dyn.t)
            
            trk_gnss.data = gnss_data
            trk_gnss.t = trk_dyn.t
            trk_gnss.calc_time_properties()
            
            x, y = self.bike_geom.transform_gnss2center(trk_gnss['x'], 
                                                        trk_gnss['y'], 
                                                        trk_dyn['psi'], 
                                                        trk_dyn['phi'])
            
            trk_gnss['x'] = x
            trk_gnss['y'] = y
            
            trk_gnss = update_yaw(trk_gnss)
            trk_cmd = update_yaw_command(trk_gnss, trk_cmd, keys_cmd=['p_x_c',
                                                                      'p_y_c'])
    

    def write_control_id_data(self, out_directory, filter_data = True, plot=False):
        """
        Write the control id data of this trial and run to file. Automatically
        infers frequency and speed category from data.
    
        Parameters
        ----------
        out_directory : str
            Directory of the results to be written in.
        """

        print("    Writing to control id data to file ... ", end="")
        
        if not os.path.isdir(out_directory):
            msg = (f"Path '{out_directory}' does not point to an existing "
                   f"directory!")   
            raise ValueError(msg) 
        
        if not self.loaded_data:
            self.load(plot_data=False)
            
        
        #base file name
        fname_base = "p"+ self.participant_id 
        f_design = np.array([1/3, 1/1.5])
        f_design_str = ['03', '06']
        
        for n_split in range(self.command_data.n):
            
            fname = fname_base
            
            if plot:
                rcid_data_dict, fig, ax = self.get_control_id_data(n_split, filter_data=filter_data, plot=True)
            else:
                rcid_data_dict = self.get_control_id_data(n_split, filter_data=filter_data)

            rcid_data_dict = {self.feature_map[k]:v for k, v in rcid_data_dict.items()}

            #frequency identifier
            f_cmd = estimate_command_frequency(rcid_data_dict['py_cmd_m'])
            i_cmd = np.argmin(np.abs(f_design - f_cmd))
            fname += "_f" + f_design_str[i_cmd] 
            
            #speed identifier
            commanded_speeds = np.array([8.0, 11.0, 14.0])
            i_speed = np.argmin(np.abs(commanded_speeds/3.6 - np.mean(rcid_data_dict['v_m/s'])))
            fname += f"_v{int(commanded_speeds[i_speed]):02d}"
            
            #split identifier
            fname += f"_r{n_split:02d}"
            
            df = pd.DataFrame(rcid_data_dict)
            
            df.to_csv(os.path.join(out_directory, fname+".csv"), sep=";")
            
            if plot:
                fig.set_size_inches(5,8)
                fig.savefig(os.path.join(out_directory, fname+".png"))

        print("done!")
    
            
    def get_control_id_data(self, split_nr, filter_data=True, plot=False):
        """ Get control id data for a specific run.

        Parameters
        ----------
        split_nr : int
            The number of the run.
        filter_data : bool, optional
            If True, applies a UKF to fuse and smooth data from different sensors. Default is True
        plot : bool, optional
            Make a plot of the the data. Default is False

        Returns
        -------
        rcid_data : dict
            A dictionary of the control id data.
        """
        
        t = np.array(self.command_data[split_nr].get_relative_time())
        
        #output feature map

        #output dict
        rcid_data = {"t": t}
        
        #command data
        for key in ["p_x_c", "p_y_c"]:
            rcid_data[key] = self.command_data[split_nr][key]
        
        #gnss data
        gnss_data = expand_timebase(self.bike_gnss_data[split_nr].data, 
                                    self.bike_gnss_data[split_nr].t,
                                    self.command_data[split_nr].t)
        
        for key in ['x', 'y', 'psi', 'psi_c']:
            idx = self.bike_gnss_data[split_nr].data_feature_keys.index(key)
            if key in ['x', 'y']:
                key = f"p_{key}"
            rcid_data[key] = gnss_data[:,idx]

        #dynamic data
        for key in ['dpsi', 'phi', 'dphi', 'delta', 'ddelta', 'v', 'a']:
            rcid_data[key] = self.bike_dynamic_data[split_nr][key]
            
        #filter data:
        if filter_data:
            rcid_data_filtered = self._filter(rcid_data)

            #plot data
            if plot:
                fig, ax = self.plot_control_id_data(rcid_data, rcid_data_filtered)
                return rcid_data_filtered, fig, ax
            else:
                return rcid_data_filtered
        else:
            #plot data
            if plot:
                fig, ax = self.plot_control_id_data(rcid_data)
                return rcid_data, fig, ax
            else:
                return rcid_data

            
    def load(self, plot_data=True, verbose=True):
        """ Load the data of this session.

        Loads the raw sensor data and performs time alignment. 

        Parameters
        ----------
        plot_data : bool, optional
            Make a plot of the data
        verbose : bool, optional
            Vebose output
        """

        # target locations
        if verbose:
            print("    Calibrating target locations ... ", end="")
        self._calibrate_target_locations()
        if verbose:
            print("done!")

        # get times
        if verbose:
            print("    Identifying time span ... ", end="")
        self.t_begin, self.t_end = self._load_times()
        if verbose:
            print("done!")

        # run gnss data
        if verbose:
            print("    Loading position data ... ", end="")
        self._load_runs_gnss()
        if verbose:
            print("done!")

        # run command data
        if verbose:
            print("    Loading command data ... ", end="")
        self._load_cmds()
        if verbose:
            print("done!")

        # run bike dynamics data
        if verbose:
            print("    Loading can data ... ", end="")
        self._load_bikedynamics()
        if verbose:
            print("done!")
            
        # transform gnss to bikecenter
        if verbose:
            print("    Transforming gnss data to rear-wheel contact point ... ", end="")
        self._transform_gnss_to_rearwheel()
        if verbose:
            print("done!")

        self.loaded_data = True

        if plot_data:
            if verbose:
                print("    Plotting data ... ", end="")
            ax_xy, axes_gnss, axes_cmd, axes_dyn = self.plot()
            if verbose:
                print("done!")

            return ax_xy, axes_gnss, axes_cmd, axes_dyn
        return None, None, None, None


    def plot(self):
        """ Plot the raw data from the different sensors. 
        """

        # plot xy
        fig_xy, ax_xy = plt.subplots(1, 1)
        ax_xy.set_title(
            f"Position Trajectories\n{self.experiment_name}/{self.participant_id}/{self.session_name}"
        )

        ax_xy.set_aspect("equal")
        self.bike_gnss_data.plot_xy(
            ax=ax_xy, colors=colors[0 : len(self.bike_gnss_data.tracks)]
        )
        self.target_locations
        ax_xy.scatter(self.target_locations[:, 0], self.target_locations[:, 1])
        plt.legend()

        # plot time
        axes_gnss = self.bike_gnss_data.plot()
        axes_gnss[0].set_title(f'GNSS Measurements\n{self.experiment_name}/{self.participant_id}/{self.session_name}')
        
        axes_cmd = self.command_data.plot()
        axes_gnss[0].set_title(f'Command data\n{self.experiment_name}/{self.participant_id}/{self.session_name}')
        
        axes_dyn = self.bike_dynamic_data.plot(plot_over_timestamps=True)
        axes_dyn[0].set_title(f'Bike dynamics\n{self.experiment_name}/{self.participant_id}/{self.session_name}')
        
        plt.legend()

        return ax_xy, axes_gnss, axes_cmd, axes_dyn

        
    def plot_control_id_data(self, data_measured, data_filtered=None, title=None):
        """ Plot measured and filtered data for control identification. 

        Creates an overlay of measured and filtered data. 

        data_measured : dict
            The measured sensor data.
        data_fitlered : dict, optional
            The filtered sensir data.
        title : str, optional
            A title for the plot. 
        """
        
        fig_t, axes_t = plt.subplots(10,1, sharex=True, layout='constrained')
        axes_t[-1].set_ylabel(self.feature_map["t"])
        
        keys = ["p_x", "p_y", "psi", "v", "phi", "delta", "dpsi",
                "dphi", "ddelta", "a"] 
        
        t = data_measured['t']
        
        #command
        axes_t[2].plot(t, data_filtered['psi_c'], label='command', color='gray')
        
        for i, k in enumerate(keys):
            
            finite = np.isfinite(data_measured[k])
            axes_t[i].plot(t[finite], data_measured[k][finite],
                           label = 'raw')
            
            if data_filtered is not None:
                axes_t[i].plot(t, data_filtered[k], label='filtered')

            axes_t[i].set_ylabel(self.feature_map[k])
            axes_t[i].grid()
       
        #axes_t[2].legend()
        
        if title is not None:
            axes_t[0].set_title(title)
            
        return fig_t, axes_t

   
class InstrumentedBikeGeometry:
    """ A class that describes the geometry of the instrumented bicycle and 
    provides functions to transform sensor measurements to bicycle coordiantes.
    """
    
    def __init__(self, bike_params):
        """
        Create and InstrumentedBikeGeometry object. 
        
        bike_params : dict
            A dictionary containing the bicycle dimensions. Must contain either
            the dimensions (hb, lb) or (lc, lsp, lgnss, alpha). If both are 
            present, a boolean 'use_seatpost' can be provided to select the 
            seatpost measurements. 

        """
        
        self.params = bike_params
        
        # Create symbols and reference frames
        E, B, C = sm.symbols('E B C ', cls=me.ReferenceFrame)
        psi, phi = me.dynamicsymbols(r'\psi \phi')
        
        h_gnss, l_gnss = sm.symbols('h_gnss, l_gnss')
        gyro_z, gyro_x = sm.symbols('gyro_z gyro_x')
        
        # Set rotations of reference frames
        B.orient_body_fixed(E, (psi, sm.pi + phi, 0), 'ZXY') # Bike frame 
        C.orient_axis(B, 0, B.x) # GNSS sensor frame equals bike frame
        
        # transformation from GNSS coordinates to bike coordinates.
        
        if not np.all([k in self.params.keys() for k in ['l_gnss', 'h_gnss']]):
            msg = (f"The calibration must include measurements of the antenna"
                   f"height over ground 'h_gnss' and horizontal distance"
                   f"to the rear-wheel contanct point 'l_gnss'.")
            raise ValueError(msg)

        P = me.Point('P')
        P1 = P.locatenew('P_1', - h_gnss * B.z)
        Pgnss = P1.locatenew('P_{GNSS}', - l_gnss * B.x)
        

        param_vals = {l_gnss: bike_params['l_gnss'],
                      h_gnss: bike_params['h_gnss']}
        r_P_Pgnss = P.pos_from(Pgnss).express(E)
        r_P_Pgnss = r_P_Pgnss.subs(param_vals)
        
        self.eval_r_P_Pgnss = sm.lambdify((psi, phi), r_P_Pgnss.to_matrix(E))
        
        # transformation of IMU measurements to roll/yaw
        Cz_w_E_exprB = C.ang_vel_in(E).dot(B.z)
        Cz_w_E_exprE = sm.trigsimp(C.ang_vel_in(E).dot(E.z))
        Cz_w_E_exprE = sm.solve(Cz_w_E_exprB - gyro_z, Cz_w_E_exprE)[0]
        
        Cx_w_E_exprB = C.ang_vel_in(E).dot(B.x)
        Cx_w_E_exprB = sm.solve(Cx_w_E_exprB - gyro_x, Cx_w_E_exprB)[0]
        
        self.eval_C_w_E = sm.lambdify((phi, gyro_x, gyro_z), 
                                      sm.Matrix([[Cz_w_E_exprE],
                                                 [Cx_w_E_exprB]]))
        

    def transform_gnss2center(self, p_x_gnss, p_y_gnss, psi, phi):
        """ Transform the coordinates into the location of the bicycle center
        """
        
        translation = self.eval_r_P_Pgnss(psi, phi)
        
        p_x = p_x_gnss + translation[0,0,:]
        p_y = p_y_gnss + translation[1,0,:]
        
        return p_x, p_y
    
    
    def transform_imu2bike(self, psi_m, phi_m, gyro_x_m, gyro_y_m, a):
        
        rotation = self.eval_C_w_E(phi_m, gyro_x_m, gyro_y_m)
        
        dpsi = rotation[0,0,:]
        dphi = rotation[1,0,:]
        
        psi = - psi_m
        
        return psi, dpsi, phi_m, dphi, a   
            
    def transform_N2E(self, data_dict):
        
        data_dict['p_y'] = - data_dict['p_y']
        data_dict['psi'] = - data_dict['psi']
        data_dict['dpsi'] = - data_dict['dpsi']
        data_dict['delta'] = - data_dict['delta']
        data_dict['ddelta'] = - data_dict['ddelta']
        
        return data_dict
    
    def transform_E2N(self, data_dict):
        
        return self.transform_N2E(data_dict)
        
# Datamanager overwrites for Postprocessed data loading and generating statistics

class RCIDTrack(Track):

    def get_command_indices(self):
        """ Return the indices of command steps."""
        indices = np.argwhere(np.abs(np.diff(self['p_y_c']))>0).flatten()

        return indices

    def get_state(self, index=0, model='balancingrider'):
        """ Return the values of this track at index as state in csf form (E-frame).

        Parameters
        ----------
        index : int, optional
            The time index of the requested state. Default is 0 (initial state)
        model : str or type, optional
            The model type for which the initial state is requested as a string ['balancingrider', 'planarpoint'] or as a type
            [cyclistsocialforces.vehicle.BalancingRiderBicycle, cyclistsocialforces.vehicle.PlanarPointBicycle]. Default is 
            'balancingrider'. 

        Returns
        -------
        s0 : np.ndarray
            The inital state of this track
        """

        if model=='balancingrider' or model==BalancingRiderBicycle:
            s0 = np.array([
                self['p_x'][index],
                self['p_y'][index],
                self['psi'][index],
                self['v'][index],
                self['delta'][index],
                self['phi'][index],
                self['ddelta'][index],
                self['dphi'][index],
            ])
        elif model=='planarpoint' or model==PlanarPointBicycle:
            s0 = np.array([
                self['p_x'][index],
                self['p_y'][index],
                self['psi'][index],
                self['v'][index],
            ])
        else:
            raise NotImplementedError(f"Requesting the initial state for model '{model}' is not implemented!")
        
        return s0
    
    def get_statistics(self):
        
        data = self.data[:,:6]
        
        #track data
        data_rms = np.sqrt(np.nanmean(data**2, axis=0))
        data_std = np.nanstd(data, axis=0)
        data_min = np.nanmin(data, axis=0)
        data_max = np.nanmax(data, axis=0)
        data_range = data_max - data_min
        
        #command data
        jumps = np.abs(np.diff(self['p_y_c']))
        jump_times = np.abs(np.diff(self['t'][1:][jumps.astype(bool)]))
        
        jumps = jumps[jumps != 0] 
        n_commands = jumps.size
        mean_cwidth = np.mean(jumps)
        max_cwidth = np.amax(jumps)
        min_cwidth = np.amin(jumps)
        
        mean_cfreq = np.mean(1/jump_times)
        max_cfreq = np.amax(1/jump_times)
        min_cfreq = np.amin(1/jump_times)
        
        #output
        out = [float(self.metadata['participant']), self.metadata['v_cmd'], self.metadata['f_cmd']]
        for i in range(6):
            out += [data_rms[i], data_std[i], data_min[i], data_max[i], data_range[i]]
        out += [n_commands, mean_cwidth, max_cwidth, min_cwidth, mean_cfreq, max_cfreq, min_cfreq]
        
        return out
    
    def get_statistics_keys(self):
        labels = ['participant', 'v_cmd', 'f_cmd']
        for i in range(6):
            key = self.data_feature_keys[i]
            labels += [key+'_rms', key+'_std', key+'_min', key+'_max', key+'_range']
        labels += ['n_commands', 'cwidth_mean', 'cwidth_max', 'cwidth_min', 'cfreq_mean', 'cfreq_max', 'cfreq_min']    
        return labels
    
    def get_statistics_dict(self):
        keys = self.get_statistics_keys()
        vals = self.get_statistics()
        
        return {k: v for k, v in zip(keys, vals) }

class RCIDDataSequence(Sequence):
    
    def get_statistics(self):
        statistics = []
        for part_run in self:
            statistics.append(part_run.get_statistics())
            
        labels = part_run.get_statistics_keys()

        stats = pd.DataFrame(statistics, columns=labels)
        
        return stats

class RCIDDataManager(DataManager):

    def load_track(self, filepath_rcid_data):
        """ Load a track given the full path a processed rider control id data .csv file

        Parameters
        ----------
        filepath_rcid_data : str
            A path to a processed rider control identification data file in .csv format.

        Returns
        -------
        trk : RCIDTrack
            A Track representing the rider control id data.
        """
        
        # load data dict from csv
        data_dict = pd.read_csv(filepath_rcid_data, sep=";", index_col=0)
        rev_feature_map = get_reversed_feature_map()
        data_dict = {rev_feature_map[k]: v for k, v in data_dict.items() if k in rev_feature_map}
        
        # convert to array
        data = []
        keys = []
        for k, v in data_dict.items():
            if k != "t":
                data.append(v)
                keys.append(k)
        data = np.array(data).T
        
        # track metadata
        fname = fileparts(filepath_rcid_data)[1]
        metadata = {
            "track_type": "PostprocessedControlIDData",
            "source": filepath_rcid_data,
            "participant_id": re.findall(r'p\d{3}', fname)[0][1:],
            "f_cmd": int(re.findall(r'f\d{2}', fname)[0][2:])/10,
            "v_cmd": int(re.findall(r'v\d{2}', fname)[0][1:]),
            "run_number": int(re.findall(r'r\d{2}', fname)[0][1:]),
            "run_id": fname
        }
            
        # make track
        trk = RCIDTrack(fileparts(filepath_rcid_data)[1],
                    0,
                    [dt.datetime(2000, 1, 1)+dt.timedelta(seconds=t) for t in data_dict['t']],
                    data,
                    yaw_feature_index = 2,
                    data_feature_keys = keys,
                    metadata=metadata,
            )
        return trk
        
        
    def load_participant(self, participant_id, subset=""):
        """ Load a all runs of a given participant. May be from a subset.

        Parameters
        ----------
        participant_id : str
            The id of the participant to load.
        subset : str
            The subset to look for the split. For example "steps".
        """
        
        dir_part = os.path.join(self.dir, participant_id, subset)
        
        files = []
        for f in os.listdir(dir_part):

            if not os.path.isfile(os.path.join(dir_part, f)):
                continue
            if not f[-4:]==".csv":
                continue
            if not f[:4]==f"p{participant_id}":
                continue
            
            files.append(f)
        
        tracks = []
        for i in range(len(files)):
            tracks.append(self.load_track(os.path.join(dir_part, files[i])))
            
        return RCIDDataSequence(tracks)
    
    
    def load_split(self, split_id, subset=""):
        """ Load a track with a given id. May be from a subset.

        Parameters
        ----------
        split_id : str
            The id o the file to load. Must be format pXXX_fXX_vXX...
        subset : str
            The subset to look for the split. For example "steps".
        """
        
        fname_pattern = r'p(\d{3})_f(\d{2})_v(\d{1,2})'
        matches = re.findall(fname_pattern, split_id)
        if len(matches) != 1:
            raise UnexpectedStringArgError('split file name', split_id, 
                                           fname_pattern)
        if split_id[-4:] != '.csv':
            split_id += '.csv'
        
        dir_part = os.path.join(self.dir, matches[0][0], subset)
        file = os.path.join(dir_part, split_id)
        #all_files = [f for f in os.listdir(dir_part) 
        #             if re.findall(fname_pattern, f)]
        #splitid = all_files.index(split_id)
        
        return self.load_track(file)


def find_drift(x1, x2, t_s, plot=False):
    """
    Given two measurements x1 and x2 of the same signal that are (rougly) 
    aligned in time, identify the time drift of one signal with respect to 
    the other. 

    Parameters
    ----------
    x1 : array-like
        First signal.
    x2 : array-like
        Second signal.
    t_s : float
        Sampling time of both signals.
    plot : bool, optional
        Plot the drift function. 

    Returns
    -------
    drift : function
        Function calculating the time offset of signal 2 relative to signal 1 
        as a function of signal time: drift(t)

    """
    
    subsignal_length = 1
    signal_length = min(len(x1) * t_s, len(x2) * t_s)
    n = int(subsignal_length / t_s)
    
    drift_offsets = []
    drift_offset_times = []
    
    for i in range(int(signal_length / subsignal_length)):
        x1i = x1[i*n:(i+1)*n]
        x2i = x2[i*n:(i+1)*n]
        
        corr = correlate(x1i, x2i, mode="full")
        
        drift_offset_times.append(t_s*(i*n+((i+1)*n - i*n)/2))
        drift_offsets.append(np.argmax(corr) - len(x2i) + 1)
    
    #estimate drift function
    drift_offsets = np.array(drift_offsets) * t_s
    drift_offset_times = np.array(drift_offset_times)[:,np.newaxis]
    reg = RANSACRegressor(max_trials=1000).fit(drift_offset_times, drift_offsets)           
    drift_offsets_fitted = reg.predict(drift_offset_times)
    
    #plot for validation
    if plot:
        fig, ax = plt.subplots(layout='constrained')
        ax.plot(drift_offset_times, drift_offsets, label="drift samples")
        ax.plot(drift_offset_times, drift_offsets_fitted, label="fitted drift function")
        ax.set_xlabel("signal time [s]")
        ax.set_ylabel("time offset [s]")
        ax.set_title(f"Time offset of signal 2 relative to signal 1 \n drift(t) = ({reg.estimator_.coef_[0]*1000:.4f}) ms/s * t + ({reg.estimator_.intercept_:.4f}) s")
    
    return reg.predict


def expand_timebase(data, time_data, time_target):
    """ Insert NaNs for time steps where data does not have measurements.

    Parameters
    ----------
    data : array_like
        The data
    time_data : array-like
        The new time of the given data. 
    time_target : array_like
        The new time to base the new data in. 
    """

    time = np.array(time_target)

    idx = np.zeros_like(time, dtype=bool)

    data_new = np.nan * np.ones((len(time), data.shape[1]))

    for i in range(len(time_data)):
        data_new[np.argwhere(time == time_data[i]), :] = data[i, :]

    return data_new 


def estimate_command_frequency(n_c):
    """ Estimate the command frequency based on the time between commands
    """
    t_design = [3, 1.5]
    command_durations = np.diff(np.argwhere(np.abs(np.diff(n_c)) > 0).flatten() * 0.01)
    mean_command_duration = np.mean(command_durations)
    i = np.argmin(abs(t_design - mean_command_duration))

    f_design = 1/t_design[i]

    return f_design
    

def estimate_yaw(x, y):
    """ Estimates the yaw angle from x-y trajectories in the world
    reference frame (z-axis into the air). 
    
    yaw = arctan(y(t+1) - y(t-1) / x(t+1) - x(t-1))
    
    Positive yaw corresponds to positive rotation about the z-axis. Zero yaw
    corresponds to an orientation in x-direction.
    
    Parameters
    ----------
    
    x : array-like,
        X trajectory
    y : array-like
        Y trajectory
        
    Returns
    -------
    
    psi : array-like
        Yaw trajectory
    """
    
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    
    if x.size != y.size:
        msg = (f"x and y must be the same size! Instead x was size {x.size} " 
               f"and y was size {y.size}!")
        raise ValueError(msg)

    finite = np.logical_and(np.isfinite(x), np.isfinite(y))

    x_fin = x[finite]
    y_fin = y[finite]

    t = np.arange(x.size)
    t_fin = t[finite][1:-1]
    
    with warnings.catch_warnings(action="ignore"): 
        #ignore div/0 warning that occurs when someone is stationary
        temp, psi = cart2polar(x_fin[2:] - x_fin[:-2], y_fin[2:] - y_fin[:-2])
    
    # stationary (i.e. dx = 0) causes nan values. Replace them with the prev.
    # orientation. 
    i_stationary = np.where(np.isnan(psi))[0]
    psi[i_stationary] = psi[i_stationary-1] 
    
    # interpolate to time base of x and y
    psi = np.interp(t, t_fin, psi)
    psi[np.logical_not(finite)]=np.nan
    
    return psi

def update_yaw(trk, keys=("x", "y", "psi")):
    """ Wrapper around estimate_track() to update the yaw for a Track object
    containing an x and a y trajectory.
    
    The track must already contain a yaw trajectory 
    (which will be overwritten.)
    
    Parameters
    ----------
    
    trk : trajdatamanger.datamanager.Track
        A track containing x, y and yaw features. 
    
    keys : list-like, optional
        A tuple of keys that retrive the x, y and psi features of the track. 
        The default is ("x", "y", "psi")
    
    Returns
    -------
    trk : trajdatamanger.datamanager.Track
        The input track with updated yaw trajectory. 
    """
    
    x = trk[keys[0]]
    y = trk[keys[1]]
        
    psi = estimate_yaw(x, y)
    
    trk[keys[2]] = psi
    
    return trk

def estimate_yaw_command(x, y, x_c, y_c):
    """ Estimate the commanded yaw angle from x-y trajectories and commanded
    x-y destinations. 
    
    Parameters
    ----------
    
    x : array-like,
        X trajectory
    y : array-like
        Y trajectory
    x_c : array-like,
        Trajectory of commanded x-destinations
    y_c : array-like
        Trajectory of commanded y-destinations
    
    Returns
    -------
    psi_c : array-like
        Trajectory of commanded yaw angles. 
    
    """
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    x_c = np.array(x_c).flatten()
    y_c = np.array(y_c).flatten()
    
    if np.all([x.size, y.size, x_c.size] == y_c.size):
        msg = (f"x, y, x_c and y_c must all be the same size!")
        raise ValueError(msg)
    
    dist, psi_c = cart2polar(x_c - x, y_c - y)
    
    return psi_c


def update_yaw_command(trk_pos, trk_cmd, 
                       keys_pos=("x", "y", "psi_c"), 
                       keys_cmd=("x_c", "y_c")):
    """ Wrapper around estimate_yaw_command() to update the commanded yaw 
    from the Track object trk_cmd using the x and a y trajectories of trk_pos.
    
    The track trk_cmd must already contain a commanded yaw trajectory 
    (which will be overwritten.)
    
    Parameters
    ----------
    
    trk_pos : trajdatamanger.datamanager.Track
        A track containing x, y, and psi_c features.      
    trk_cmd : trajdatamanger.datamanager.Track
        A track containing x_c, y_c, command features. 
    keys_pos : list-like, optional
        A tuple of keys that retrive the x, y and psi_c features of trk_pos. 
        The default is ("x", "y", "psi_c")
    keys_pos : list-like, optional
        A tuple of keys that retrive the x_c, y_c features of 
        trk_cmd. The default is ("x_c", "y_c")
    
    Returns
    -------
    trk_cmd : trajdatamanger.datamanager.Track
        The input command track with updated yaw command trajectory. 
    """
        
    x = trk_pos[keys_pos[0]]
    y = trk_pos[keys_pos[1]]
    x_c = trk_cmd[keys_cmd[0]]
    y_c = trk_cmd[keys_cmd[1]]
        
    psi_c = estimate_yaw_command(x, y, x_c, y_c)
    
    trk_pos[keys_pos[2]] = psi_c
    
    return trk_pos


def get_feature_map():
    """ Get the map between short feature names used for coding and verbose feature names
    used for data storage. 
    """

    feature_map = {
            "t": "t_s",
            "p_x": "px_m", 
            "p_y": "py_m", 
            "p_x_c": "px_cmd_m", 
            "p_y_c": "py_cmd_m", 
            "psi": "psi_rad",
            "psi_c": "psi_cmd_rad",
            "dpsi": "psidot_rad/s",
            "phi": "phi_rad",
            "dphi": "phidot_rad/s",
            "delta": "delta_rad",
            "ddelta": "deltadot_rad/s",
            "v": "v_m/s",
            "a": "a_m/s2"
        }
    
    return feature_map


def get_reversed_feature_map():
    """ Get the map between verbose feature names
    used for data storage and short feature names used for coding. 
    """

    reversed_feature_map = {v: k for k, v in get_feature_map().items()}

    return reversed_feature_map


def decode_CANedge(logfiles, databases):
    """ Decode CAN log files generated by CanEdge 2

    Parameters:
    logfiles : list
        List of paths to the MDF log files.
    databases : dict
        Dictionary containing the bus types as keys and a list of tuples with 
        DBC file paths and corresponding channel numbers as values.

    Mostly a copy of process_can_edge by Anna Marbus (no license) developed 
    during her research project. 
    """
    # Concatenate the MDF log files
    mdf = asammdf.MDF.concatenate(logfiles)
    
    # Extract bus logging data using the specified databases
    mdf_scaled = mdf.extract_bus_logging(databases)
    
    # Covert to dataframe. use_interpolation enforces a joint time grid. 
    df_can_edge = mdf_scaled.to_dataframe(use_interpolation=True,
                                          time_as_date=True)
    
    return df_can_edge

def decode_parquet(logfile):
    """ Decode a parquet CAN log file.

    Parameters
    ----------
    logfile : str
        Filepath of the CAN log file

    Returns
    -------
    df : DataFrame
        Logs
    """

    df = pd.read_parquet(logfile)

    keys = {
        'gyro_z_rad/s': 'gyro_z',
        'gyro_y_rad/s': 'gyro_y',
        'gyro_x_rad/s': 'gyro_x', 
        'accel_z_m/s2': 'accel_z', 
        'accel_y_m/s2': 'accel_y', 
        'accel_x_m/s2': 'accel_x', 
        'yaw_rad': 'yaw', 
        'pitch_rad': 'pitch', 
        'roll_rad': 'roll', 
        'wheelspeed_rear_rev/s': 'ws_rear', 
        'steer_deg': 'LWS_ANGLE', 
        'steer_rate_deg/s': 'LWS_SPEED'}
    
    df.rename(columns=keys, inplace=True)

    return df