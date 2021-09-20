# Class for parcel calculations on real soundings
# Thomas Schanzer, UNSW Sydney
# September 2021

import numpy as np
import pandas as pd

import metpy.calc as mpcalc
import metpy.constants as const
from metpy.units import units
from metpy.units import concatenate

from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

import sys


class MotionResult():
    """
    Class for results of parcel motion calculations.

    Attributes:
        height: An array of heights, with each row corresponding to a
            different initial condition.
        velocity: An array of velocitites, with each row corresponding
            to a different initial condition.
        neutral_buoyancy_time: Times at which parcels reached their
            neutral buoyancy levels.
        hit_ground_time: Times at which parcels hit the ground.
        min_height_time: Times at which parcels reached their minimum
            heights (without hitting the ground)
        neutral_buoyancy_height: The heights of the neutral buoyancy
            levels.
        neutral_buoyancy_velocity: The velocities at the neutral
            buoyancy levels.
        hit_ground_velocity: The velocities with which the parcels hit
            the ground.
        min_height: The minimum heights of the parcels (that did not hit
            the ground).
    """

    def __init__(
            self, height, velocity, neutral_buoyancy_time, hit_ground_time,
            min_height_time, neutral_buoyancy_height,
            neutral_buoyancy_velocity, hit_ground_velocity, min_height):
        """
        Instantiates a MotionResult.
        """

        self.height = height
        self.velocity = velocity
        self.neutral_buoyancy_time = neutral_buoyancy_time
        self.hit_ground_time = hit_ground_time
        self.min_height_time = min_height_time
        self.neutral_buoyancy_height = neutral_buoyancy_height
        self.neutral_buoyancy_velocity = neutral_buoyancy_velocity
        self.hit_ground_velocity = hit_ground_velocity
        self.min_height = min_height


class Environment():
    """
    Class for parcel calculations on a real atmospheric sounding.
    """

    def __init__(self, pressure, temperature, dewpoint):
        """
        Instantiates an Environment.

        Allows all variables to be calculated as functions of height
        rather than pressure by assuming hydrostatic balance.

        Args:
            pressure: Pressure array in the sounding.
            temperature: Temperature array in the sounding.
            dewpoint: Dewpoint array in the sounding.
        """

        self.pressure_raw = pressure.to(units.mbar)
        self.temperature_raw = temperature.to(units.celsius)
        self.dewpoint_raw = dewpoint.to(units.celsius)

        self.temperature_interp = interp1d(
            self.pressure_raw.m, self.temperature_raw.m,
            fill_value='extrapolate')
        self.dewpoint_interp = interp1d(
            self.pressure_raw.m, self.dewpoint_raw.m,
            fill_value='extrapolate')

        def environment_dzdp(pressure, height):
            """
            Calculates the rate of height change w.r.t. pressure, dz/dp.

            Args:
                pressure: The pressure at the point of interest, in Pa.
                height: The height of the point of interest, in m.

            Returns:
                The derivative dz/dp in m/Pa.
            """

            pressure = pressure*units.pascal
            temperature = self.temperature_from_pressure(pressure)
            dewpoint = self.dewpoint_from_pressure(pressure)

            specific_humidity = mpcalc.specific_humidity_from_dewpoint(
                pressure, dewpoint)
            mixing_ratio = mpcalc.mixing_ratio_from_specific_humidity(
                specific_humidity)
            density = mpcalc.density(pressure, temperature, mixing_ratio)
            dzdp = - 1 / (density.to(units.kg/units.meter**3).m * const.g)

            return dzdp

        # integrate the hydrostatic equation
        p_min = np.min(self.pressure_raw).m * 1e2
        p_max = np.max(self.pressure_raw).m * 1e2
        sol = solve_ivp(
            environment_dzdp, (p_max, p_min), [0],
            t_eval=np.arange(p_max, p_min - 1e-3, -10e2))
        self.pressure_interp_from_height = interp1d(
            np.squeeze(sol.y), sol.t, fill_value='extrapolate')
        self.height_interp_from_pressure = interp1d(
            sol.t, np.squeeze(sol.y), fill_value='extrapolate')


    def temperature_from_pressure(self, pressure):
        """
        Finds the environmental temperature at a given pressure.
        """

        temperature = self.temperature_interp(pressure.to(units.mbar).m)
        return temperature*units.celsius

    def dewpoint_from_pressure(self, pressure):
        """
        Finds the environmental dew point at a given pressure.
        """

        dewpoint = self.dewpoint_interp(pressure.to(units.mbar).m)
        return dewpoint*units.celsius

    def pressure(self, height):
        """
        Finds the environmental pressure at a given height.
        """

        pressure = self.pressure_interp_from_height(height.to(units.meter).m)
        return (pressure*units.pascal).to(units.mbar)

    def height(self, pressure):
        """
        Finds the height at a given environmental pressure.
        """

        height = self.height_interp_from_pressure(pressure.to(units.pascal).m)
        return height*units.meter

    def temperature(self, height):
        """
        Finds the environmental temperature at a given height.
        """

        pressure = self.pressure(height)
        temperature = self.temperature_from_pressure(pressure)
        return temperature

    def dewpoint(self, height):
        """
        Finds the environmental dew point at a given height.
        """

        pressure = self.pressure(height)
        dewpoint = self.dewpoint_from_pressure(pressure)
        return dewpoint

    def specific_humidity(self, height):
        """
        Finds the environmental specific humidity at a given height.
        """

        pressure = self.pressure(height)
        dewpoint = self.dewpoint_from_pressure(pressure)
        specific_humidity = mpcalc.specific_humidity_from_dewpoint(
            pressure, dewpoint)
        return specific_humidity

    def temperature_change(self, dq):
        """
        Calculates the temperature change due to evaporation of water.
        """

        dT = (- const.water_heat_vaporization
              * dq / const.dry_air_spec_heat_press)
        return dT.to(units.delta_degC)

    def dq_root(self, dq, height):
        """
        Calculates the saturated vs. actual specific humidity difference.

        Args:
            dq: Specific humidity change resulting from evaporation.
            height: Initial height of the parcel.

        Returns:
            The difference between saturated and actual specific
                humidity at the temperature that results from the
                evaporation.
        """

        pressure = self.pressure(height)
        initial_temperature = self.temperature(height)
        initial_specific_humidity = self.specific_humidity(height)

        final_temperature = initial_temperature + self.temperature_change(dq)
        saturation_mixing_ratio = mpcalc.saturation_mixing_ratio(
            pressure, final_temperature)
        saturation_specific_humidity = \
            mpcalc.specific_humidity_from_mixing_ratio(saturation_mixing_ratio)
        return saturation_specific_humidity - initial_specific_humidity - dq

    def maximum_specific_humidity_change(self, height):
        """
        Calculates the maximum specific humidity increase for the parcel.

        Finds the root of dq_root at the specified height.
        """

        height = np.atleast_1d(height)
        sol = [
            root_scalar(self.dq_root, args=(z,), bracket=[0, 20e-3]).root
            for z in height]
        return np.squeeze(concatenate(sol)) * 1

    def density(self, height):
        """
        Finds the environmental density at a given height.
        """

        pressure = self.pressure(height)
        dewpoint = self.dewpoint(height)
        temperature = self.temperature(height)

        specific_humidity = mpcalc.specific_humidity_from_dewpoint(
            pressure, dewpoint)
        mixing_ratio = mpcalc.mixing_ratio_from_specific_humidity(
            specific_humidity)
        density = mpcalc.density(pressure, temperature, mixing_ratio)

        return density

    def parcel_density(self, height, initial_height, specific_humidity_change):
        """
        Calculates the density of a parcel after precipitation.

        Args:
            height: The height of the parcel.
            initial_height: The initial height of the parcel (i.e., its
                height when the precipitation occurred).
            specific_humidity_change: The change in specific humidity
                that resulted from the precipitation.

        Returns:
            The density of the parcel.
        """

        initial_pressure = self.pressure(initial_height)
        pressure = self.pressure(height)
        initial_temperature = (self.temperature(initial_height)
                               + self.temperature_change(
                                   specific_humidity_change))
        specific_humidity = (self.specific_humidity(initial_height)
                             + specific_humidity_change)
        mixing_ratio = mpcalc.mixing_ratio_from_specific_humidity(
            specific_humidity)
        temperature = mpcalc.dry_lapse(
            pressure, initial_temperature, reference_pressure=initial_pressure)
        density = mpcalc.density(pressure, temperature, mixing_ratio)

        return density

    def parcel_buoyancy(
            self, height, initial_height, specific_humidity_change):
        """
        Calculates the buoyancy of a parcel after precipitation.

        Args:
            height: The height of the parcel.
            initial_height: The initial height of the parcel (i.e., its
                height when the precipitation occurred).
            specific_humidity_change: The change in specific humidity
                that resulted from the precipitation.

        Returns:
            The buoyant force per unit mass on the parcel.
        """

        environment_density = self.density(height)
        density = self.parcel_density(
            height, initial_height, specific_humidity_change)
        buoyancy = (environment_density - density) / density * const.g
        return buoyancy

    def parcel_buoyancy_root_function(
            self, height, initial_height, specific_humidity_change):
        """
        Calculates the buoyancy of a parcel after precipitation.

        Args:
            height: The height of the parcel in metres, as a dimensionless
                number.
            initial_height: The initial height of the parcel (i.e., its
                height when the precipitation occurred).
            specific_humidity_change: The change in specific humidity
                that resulted from the precipitation.

        Returns:
            The buoyant force per unit mass on the parcel.
        """

        buoyancy = self.parcel_buoyancy(
            height*units.meter, initial_height, specific_humidity_change).m
        return buoyancy

    def neutral_buoyancy_level(self, initial_height, specific_humidity_change):
        """
        Calculates the neutral buoyancy height of a parcel.

        Args:
            initial_height: The initial height of the parcel.
            specific_humidity_change: The change in specific humidity due to
                evaporation.

        Returns:
            The height at which the buoyancy of the parcel is zero.
        """

        initial_height = np.atleast_1d(initial_height)
        specific_humidity_change = np.atleast_1d(specific_humidity_change)
        sol = np.zeros((len(initial_height), len(specific_humidity_change)))
        max_height = self.height(np.min(self.pressure_raw) + 10*units.mbar).m

        for i, z0 in enumerate(initial_height):
            for ii, dq in enumerate(specific_humidity_change):
                if dq <= self.maximum_specific_humidity_change(z0):
                    try:
                        sol[i,ii] = root_scalar(
                            self.parcel_buoyancy_root_function,
                            args=(z0, dq),
                            x0=z0.to(units.meter).m, x1=0,
                            bracket=[0, z0.to(units.meter).m]
                        ).root
                    except ValueError:
                        sol[i,ii] = 0
                    sys.stdout.write(
                        '\rCalculating buoyancy level '
                        '{} of {}.'.format(
                            i*len(specific_humidity_change) + ii + 1,
                            len(specific_humidity_change)*len(initial_height)))
                else:
                    sol[i,ii] = np.nan

        return np.squeeze(sol)/1e3*units.km

    def modified_motion(self, time, initial_height, dq):
        """
        Calculates parcel motion for different specific humidity changes.

        Args:
            time: Array of time points.
            initial_height: Array of initial heights.
            dq_range: Array of initial changes in specific humidity due
                to evaporation.

        Returns:
            A MotionResult object.
        """

        # independent variables
        initial_height = np.atleast_1d(initial_height).to(units.meter).m
        dq = np.atleast_1d(dq)
        if len(initial_height) != len(dq):
            raise ValueError(
                'Initial height and specific humidity change arrays must '
                'have the same length.')
        initial_state = [[z0, 0] for z0 in initial_height]
        time = time.to(units.second).m

        def motion_ode(
                time, state, initial_height, dq):
            """
            Defines the equation of motion for a parcel.
            """

            buoyancy = self.parcel_buoyancy(
                state[0]*units.meter, initial_height*units.meter, dq)
            return [state[1], buoyancy.magnitude]

        # event function for solve_ivp, zero when parcel reaches min height
        min_height = lambda time, state, *args: state[1]
        min_height.direction = 1  # find zero that goes from negative to positive
        min_height.terminal = True  # stop integration at minimum height

        # event function for solve_ivp, zero when parcel hits ground
        hit_ground = lambda time, state, *args: state[0]
        hit_ground.terminal = True  # stop integration at ground

        # prepare empty arrays for data
        height = np.zeros((len(dq), len(time)))
        height[:] = np.nan
        velocity = np.zeros((len(dq), len(time)))
        velocity[:] = np.nan

        neutral_buoyancy_time = np.zeros(len(dq))
        hit_ground_time = np.zeros(len(dq))
        min_height_time = np.zeros(len(dq))

        neutral_buoyancy_height = np.zeros(len(dq))
        neutral_buoyancy_velocity = np.zeros(len(dq))
        hit_ground_velocity = np.zeros(len(dq))
        min_height_height = np.zeros(len(dq))

        for i in range(len(dq)):
            sys.stdout.write(
                '\rCalculating profile {0} of {1}.'
                '   '.format(i+1, len(dq)))

            # event function for solve_ivp, zero when parcel is neutrally
            # buoyant
            neutral_buoyancy = lambda time, state, *args: motion_ode(
                time, state, initial_height[i], dq[i])[1]

            sol = solve_ivp(
                motion_ode,
                [np.min(time), np.max(time)],
                initial_state[i],
                t_eval=time,
                args=(initial_height[i], dq[i]),
                events=[neutral_buoyancy, hit_ground, min_height])

            height[i,:len(sol.y[0,:])] = sol.y[0,:]
            velocity[i,:len(sol.y[1,:])] = sol.y[1,:]

            # record times of events
            # sol.t_events[i].size == 0 means the event did not occur
            neutral_buoyancy_time[i] = (  # record only the first instance
                sol.t_events[0][0] if sol.t_events[0].size > 0 else np.nan)
            hit_ground_time[i] = (
                sol.t_events[1][0] if sol.t_events[1].size > 0 else np.nan)
            min_height_time[i] = (
                sol.t_events[2][0] if sol.t_events[2].size > 0 else np.nan)

            # record states at event times
            neutral_buoyancy_height[i] = (  # record only the first instance
                sol.y_events[0][0,0] if sol.y_events[0].size > 0 else np.nan)
            neutral_buoyancy_velocity[i] = (  # record only the first instance
                sol.y_events[0][0,1] if sol.y_events[0].size > 0 else np.nan)
            hit_ground_velocity[i] = (
                sol.y_events[1][0,1] if sol.y_events[1].size > 0 else np.nan)
            min_height_height[i] = (
                sol.y_events[2][0,0] if sol.y_events[2].size > 0 else np.nan)

        result = MotionResult(
            np.squeeze(height)*units.meter,
            np.squeeze(velocity)*units.meter/units.second,
            np.squeeze(neutral_buoyancy_time)*units.second,
            np.squeeze(hit_ground_time)*units.second,
            np.squeeze(min_height_time)*units.second,
            np.squeeze(neutral_buoyancy_height)*units.meter,
            np.squeeze(neutral_buoyancy_velocity)*units.meter/units.second,
            np.squeeze(hit_ground_velocity)*units.meter/units.second,
            np.squeeze(min_height_height)*units.meter)

        return result
