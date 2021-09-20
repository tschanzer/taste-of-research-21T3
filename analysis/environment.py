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


class Environment():
    def __init__(self, pressure, temperature, dewpoint):
        self.pressure_raw = pressure.to(units.mbar)
        self.temperature_raw = temperature.to(units.celsius)
        self.dewpoint_raw = dewpoint.to(units.celsius)

        self.temperature_interp = interp1d(
            self.pressure_raw.m, self.temperature_raw.m, kind='linear')
        self.dewpoint_interp = interp1d(
            self.pressure_raw.m, self.dewpoint_raw.m, kind='linear')

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
        self.pressure_interp_from_height = interp1d(np.squeeze(sol.y), sol.t)
        self.height_interp_from_pressure = interp1d(sol.t, np.squeeze(sol.y))


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

    def motion_from_dq(self, time, dq_range, initial_height):
        """
        Calculates parcel motion for different specific humidity changes.

        Args:
            time: Array of time points.
            dq_range: Array of initial changes in specific humidity due
                to evaporation.
            initial_height: Initial height of the parcels.

        Returns:
            An array of parcel heights and an array of vertical
                velocities, with each row corresponding to a different
                initial specific humidity change and each column
                corresponding to a different point in time.
        """

        def motion_ode(
                time, state, initial_height, specific_humidity_change):
            buoyancy = self.parcel_buoyancy(
                state[0]*units.meter, initial_height, specific_humidity_change)
            return [state[1], buoyancy.magnitude]

        dq_range = np.atleast_1d(dq_range)
        initial_state = [initial_height.to(units.meter).m, 0]
        height = np.zeros((len(dq_range), len(time)))
        velocity = np.zeros((len(dq_range), len(time)))
        time = time.to(units.second).m

        for i in range(len(dq_range)):
            sol = solve_ivp(
                motion_ode, [np.min(time), np.max(time)], initial_state,
                t_eval=time, args=(initial_height, dq_range[i]))
            height[i,:] = sol.y[0,:]
            velocity[i,:] = sol.y[1,:]
            sys.stdout.write(
                '\rCalculating profile {0} of {1}.'
                '   '.format(i+1, len(dq_range)))

        return (np.squeeze(height)*units.meter,
                np.squeeze(velocity)*units.meter/units.second)

    def motion_from_height(self, time, initial_height_range, dq):
        """
        Calculates parcel motion for different initial heights.

        Args:
            time: Array of time points.
            initial_height_range: Array of initial parcel heights.
            dq: Initial change in specific humidity due to evaporation.

        Returns:
            An array of parcel heights and an array of vertical
                velocities, with each row corresponding to a different
                initial specific humidity change and each column
                corresponding to a different point in time.
        """

        def motion_ode(
                time, state, initial_height, specific_humidity_change):
            buoyancy = self.parcel_buoyancy(
                state[0]*units.meter, initial_height,
                specific_humidity_change)
            return [state[1], buoyancy.magnitude]

        initial_height_range = np.atleast_1d(
            initial_height_range).to(units.meter)
        initial_state = np.array([[z0.m, 0] for z0 in initial_height_range])

        height = np.zeros((len(initial_height_range), len(time)))
        velocity = np.zeros((len(initial_height_range), len(time)))
        time = time.to(units.second).m

        for i in range(len(initial_height_range)):
            sol = solve_ivp(
                motion_ode, [np.min(time), np.max(time)],
                initial_state[i,:], t_eval=time,
                args=(initial_height_range[i], dq))
            height[i,:] = sol.y[0,:]
            velocity[i,:] = sol.y[1,:]
            sys.stdout.write(
                '\rCalculating profile {0} of {1}.'
                '   '.format(i+1, initial_state.shape[0]))

        return (np.squeeze(height)*units.meter,
                np.squeeze(velocity)*units.meter/units.second)
