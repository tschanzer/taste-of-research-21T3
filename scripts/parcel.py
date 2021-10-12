# Class for parcel theory calculations on real atmospheric soundings
# Thomas Schanzer, UNSW Sydney
# October 2021

import numpy as np

import metpy.calc as mpcalc
import metpy.constants as const
from metpy.units import units

from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

import sys

from thermo import moist_lapse, remaining_liquid_ratio, evaporation_level
from thermo import extra_liquid_descent_profile
from thermo import specific_humidity_from_descent_profile
from thermo import temperature_change


class MotionResult:
    """Class for results of parcel motion calculations."""

    pass


class Parcel:
    """
    Class for parcel theory calculations on real atmospheric soundings.
    """

    def __init__(self, environment):
        """
        Instantiates a Parcel.

        Args:
            environment: An instance of Environment on which the
                calculations are to be performed.
        """

        self._env = environment

    def dry_parcel_density(
            self, height, initial_height, specific_humidity_change):
        """
        Calculates the density of a parcel after precipitation.

        Assumes that the specific humidity is initially increased
        by a known amount via evaporation, with the parcel remaining
        subsaturated at all times.

        Args:
            height: The height of the parcel.
            initial_height: The initial height of the parcel (i.e., its
                height when the precipitation occurred).
            specific_humidity_change: The change in specific humidity
                that resulted from the precipitation.

        Returns:
            The density of the parcel.
        """

        initial_pressure = self._env.pressure(initial_height)
        pressure = self._env.pressure(height)
        initial_temperature = (self._env.temperature(initial_height)
                               + temperature_change(specific_humidity_change))
        specific_humidity = (self._env.specific_humidity(initial_height)
                             + specific_humidity_change)
        mixing_ratio = mpcalc.mixing_ratio_from_specific_humidity(
            specific_humidity)
        temperature = mpcalc.dry_lapse(
            pressure, initial_temperature, reference_pressure=initial_pressure)
        density = mpcalc.density(pressure, temperature, mixing_ratio)

        return density

    def saturated_parcel_density(
            self, height, initial_height, initial_temperature):
        """
        Calculates the density of a parcel after precipitation.

        Assumes that the parcel is initially saturated via evaporation
        and remains saturated at all times.

        Args:
            height: The height of the parcel.
            initial_height: The initial height of the parcel (i.e., its
                height when the precipitation occurred).
            initial_temperature: The initial temperature of the parcel.

        Returns:
            The density of the parcel.
        """

        initial_pressure = self._env.pressure(initial_height)
        pressure = self._env.pressure(height)

        temperature = moist_lapse(
            pressure, initial_temperature, reference_pressure=initial_pressure)

        mixing_ratio = mpcalc.saturation_mixing_ratio(pressure, temperature)
        density = mpcalc.density(pressure, temperature, mixing_ratio)

        return density

    def limited_water_parcel_density(
            self, height, initial_height, initial_temperature,
            evaporation_level, level_temperature, initial_liquid_ratio):
        """
        Calculates the density of a parcel after precipitation.

        Assumes that the parcel is initially saturated via evaporation,
        but retains only a limited amount of liquid water so that it
        may not remain saturated during descent.

        Args:
            height: The height of the parcel.
            initial_height: The initial height of the parcel (i.e., its
                height when the precipitation occurred).
            initial_temperature: The initial temperature of the parcel.
            evaporation_level: The pressure at which the remaining
                liquid water will completely evaporate.
            level_temperature: The parcel's temperature at the
                evaporation level.

        Returns:
            The density of the parcel.
        """

        initial_pressure = self._env.pressure(initial_height)
        pressure = self._env.pressure(height)

        temperature = extra_liquid_descent_profile(
            pressure, initial_temperature, evaporation_level,
            level_temperature, reference_pressure=initial_pressure)

        specific_humidity = specific_humidity_from_descent_profile(
            pressure, temperature, evaporation_level, level_temperature)
        mixing_ratio = mpcalc.mixing_ratio_from_specific_humidity(
            specific_humidity)

        gas_density = mpcalc.density(pressure, temperature, mixing_ratio)
        liquid_ratio = remaining_liquid_ratio(
                pressure, initial_pressure, initial_temperature,
                initial_liquid_ratio, min_zero=True)

        density = gas_density / (1 - liquid_ratio)  # liquid correction
        return density

    def parcel_buoyancy(self, height, *args, regime='dry'):
        """
        Calculates the buoyancy of a parcel after precipitation.

        Args:
            height: The height of the parcel.
            regime: The buoyancy regime for the calculation: 'dry',
                'saturated' or 'limited'. Default is 'dry'.
            The remaining positional arguments are the same as those for
                the density function corresponding to the regime.

        Returns:
            The buoyant force per unit mass on the parcel.
        """

        if regime not in ['dry', 'saturated', 'limited']:
            raise ValueError(
                "regime must be 'dry', 'saturated' or 'limited'.")

        environment_density = self._env.density(height)

        density_function = {
            'dry': self.dry_parcel_density,
            'saturated': self.saturated_parcel_density,
            'limited': self.limited_water_parcel_density}[regime]
        density = density_function(height, *args)

        buoyancy = (environment_density - density) / density * const.g
        return buoyancy

    def dry_neutral_buoyancy_level(
            self, initial_height, specific_humidity_change):
        """
        Calculates the neutral buoyancy heights of parcels.

        Assumes the dry buoyancy regime.

        Args:
            initial_height: An array of initial heights.
            specific_humidity_change: An array of initial specific
                humidity changes due to evaporation.

        Returns:
            An array of neutral buoyancy heights, with each row
                corresponding to one initial height and each column
                corresponding to one initial specific humidity change.
        """

        initial_height = np.atleast_1d(initial_height)
        specific_humidity_change = np.atleast_1d(specific_humidity_change)
        sol = np.zeros((len(initial_height), len(specific_humidity_change)))

        root_function = lambda height, *args: self.parcel_buoyancy(
            height*units.meter, *args, regime='dry').m

        for i, z0 in enumerate(initial_height):
            for ii, dq in enumerate(specific_humidity_change):
                sys.stdout.write(
                        '\rCalculating buoyancy level '
                        '{} of {}.       '.format(
                            i*len(specific_humidity_change) + ii + 1,
                            len(specific_humidity_change)*len(initial_height)))
                if dq <= self._env.maximum_specific_humidity_change(z0):
                    try:
                        sol[i,ii] = root_scalar(
                            root_function,
                            args=(z0, dq),
                            x0=z0.to(units.meter).m, x1=0,
                            bracket=[0, z0.to(units.meter).m]
                        ).root
                    except ValueError:
                        sol[i,ii] = 0
                else:
                    sol[i,ii] = np.nan

        sol = np.squeeze(sol)
        if sol.size == 1:
            sol = sol.item()

        return sol/1e3*units.km

    def saturated_neutral_buoyancy_level(self, initial_height):
        """
        Calculates the neutral buoyancy height of a parcel.

        Assumes the saturated buoyancy regime.

        Args:
            initial_height: The initial height of the parcel.

        Returns:
            The height at which the buoyancy of the parcel is zero.
        """

        initial_height = np.atleast_1d(initial_height)
        initial_temperature = self._env.wetbulb_temperature(initial_height)
        sol = np.zeros(len(initial_height))

        root_function = lambda height, *args: self.parcel_buoyancy(
            height*units.meter, *args, regime='saturated').m

        for i, z0 in enumerate(initial_height):
            sys.stdout.write(
                '\rCalculating buoyancy level '
                '{} of {}.       '.format(i+1, len(initial_height)))
            try:
                sol[i] = root_scalar(
                    root_function,
                    args=(z0, initial_temperature[i]),
                    x0=z0.to(units.meter).m, x1=0,
                    bracket=[0, z0.to(units.meter).m]
                ).root
            except ValueError:
                sol[i] = 0

        if sol.size == 1:
            sol = sol.item()

        return sol/1e3*units.km

    def limited_neutral_buoyancy_level(self, initial_height, liquid_ratio):
        """
        Calculates the neutral buoyancy height of a parcel.

        Assumes the limited buoyancy regime.

        Args:
            initial_height: An array of initial heights.
            liquid_ratio: An array of initial liquid water amounts,
                as fractions of total parcel mass.

        Returns:
            An array of neutral buoyancy heights, with each row
                corresponding to one initial height and each column
                corresponding to one initial liquid water ratio.
        """

        initial_height = np.atleast_1d(initial_height)
        initial_pressure = self._env.pressure(initial_height)
        initial_temperature = self._env.wetbulb_temperature(initial_height)
        liquid_ratio = np.atleast_1d(liquid_ratio)
        sol = np.zeros((len(initial_height), len(liquid_ratio)))

        root_function = lambda height, *args: self.parcel_buoyancy(
            height*units.meter, *args, regime='limited').m

        for i, z0 in enumerate(initial_height):
            for ii, lr in enumerate(liquid_ratio):
                sys.stdout.write(
                        '\rCalculating buoyancy level '
                        '{} of {}.       '.format(
                            i*len(liquid_ratio) + ii + 1,
                            len(liquid_ratio)*len(initial_height)))
                level, level_temperature = evaporation_level(
                    initial_pressure[i], initial_temperature[i], lr)
                try:
                    sol[i,ii] = root_scalar(
                        root_function,
                        args=(z0, initial_temperature[i], level,
                              level_temperature, lr),
                        x0=z0.to(units.meter).m, x1=0,
                        bracket=[0, z0.to(units.meter).m]
                    ).root
                except ValueError:
                    sol[i,ii] = 0

        sol = np.squeeze(sol)
        if sol.size == 1:
            sol = sol.item()

        return sol/1e3*units.km

    def modified_motion(
        self, time, initial_height, initial_velocity, *args, regime='dry'):
        """
        Calculates parcel motion.

        Args:
            time: Array of time points for the solution.
            initial_height: Array of initial heights.
            *args:
                Initial specific humidity change array if regime is
                    'dry',
                Nothing if regime is 'saturated',
                Initial liquid water ratio if regime is 'limited'.
            regime: The buoyancy regime to use: 'dry', 'saturated' or
                'limited'.

        Returns:
            A MotionResult object.
        """

        # independent variables
        initial_height = np.atleast_1d(initial_height).to(units.meter).m
        length = len(initial_height)
        initial_velocity = np.atleast_1d(
            initial_velocity).to(units.meter/units.second).m
        if length != len(initial_velocity):
            raise ValueError(
                    'Initial height and velocity arrays must have the same '
                    'length.')
        initial_state = [
            list(x) for x in zip(initial_height, initial_velocity)]
        time = time.to(units.second).m

        # functions to calculate the extra arguments to pass to motion_ode
        if regime == 'dry':
            dq = np.atleast_1d(args[0])
            if len(dq) != length:
                raise ValueError(
                    'Initial height and specific humidity change arrays must '
                    'have the same length.')
            ode_args = lambda i: (initial_height[i]*units.meter, dq[i])
        elif regime == 'saturated':
            ode_args = lambda i: (
                initial_height[i]*units.meter,
                self._env.wetbulb_temperature(initial_height[i]*units.meter),
            )
        elif regime == 'limited':
            liquid_ratio = np.atleast_1d(args[0])
            if len(liquid_ratio) != length:
                raise ValueError(
                    'Initial height and liquid ratio arrays must '
                    'have the same length.')
            def ode_args(i):
                height = initial_height[i]*units.meter
                initial_pressure = self._env.pressure(height)
                initial_temperature = self._env.wetbulb_temperature(height)
                level, level_temperature = evaporation_level(
                    initial_pressure, initial_temperature, liquid_ratio[i])
                return (
                    height, initial_temperature,
                    level, level_temperature, liquid_ratio[i],
                )
        else:
            raise ValueError(
                "regime must be 'dry', 'saturated' or 'limited'.")

        def motion_ode(time, state, *args):
            """
            Defines the equation of motion for a parcel.
            """

            # for some reason solve_ivp likes to test large negative
            # heights which can cause overflows
            height = np.max([state[0], 0])*units.meter
            buoyancy = self.parcel_buoyancy(
                height, *args, regime=regime)
            return [state[1], buoyancy.magnitude]

        # event function for solve_ivp, zero when parcel reaches min height
        min_height = lambda time, state, *args: state[1]
        min_height.direction = 1  # find zero that goes from - to +
        min_height.terminal = True  # stop integration at minimum height

        # event function for solve_ivp, zero when parcel hits ground
        hit_ground = lambda time, state, *args: state[0]
        hit_ground.terminal = True  # stop integration at ground

        # prepare empty arrays for data
        height = np.zeros((length, len(time)))
        height[:] = np.nan
        velocity = np.zeros((length, len(time)))
        velocity[:] = np.nan

        neutral_buoyancy_time = np.zeros(length)
        hit_ground_time = np.zeros(length)
        min_height_time = np.zeros(length)

        neutral_buoyancy_height = np.zeros(length)
        neutral_buoyancy_velocity = np.zeros(length)
        hit_ground_velocity = np.zeros(length)
        min_height_height = np.zeros(length)

        for i in range(length):
            sys.stdout.write(
                '\rCalculating profile {0} of {1}.'
                '   '.format(i+1, length))

            # event function for solve_ivp, zero when parcel is neutrally
            # buoyant
            neutral_buoyancy = lambda time, state, *args: motion_ode(
                time, state, *args)[1]

            sol = solve_ivp(
                motion_ode,
                [np.min(time), np.max(time)],
                initial_state[i],
                t_eval=time,
                args=ode_args(i),
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

        result = MotionResult()
        result.height = np.squeeze(height)*units.meter
        result.velocity = np.squeeze(velocity)*units.meter/units.second
        result.neutral_buoyancy_time = (
            np.squeeze(neutral_buoyancy_time)*units.second)
        result.hit_ground_time = np.squeeze(hit_ground_time)*units.second
        result.min_height_time = np.squeeze(min_height_time)*units.second
        result.neutral_buoyancy_height = (
            np.squeeze(neutral_buoyancy_height)*units.meter)
        result.neutral_buoyancy_velocity = (
            np.squeeze(neutral_buoyancy_velocity)*units.meter/units.second)
        result.hit_ground_velocity = (
            np.squeeze(hit_ground_velocity)*units.meter/units.second)
        result.min_height = np.squeeze(min_height_height)*units.meter

        return result
