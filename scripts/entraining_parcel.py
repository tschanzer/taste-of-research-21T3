# Class for parcel theory calculations on real atmospheric soundings
# with entrainment.

# Thomas Schanzer, UNSW Sydney
# October 2021

import numpy as np

import metpy.calc as mpcalc
from metpy.units import units
from metpy.units import concatenate
import metpy.constants as const

from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

from thermo import descend, equilibrate


class MotionResult:
    pass


class EntrainingParcel:
    """
    Class for parcel theory calculations with entrainment.
    """

    def __init__(self, environment):
        """
        Instantiates an EntrainingParcel.

        Args:
            environment: An instance of Environment on which the
                calculations are to be performed.
        """

        self._env = environment

    def _entrain_discrete(self, height, state, rate, dz, kind='pseudo'):
        """
        Finds parcel properties after descent/entrainment.

        Only valid for small steps.

        Args:
            height: Initial height.
            state: 3-tuple of initial temperature, specific humidity
                and liquid ratio.
            rate: Entrainment rate.
            dz: Size of *downward* step, i.e. initial minus final height.
            kind: 'pseudo' for pseudoadiabats, 'reversible' for reversible
                adiabats.

        Returns:
            3-tuple of final temperature, specific humidity and liquid ratio.
        """

        height = height
        t_parcel = state[0]
        q_parcel = state[1]
        l_parcel = state[2]
        p_initial = self._env.pressure(height)
        p_final = self._env.pressure(height - dz)

        # steps 1 and 2: mixing and phase equilibration
        t_eq, q_eq, l_eq = equilibrate(
            p_initial, t_parcel, q_parcel, l_parcel,
            self._env.temperature(height), self._env.specific_humidity(height),
            self._env.liquid_ratio(height), rate, dz)

        # step 3: dry or moist adiabatic descent
        t_final, q_final, l_final = descend(
            p_final, t_eq, q_eq, l_eq, p_initial, kind=kind)

        return (t_final, q_final, l_final)

    def profile(
            self, height, t_initial, q_initial, l_initial, rate,
            dz=50*units.meter, reference_height=None, kind='pseudo'):
        """
        Calculates parcel properties for descent with entrainment.

        Valid for arbitrary steps.

        Args:
            height: Array of heights of interest.
            t_initial: Initial parcel temperature.
            q_initial: Initial parcel specific humidity.
            l_initial: Initial parcel liquid ratio.
            rate: Entrainment rate.
            dz: Size of *downward* step for computing finite differences.
            kind: 'pseudo' for pseudoadiabats, 'reversible' for reversible
                adiabats.

        Returns:
            3-tuple containing the temperature, specific humidity and
                liquid ratio arrays for the given height array.
        """

        height = np.atleast_1d(height).m_as(units.meter)
        dz = dz.m_as(units.meter)
        if reference_height is not None:
            reference_height = reference_height.m_as(units.meter)
            if height.size == 1 and height.item() >= reference_height:
                # no descent needed, return initial values
                return t_initial, q_initial, l_initial

        # create height array with correct spacing
        if reference_height is None or reference_height == height[0]:
            all_heights = np.arange(height[0], height[-1], -dz)
            all_heights = np.append(all_heights, height[-1])*units.meter
        else:
            all_heights = np.arange(reference_height, height[-1], -dz)
            all_heights = np.append(all_heights, height[-1])*units.meter

        # calculate t, q and l one downward step at a time
        sol_states = [(t_initial, q_initial, l_initial)]
        for i in range(all_heights.size - 1):
            next_state = self._entrain_discrete(
                all_heights[i], sol_states[i], rate,
                all_heights[i] - all_heights[i+1], kind=kind)
            sol_states.append(next_state)

        t_sol = concatenate(
            [state[0] for state in sol_states]).m_as(units.celsius)
        q_sol = concatenate([state[1] for state in sol_states]).m
        l_sol = concatenate([state[2] for state in sol_states]).m

        # find the values of t, q and l at the originally specified heights
        t_interp = interp1d(all_heights.m, t_sol)
        t_out = t_interp(height)*units.celsius
        q_interp = interp1d(all_heights.m, q_sol)
        q_out = q_interp(height)*units.dimensionless
        l_interp = interp1d(all_heights.m, l_sol)
        l_out = l_interp(height)*units.dimensionless

        # return scalars if only one height was given
        if height.size == 1:
            t_out = t_out.item()
            q_out = q_out.item()
            l_out = l_out.item()

        return t_out, q_out, l_out
    
    def density(
            self, height, initial_height, t_initial, q_initial, l_initial,
            rate, step=50*units.meter, kind='pseudo', liquid_correction=True):
        """
        Calculates parcel density as a function of height.

        Args:
            height: Height of the parcel.
            initial_height: Initial height.
            t_initial: Initial temperature.
            q_initial: Initial specific humidity.
            l_initial: Initial liquid ratio.
            rate: Entrainment rate.
            step: Step size for entrainment calculation.
            kind: 'pseudo' for pseudoadiabats, 'reversible' for reversible
                adiabats.
            liquid_correction: Whether or not to account for the mass
                of liquid water.

        Returns:
            The density of the parcel at <height>.
        """

        t_final, q_final, l_final = self.profile(
            height, t_initial, q_initial, l_initial, rate, dz=step,
            reference_height=initial_height, kind=kind)
        r_final = mpcalc.mixing_ratio_from_specific_humidity(q_final)
        p_final = self._env.pressure(height)

        gas_density = mpcalc.density(p_final, t_final, r_final)
        return gas_density/(1 - l_final.m*liquid_correction)
    
    def buoyancy(
            self, height, initial_height, t_initial, q_initial, l_initial,
            rate, step=50*units.meter, kind='pseudo', liquid_correction=True):
        """
        Calculates parcel buoyancy as a function of height.

        Args:
            height: Height of the parcel.
            initial_height: Initial height.
            t_initial: Initial temperature.
            q_initial: Initial specific humidity.
            l_initial: Initial liquid ratio.
            rate: Entrainment rate.
            step: Step size for entrainment calculation.
            kind: 'pseudo' for pseudoadiabats, 'reversible' for reversible
                adiabats.
            liquid_correction: Whether or not to account for the mass
                of liquid water.

        Returns:
            The buoyancy of the parcel at <height>.
        """

        env_density = self._env.density(height)
        pcl_density = self.density(
            height, initial_height, t_initial, q_initial, l_initial, rate,
            step, kind=kind, liquid_correction=liquid_correction)

        return (env_density - pcl_density)/pcl_density*const.g
    
    def motion(
            self, time, initial_height, initial_velocity, t_initial,
            q_initial, l_initial, rate, step=50*units.meter,
            kind='pseudo', liquid_correction=True):
        """
        Solves the equation of motion for the parcel.

        Args:
            time: Array of times for which the results will be reported.
            initial_height: Initial height.
            initial_velocity: Initial vertical velocity.
            t_initial: Initial temperature.
            q_initial: Initial specific humidity.
            l_initial: Initial liquid ratio.
            rate: Entrainment rate.
            step: Step size for entrainment calculation.
            kind: 'pseudo' for pseudoadiabats, 'reversible' for reversible
                adiabats.
            liquid_correction: Whether or not to account for the mass
                of liquid water.

        Returns:
            An instance of MotionResult.
        """

        def motion_ode(time, state, *args):
            height = np.max([state[0], 0])*units.meter
            b = self.buoyancy(
                height, *args, kind=kind, liquid_correction=liquid_correction)
            return [state[1], b.m]

        initial_height = initial_height.m_as(units.meter)
        initial_velocity = initial_velocity.m_as(units.meter/units.second)
        time = time.to(units.second).m

        # event function for solve_ivp, zero when parcel reaches min height
        min_height = lambda time, state, *args: state[1]
        min_height.direction = 1  # find zero that goes from - to +
        min_height.terminal = True  # stop integration at minimum height

        # event function for solve_ivp, zero when parcel hits ground
        hit_ground = lambda time, state, *args: state[0]
        hit_ground.terminal = True  # stop integration at ground

        # event function for solve_ivp, zero when parcel is neutrally
        # buoyant
        neutral_buoyancy = lambda time, state, *args: motion_ode(
            time, state, *args)[1]

        # prepare empty arrays for data
        height = np.zeros(len(time))
        height[:] = np.nan
        velocity = np.zeros(len(time))
        velocity[:] = np.nan

        sol = solve_ivp(
            motion_ode,
            [np.min(time), np.max(time)],
            [initial_height, initial_velocity],
            t_eval=time,
            args=(
                initial_height*units.meter, t_initial,
                q_initial, l_initial, rate, step),
            events=[neutral_buoyancy, hit_ground, min_height])

        height[:len(sol.y[0,:])] = sol.y[0,:]
        velocity[:len(sol.y[1,:])] = sol.y[1,:]

        # record times of events
        # sol.t_events[i].size == 0 means the event did not occur
        neutral_buoyancy_time = (  # record only the first instance
            sol.t_events[0][0] if sol.t_events[0].size > 0 else np.nan)
        hit_ground_time = (
            sol.t_events[1][0] if sol.t_events[1].size > 0 else np.nan)
        min_height_time = (
            sol.t_events[2][0] if sol.t_events[2].size > 0 else np.nan)

        # record states at event times
        neutral_buoyancy_height = (  # record only the first instance
            sol.y_events[0][0,0] if sol.y_events[0].size > 0 else np.nan)
        neutral_buoyancy_velocity = (  # record only the first instance
            sol.y_events[0][0,1] if sol.y_events[0].size > 0 else np.nan)
        hit_ground_velocity = (
            sol.y_events[1][0,1] if sol.y_events[1].size > 0 else np.nan)
        min_height_height = (
            sol.y_events[2][0,0] if sol.y_events[2].size > 0 else np.nan)

        result = MotionResult()
        result.height = height*units.meter
        result.velocity = velocity*units.meter/units.second
        result.neutral_buoyancy_time = neutral_buoyancy_time*units.second
        result.hit_ground_time = hit_ground_time*units.second
        result.min_height_time = min_height_time*units.second
        result.neutral_buoyancy_height = neutral_buoyancy_height*units.meter
        result.neutral_buoyancy_velocity = (
            neutral_buoyancy_velocity*units.meter/units.second)
        result.hit_ground_velocity = (
            hit_ground_velocity*units.meter/units.second)
        result.min_height = min_height_height*units.meter

        return result
