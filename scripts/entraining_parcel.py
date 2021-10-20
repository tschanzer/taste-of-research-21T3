# Class for parcel theory calculations on real atmospheric soundings
# with entrainment.

# Thomas Schanzer, UNSW Sydney
# October 2021

import numpy as np

import metpy.calc as mpcalc
from metpy.units import units
from metpy.units import concatenate

from scipy.interpolate import interp1d

from thermo import descend, equilibrate


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

    def _entrain_discrete(self, height, state, rate, dz):
        """
        Finds parcel properties after descent/entrainment.

        Only valid for small steps.

        Args:
            height: Initial height.
            state: 3-tuple of initial temperature, specific humidity
                and liquid ratio.
            rate: Entrainment rate.
            dz: Size of *downward* step, i.e. initial minus final height.

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
            p_final, t_eq, q_eq, l_eq, p_initial, improve=3)

        return (t_final, q_final, l_final)

    def profile(
            self, height, t_initial, q_initial, l_initial, rate,
            dz=50*units.meter, reference_height=None):
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

        Returns:
            3-tuple containing the temperature, specific humidity and
                liquid ratio arrays for the given height array.
        """

        height = np.atleast_1d(height).m_as(units.meter)
        dz = dz.m_as(units.meter)
        if reference_height is not None:
            reference_height = reference_height.m_as(units.meter)
            if height.size == 1 and height.item() == reference_height:
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
                all_heights[i] - all_heights[i+1])
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
