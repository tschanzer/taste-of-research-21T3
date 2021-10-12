# Class for atmospheric sounding data
# Thomas Schanzer, UNSW Sydney
# September 2021

import numpy as np

import metpy.calc as mpcalc
from metpy.units import units
from metpy.units import concatenate

from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

from thermo import theta_e, wetbulb, temperature_change
from thermo import saturation_specific_humidity


class Environment:
    """
    Class for atmospheric sounding data.
    """

    def __init__(
            self, pressure, height, temperature, dewpoint, liquid_ratio=None,
            info='', name=''):
        """
        Instantiates an Environment.

        Args:
            pressure: Pressure array in the sounding.
            height: Height array in the sounding.
            temperature: Temperature array in the sounding.
            dewpoint: Dewpoint array in the sounding.
            liquid_ratio: Array of liquid water partial density to
                total density in the sounding (optional, defaults to
                all zero).
            info: Information to store with the sounding, e.g. date
                (optional)
            name: Short name for the sounding, e.g. 'Sydney' (optional).
        """

        self._pressure_raw = pressure.to(units.mbar)
        self._height_raw = height.to(units.meter)
        self._height_raw -= np.min(self._height_raw)  # set z=0 at surface
        self._temperature_raw = temperature.to(units.celsius)
        self._dewpoint_raw = dewpoint.to(units.celsius)

        if liquid_ratio is None:
            self._liquid_ratio_raw = (
                np.zeros(self._pressure_raw.size)*units.dimensionless)
        else:
            if hasattr(liquid_ratio, 'units'):
                self._liquid_ratio_raw = liquid_ratio.to(units.dimensionless)
            else:
                self._liquid_ratio_raw = liquid_ratio*units.dimensionless

        self.info = info
        self.name = name

        self._p_to_T_interp = interp1d(
            self._pressure_raw.m, self._temperature_raw.m,
            fill_value='extrapolate')
        self._z_to_T_interp = interp1d(
            self._height_raw.m, self._temperature_raw.m,
            fill_value='extrapolate')

        self._p_to_Td_interp = interp1d(
            self._pressure_raw.m, self._dewpoint_raw.m,
            fill_value='extrapolate')
        self._z_to_Td_interp = interp1d(
            self._height_raw.m, self._dewpoint_raw.m,
            fill_value='extrapolate')

        self._p_to_l_interp = interp1d(
            self._pressure_raw.m, self._liquid_ratio_raw.m,
            fill_value='extrapolate')
        self._z_to_l_interp = interp1d(
            self._height_raw.m, self._liquid_ratio_raw.m,
            fill_value='extrapolate')

        self._p_to_z_interp = interp1d(
            self._pressure_raw.m, self._height_raw.m,
            fill_value='extrapolate')
        self._z_to_p_interp = interp1d(
            self._height_raw.m, self._pressure_raw.m,
            fill_value='extrapolate')

    def temperature_from_pressure(self, pressure):
        """
        Finds the environmental temperature at a given pressure.
        """

        temperature = self._p_to_T_interp(pressure.m_as(units.mbar))
        if temperature.size == 1:
            temperature = temperature.item()
        return temperature*units.celsius

    def dewpoint_from_pressure(self, pressure):
        """
        Finds the environmental dew point at a given pressure.
        """

        dewpoint = self._p_to_Td_interp(pressure.m_as(units.mbar))
        if dewpoint.size == 1:
            dewpoint = dewpoint.item()
        return dewpoint*units.celsius

    def liquid_ratio_from_pressure(self, pressure):
        """
        Finds the environmental dew point at a given pressure.
        """

        l = self._p_to_l_interp(pressure.m_as(units.mbar))
        if l.size == 1:
            l = l.item()
        return l*units.dimensionless

    def pressure(self, height):
        """
        Finds the environmental pressure at a given height.
        """

        pressure = self._z_to_p_interp(height.m_as(units.meter))
        if pressure.size == 1:
            pressure = pressure.item()
        return pressure*units.mbar

    def height(self, pressure):
        """
        Finds the height at a given environmental pressure.
        """

        height = self._p_to_z_interp(pressure.m_as(units.mbar))
        if height.size == 1:
            height = height.item()
        return height*units.meter

    def temperature(self, height):
        """
        Finds the environmental temperature at a given height.
        """

        temperature = self._z_to_T_interp(height.m_as(units.meter))
        if temperature.size == 1:
            temperature = temperature.item()
        return temperature*units.celsius

    def dewpoint(self, height):
        """
        Finds the environmental dew point at a given height.
        """

        dewpoint = self._z_to_Td_interp(height.m_as(units.meter))
        if dewpoint.size == 1:
            dewpoint = dewpoint.item()
        return dewpoint*units.celsius

    def liquid_ratio(self, height):
        """
        Finds the environmental dew point at a given height.
        """

        l = self._z_to_l_interp(height.m_as(units.meter))
        if l.size == 1:
            l = l.item()
        return l*units.dimensionless

    def wetbulb_temperature(self, height):
        """
        Finds the environmental wet-bulb temperature at a given height.
        """

        pressure = self.pressure(height)
        temperature = self.temperature(height)
        dewpoint = self.dewpoint(height)
        specific_humidity = mpcalc.specific_humidity_from_dewpoint(
            pressure, dewpoint)
        ept = theta_e(pressure, temperature, specific_humidity)
        return wetbulb(pressure, ept, improve=True)

    def specific_humidity(self, height):
        """
        Finds the environmental specific humidity at a given height.
        """

        pressure = self.pressure(height)
        dewpoint = self.dewpoint(height)
        return mpcalc.specific_humidity_from_dewpoint(pressure, dewpoint)

    def maximum_specific_humidity_change(self, height):
        """
        Calculates the maximum specific humidity increase for the parcel.

        Finds the root of dq_root at the specified height.
        """

        pressure = self.pressure(height)
        initial_temperature = self.temperature(height)
        initial_specific_humidity = self.specific_humidity(height)

        def dq_root(dq):
            """
            Calculates the saturated vs. actual specific humidity difference.
            """

            final_temperature = initial_temperature + temperature_change(dq)
            q_sat = saturation_specific_humidity(pressure, final_temperature)
            return q_sat - initial_specific_humidity - dq

        sol = root_scalar(dq_root, bracket=[0, 20e-3])
        return sol.root*units.dimensionless

    def density(self, height):
        """
        Finds the environmental density at a given height.
        """

        pressure = self.pressure(height)
        temperature = self.temperature(height)
        mixing_ratio = mpcalc.mixing_ratio_from_specific_humidity(
            self.specific_humidity(height))
        return mpcalc.density(pressure, temperature, mixing_ratio)
