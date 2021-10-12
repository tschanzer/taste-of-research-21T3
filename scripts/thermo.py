# Thermodynamic calculations for parcel theory
# Thomas Schanzer, UNSW Sydney
# October 2021

import numpy as np

import metpy.calc as mpcalc
import metpy.constants as const
from metpy.units import units
from metpy.units import concatenate

from scipy.optimize import root_scalar

def moist_lapse(pressure, initial_temperature, reference_pressure=None):
    """
    Equivalent to metpy.calc.moist_lapse, circumventing bugs
    """

    pressure = np.atleast_1d(pressure)
    if reference_pressure is None:
        reference_pressure = pressure[0]
    if pressure.size == 1:
        temperature = mpcalc.moist_lapse(
            pressure, initial_temperature,
            reference_pressure=reference_pressure)
        return np.atleast_1d(temperature)[0]
    else:
        pressure = pressure[1:]
        temperature = mpcalc.moist_lapse(
            pressure, initial_temperature,
            reference_pressure=reference_pressure)
        return concatenate([initial_temperature, temperature])


def theta_w(theta_e):
    """
    Calculates theta-w from theta-e using Eq. 3.8 of Davies-Jones 2008.

    Args:
        theta_e: Equivalent potential temperature.

    Returns:
        Wet bulb potential temperature.
    """

    theta_e = theta_e.m_as(units.kelvin)

    C=273.15
    X = theta_e/C

    # coefficients
    a0 = 7.101574
    a1 = -20.68208
    a2 = 16.11182
    a3 = 2.574631
    a4 = -5.205688
    b1 = -3.552497
    b2 = 3.781782
    b3 = -0.6899655
    b4 = -0.5929340

    theta_w = (
        theta_e - C
        - np.exp((a0 + a1*X + a2*X**2 + a3*X**3 + a4*X**4)
              /(1 + b1*X + b2*X**2 + b3*X**3 + b4*X**4))*(theta_e >= 173.15)
    )

    return theta_w*units.celsius


def _daviesjones_f(Tw, pi):
    """
    Evaluates the function f defined in eq. 2.3 of Davies-Jones 2008.

    Args:
        Tw: Wet-bulb temperature in KELVIN.
        pi: Nondimensional pressure.

    Returns:
        The value of f(Tw, pi).
    """

    pressure = 1000.0 * pi**3.504  # in mbar

    # coefficients
    k0 = 3036
    k1 = 1.78
    k2 = 0.448
    nu = 0.2854  # poisson constant for dry air
    C = 273.15

    # saturation mixing ratio and vapour pressure calculated using
    # eq. 10 of Bolton 1980
    rs = mpcalc.saturation_mixing_ratio(
        pressure*units.mbar, Tw*units.kelvin).m_as(units.dimensionless)
    es = mpcalc.saturation_vapor_pressure(Tw*units.kelvin).m_as(units.mbar)

    G = (k0/Tw - k1)*(rs + k2*rs**2)
    f = (C/Tw)**3.504 * (1 - es/pressure)**(3.504*nu) * np.exp(-3.504*G)

    return f


def _daviesjones_fprime(tau, pi):
    """
    Evaluates df/dtau (pi fixed) defined in eqs. A.1-A.5 of Davies-Jones 2008.

    Args:
        tau: Temperature in KELVIN.
        pi: Nondimensional pressure.

    Returns:
        The value of f'(Tau, pi) for fixed pi.
    """

    pressure = 1000.0 * pi**3.504  # in mbar

    # coefficients
    k0 = 3036
    k1 = 1.78
    k2 = 0.448
    nu = 0.2854  # poisson constant for dry air
    C = 273.15
    epsilon = 0.6220

    # saturation mixing ratio and vapour pressure calculated using
    # eq. 10 of Bolton 1980
    rs = mpcalc.saturation_mixing_ratio(
        pressure*units.mbar, tau*units.kelvin).m_as(units.dimensionless)
    es = mpcalc.saturation_vapor_pressure(tau*units.kelvin).m_as(units.mbar)

    des_dtau = es*17.67*243.5/(tau - C + 243.5)**2  # eq. A.5
    drs_dtau = epsilon*pressure/(pressure - es)**2 * des_dtau  # eq. A.4
    dG_dtau = (-k0/tau**2 * (rs + k2*rs**2)
               + (k0/tau - k1)*(1 + 2*k2*rs)*drs_dtau)  # eq. A.3
    dlogf_dtau = -3.504*(1/tau + nu/(pressure - es)*des_dtau
                         + dG_dtau)  # eq. A.2
    df_dtau = _daviesjones_f(tau, pi) * dlogf_dtau  # eq. A.1

    return df_dtau


def wetbulb(pressure, theta_e, improve=False):
    """
    Calculates wet bulb temperature using the method in Davies-Jones 2008.

    Args:
        pressure: Pressure.
        theta_e: Equivalent potential temperature.
        improve: Whether or not to perform a single iteration of
            Newton's method to improve accuracy (defaults to False).

    Returns:
        Wet bulb temperature.
    """

    # changing to correct units
    pressure = pressure.m_as(units.mbar)
    theta_e = theta_e.m_as(units.kelvin)

    pi = (pressure/1000.0)**(1./3.504)
    Teq = theta_e*pi
    C = 273.15
    X = (C/Teq)**3.504

    # slope and intercept for guesses - eq. 4.3, 4.4
    k1 = -38.5*pi**2 + 137.81*pi - 53.737
    k2 = -4.392*pi**2 + 56.831*pi - 0.384

    # transition point between approximation schemes - eq. 4.7
    D = 1/(0.1859*pressure/1000 + 0.6512)

    # initial guess
    if X > D:
        A = 2675.0
        # saturation mixing ratio calculated via vapour pressure using
        # eq. 10 of Bolton 1980
        rs = mpcalc.saturation_mixing_ratio(
            pressure*units.mbar, Teq*units.kelvin).m_as(units.dimensionless)
        # d(log(e_s))/dT calculated also from eq. 10, Bolton 1980
        d_log_es_dt = 17.67*243.5/(Teq + 243.5)**2

        # approximate wet bulb temperature in celsius
        Tw = Teq - C - A*rs/(1 + A*rs*d_log_es_dt)
    elif 1 <= X <= D:
        Tw = k1 - k2*X
    elif 0.4 <= X < 1:
        Tw = (k1 - 1.21) - (k2 - 1.21)*X
    else:
        Tw = (k1 - 2.66) - (k2 - 1.21)*X + 0.58/X

    if improve:
        # execute a single iteration of Newton's method (eq. 2.6)
        slope = _daviesjones_fprime(Tw + C, pi)
        fvalue = _daviesjones_f(Tw + C, pi)
        Tw = Tw - (fvalue - X)/slope

    return Tw*units.celsius


def theta_e(p, Tk, q, prime=False, with_units=False):
    """
    Calculates the partial derivative of theta-e w.r.t. temperature.

    Uses the approximation of theta-e given in eq. 39 of Bolton (1980).

    Args:
        p: Pressure.
        Tk: Temperature.
        q: Specific humidity.

    Returns:
        The partial derivative of equivalent potential temperature
            with respect to temperature.
    """

    # ensure correct units
    p = p.m_as(units.mbar)
    Tk = Tk.m_as(units.kelvin)
    if hasattr(q, 'units'):
        q = q.m_as(units.dimensionless)

    # constants
    a = 17.67  # dimensionless
    b = 243.5  # kelvin
    C = 273.15  # 0C (kelvin)
    e0 = 6.112  # saturation vapour pressure at 0C (mbar)
    epsilon = const.epsilon.m  # molar mass ratio of dry air to water vapour
    kappa = const.kappa.m # poisson constant of dry air

    # other variables
    es = e0*np.exp(a*(Tk - C)/(Tk - C + b))  # sat. vapour pressure in mbar
    U = q/(1 - q)*(p - es)/(epsilon*es)  # relative humidity
    e = U*es # vapour pressure in mbar
    Td = b*np.log(U*es/e0)/(a - np.log(U*es/e0)) + C  # dew point in kelvin
    r = q/(1-q)  # mixing ratio

    # LCL temperature in kelvin
    Tl = (1/(Td - 56) + np.log(Tk/Td)/800)**(-1) + 56
    # LCL potential temperature in kelvin
    thetadl = Tk*(1000/(p - e))**kappa*(Tk/Tl)**(0.28*r)
    # equivalent potential temperature in kelvin
    thetae = thetadl*np.exp((3036/Tl - 1.78)*r*(1 + 0.448*r))

    if prime is False:
        return thetae if with_units is False else thetae*units.kelvin

    # derivative of sat. vapour pressure w.r.t. temperature
    dloges_dTk = a*b/(Tk - C + b)**2
    # derivative of dew point w.r.t. temperature
    dTd_dTk = a*b/(a - np.log(U*es/e0))**2 * dloges_dTk
    # derivative of LCL temperature w.r.t. temperature
    dTl_dTk = (-(1/(Td - 56) + np.log(Tk/Td)/800)**(-2)
               *(-1/(Td - 56)**2*dTd_dTk + 1/800*(1/Tk - 1/Td*dTd_dTk)))
    # derivative of log(LCL potential temperature) w.r.t. temperature
    dlogthetadl_dTk = (1 + 0.28*r)/Tk - 0.28*r/Tl*dTl_dTk
    # derivative of log(equivalent potential temperature) w.r.t. temperature
    dlogthetae_dTk = dlogthetadl_dTk - 3036/Tl**2*r*(1 + 0.448*r)*dTl_dTk

    return (thetae if with_units is False else thetae*units.kelvin,
            thetae*dlogthetae_dTk)


def saturation_specific_humidity(pressure, temperature):
    return mpcalc.specific_humidity_from_mixing_ratio(
        mpcalc.saturation_mixing_ratio(pressure, temperature))


def descend(
        pressure, temperature, specific_humidity, liquid_ratio,
        reference_pressure, improve=1):
    """
    Calculates the temperature of a descending parcel.

    Uses conservation of equivalent potential temperature to determine
    the final temperature if the parcel switches from a moist to a dry
    adiabat.

    Args:
        pressure: Final pressure.
        temperature: Initial temperature.
        specific_humidity: Initial specific humidity.
        liquid_ratio: Initial liquid ratio.
        reference_pressure: Initial pressure.
        improve: Number of iterations to use if the parcel must switch
            from moist to dry adiabat (default: 1). Alternatively,
            specify False to skip iteration and take the moist
            adiabatic value, or 'exact' to iterate until convergence.

    Returns:
        Final temperature, specific humidity and liquid ratio.
    """

    # calculate dry adiabatic value outside if statement since it
    # is needed for the guess in case 2.2
    t_final_dry = mpcalc.dry_lapse(pressure, temperature, reference_pressure)

    if liquid_ratio <= 0:
        # case 1: dry adiabat only
        q_final = q_initial
        l_final = 0*units.dimensionless
        return t_final_dry, q_final, l_final
    else:
        # case 2: some moist descent
        t_final_moist = moist_lapse(pressure, temperature, reference_pressure)
        q_final_moist = saturation_specific_humidity(pressure, t_final_moist)
        l_final_moist = specific_humidity + liquid_ratio - q_final_moist
        if l_final_moist >= 0 or improve is False:
            # case 2.1: moist adiabat only
            return t_final_moist, q_final_moist, l_final_moist
        else:
            # case 2.2: adiabat switching
            # use amount of liquid to place guess between dry and moist values
            t_final_guess = (
                t_final_dry.to(units.kelvin)
                + liquid_ratio/(q_final_moist - specific_humidity).m
                * (t_final_moist.to(units.kelvin)
                   - t_final_dry.to(units.kelvin)))
            q_final = specific_humidity + liquid_ratio
            l_final = 0*units.dimensionless

            theta_e_initial = theta_e(
                reference_pressure, temperature, specific_humidity)
            if improve == 'exact':
                # iterate until convergence using Newton's method
                def root_function(T):
                    value, slope = theta_e(
                        pressure, T*units.kelvin, q_final, prime=True)
                    return value - theta_e_initial, slope
                sol = root_scalar(
                    root_function, x0=t_final_guess.m,
                    fprime=True, method='newton')
                t_final = sol.root*units.kelvin
            else:
                # apply a fixed number of iterations
                t_final = t_final_guess.m
                for i in range(improve):
                    value, slope = theta_e(
                        pressure, t_final*units.kelvin, q_final, prime=True)
                    t_final = t_final - (value - theta_e_initial)/slope
                t_final = t_final*units.kelvin

            return t_final, q_final, l_final


# objective function for the root-finding algorithm
def remaining_liquid_ratio(
        pressure, initial_pressure, initial_temperature,
        initial_liquid_ratio, min_zero=False):
    """
    Calculates the amount of liquid water left in the parcel.

    It is assumed that the parcel is initially saturated.
    The arguments must be given as plain numbers, without units.

    Args:
        pressure: The pressure level of interest in millibars.
        initial_pressure: The initial pressure of the parcel in
            millibars.
        initial_temperature: The initial temperature of the parcel in
            degrees celsius.
        initial_liquid_ratio: The ratio of the initial mass of liquid
            water to the total mass of the parcel.
        min_zero: Whether or not to return a non-negative result
            (defaults to False).

    Returns:
        The ratio of the remaining mass of liquid water to the total
            mass of the parcel.
    """

    if not hasattr(pressure, 'units'):
        pressure = pressure*units.mbar
    if not hasattr(initial_pressure, 'units'):
        initial_pressure = initial_pressure*units.mbar
    if not hasattr(initial_temperature, 'units'):
        initial_temperature = initial_temperature*units.celsius

    initial_specific_humidity = mpcalc.specific_humidity_from_dewpoint(
        initial_pressure, initial_temperature)
    final_temperature = moist_lapse(
        pressure, initial_temperature, reference_pressure=initial_pressure)
    final_specific_humidity = mpcalc.specific_humidity_from_dewpoint(
        pressure, final_temperature)
    remaining_ratio = (initial_specific_humidity + initial_liquid_ratio
                       - final_specific_humidity)
    if min_zero:
        remaining_ratio = np.max([remaining_ratio, 0])

    return remaining_ratio


def evaporation_level(
        initial_pressure, initial_temperature, initial_liquid_ratio):
    """
    Finds the pressure at which all liquid water evaporates.

    Args:
        initial_pressure: The initial pressure of the parcel.
        initial_temperature: The initial temperature of the parcel.
        initial_liquid_ratio: The ratio of the initial mass of liquid
            water to the total mass of the parcel.

    Returns:
        A tuple containing the pressure at which all liquid water
            evaporates, and the temperature of the parcel at this point.
    """

    solution = root_scalar(
        remaining_liquid_ratio,
        args=(initial_pressure, initial_temperature, initial_liquid_ratio),
        bracket=[initial_pressure.to(units.mbar).m, 2000])
    level = solution.root*units.mbar

    level_temperature = moist_lapse(
            level, initial_temperature, reference_pressure=initial_pressure)

    return level, level_temperature


def extra_liquid_descent_profile(
        pressure, initial_temperature, evaporation_level, level_temperature,
        reference_pressure=None):
    """
    Calculates the temperature of a descending air parcel.

    The parcel has some initial liquid water content and is assumed
    to be initially saturated.

    Args:
        pressure: Pressure levels of interest (must be monotonically
            increasing).
        initial_temperature: Initial temperature of the parcel,
            corresponding to pressure[0].
        evaporation_level: The pressure at which all liquid water in
            the parcel will evaporate.
        level_temperature: Parcel temperature at the evaporation level.
        reference_pressure: (optional) The pressure corresponding to
            initial_temperature. Defaults to pressure[0].

    Returns:
        The temperature of the parcel at the given pressure levels
            as an array.
    """

    pressure = np.atleast_1d(pressure)
    if reference_pressure is None:
        reference_pressure = pressure[0]

    # find the position of the evaporation level in the pressure array
    level_index = np.searchsorted(pressure.m, evaporation_level.m, side='left')

    # moist descent to the evaporation level
    if level_index > 0:  # some levels are above EL
        moist_temperature = moist_lapse(
            pressure[:level_index], initial_temperature,
            reference_pressure=reference_pressure)
    else:  # no levels are above EL
        moist_temperature = np.array([])*initial_temperature.units

    # dry descent from the evaporation level
    if level_index != len(pressure):  # some levels are at or below EL
        dry_temperature = mpcalc.dry_lapse(
            pressure[level_index:], level_temperature,
            reference_pressure=evaporation_level)
    else:  # no levels are at or below EL
        dry_temperature = np.array([])*initial_temperature.units

    # join moist and dry descent
    temperature = concatenate([moist_temperature, dry_temperature])
    if temperature.size == 1:
        temperature = temperature.item()

    return temperature


def specific_humidity_from_descent_profile(
        pressure, temperature, evaporation_level, level_temperature):
    """
    Calculates the specific humidity of a descending parcel.

    Args:
        pressure: Pressure levels of interest.
        temperature: Temperature array corresponding to given pressure
            levels.
        evaporation_level: Evaporation level of the parcel.
        level_temperature: Parcel temperature at the evaporation level.

    Returns:
        The specific humidity array corresponding to the specified
        pressure levels.
    """

    pressure = np.atleast_1d(pressure)
    temperature = np.atleast_1d(temperature)
    above_level_mask = pressure <= evaporation_level
    if np.sum(above_level_mask) > 0:
        # dew point above the evaporation level is the temperature
        q_above_level = mpcalc.specific_humidity_from_dewpoint(
                pressure[above_level_mask], temperature[above_level_mask])
    else:
        q_above_level = np.array([])

    # specific humidity below the evaporation level is the specific
    # humidity at the evaporation level
    q_below_level = (
        np.ones(np.sum(~above_level_mask))
        * mpcalc.specific_humidity_from_dewpoint(
             evaporation_level, level_temperature)
    )

    specific_humidity = concatenate([q_above_level, q_below_level])
    if specific_humidity.size == 1:
        specific_humidity = specific_humidity.item()

    return specific_humidity
