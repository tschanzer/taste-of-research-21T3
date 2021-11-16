# Thermodynamic calculations for parcel theory
# Thomas Schanzer, UNSW Sydney
# October 2021

import numpy as np

import metpy.calc as mpcalc
import metpy.constants as const
from metpy.units import units
from metpy.units import concatenate

from scipy.special import lambertw
from scipy.optimize import minimize_scalar, root_scalar
from scipy.integrate import simps


# ---------- Basic thermodynamic calculations ----------

def moist_lapse(
        pressure, initial_temperature, reference_pressure=None,
        method='integration', improve=True):
    """
    Computes temperature from pressure along pseudoadiabats.
    
    Args:
        pressure: Array of pressures for which the temperature is to
            be found.
        initial_temperature: Initial parcel temperature.
        reference_pressure: The pressure corresponding to
            initial_temperature. Optional, defaults to pressure[0].
        method: 'integration' for the MetPy method, 'fast' for
            the Davies-Jones (2008) method.
        improve: Whether or not to apply an iteration of Newton's
            method (only relevant for method == 'fast').
    
    Returns:
        Array of final temperatures.
    """
    
    pressure = np.atleast_1d(pressure)
    if reference_pressure is None:
        reference_pressure = pressure[0]
    if method == 'integration':
        if pressure.size == 1:
            try:
                temperature = mpcalc.moist_lapse(
                    pressure, initial_temperature,
                    reference_pressure=reference_pressure)
                return np.atleast_1d(temperature)[0]
            except:
                return mpcalc.moist_lapse(
                    pressure.item(), initial_temperature.item(),
                    reference_pressure=reference_pressure.item())
        else:
            pressure = pressure[1:]
            temperature = mpcalc.moist_lapse(
                pressure, initial_temperature,
                reference_pressure=reference_pressure)
            return concatenate([initial_temperature, temperature])
    elif method == 'fast':
        q_initial = saturation_specific_humidity(
            reference_pressure, initial_temperature)
        # find initial theta-e (equal to final theta-e)
        thetae_conserved = theta_e(
            reference_pressure, initial_temperature, q_initial)
        # final temperature is equal to final wet bulb temperature
        temperature = concatenate([
            wetbulb(p, thetae_conserved, improve) for p in pressure
        ])
        if temperature.size == 1:
            return temperature.item()
        else:
            return temperature
    else:
        raise ValueError("method must be 'fast' or 'integration'.")

def temperature_change(dq):
    """
    Calculates the temperature change due to evaporation of water.
    """

    dT = (- const.water_heat_vaporization
          * dq / const.dry_air_spec_heat_press)
    return dT.to(units.delta_degC)


def saturation_specific_humidity(pressure, temperature):
    """Calculates saturation specific humidity."""
    
    return mpcalc.specific_humidity_from_mixing_ratio(
        mpcalc.saturation_mixing_ratio(pressure, temperature))


def theta_e(p, Tk, q, prime=False, with_units=True):
    """
    Calculates equivalent potential temperature.

    Uses the approximation of theta-e given in eq. 39 of Bolton (1980).

    Args:
        p: Pressure.
        Tk: Temperature.
        q: Specific humidity.
        prime: Whether or not to also return the derivative of
            theta-e with respect to temperature at the given temperature
            and pressure (optional, defaults to False).
        with_units: Whether or not to return a result with units
            (optional, defaults to True).

    Returns:
        The equivalent potential temperature (and its derivative
        w.r.t. temperature if prime=True).
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


def dcape_dcin(sounding, samples=10000):
    """
    Computes DCAPE and DCIN for a sounding according to Market et. al. (2017).
    
    Args:
        sounding: An Environment object.
        samples: Number of samples to use for integration (optional).
    
    Returns:
        DCAPE and DCIN for the sounding.
    """
    
    # find minimum wet bulb temperature in lowest 6 km
    wetbulb = lambda z: sounding.wetbulb_temperature(
        z*units.meter).m_as(units.kelvin)
    sol = minimize_scalar(wetbulb, bounds=(0, 6000), method='bounded')
    z_initial = sol.x
    p_initial = sounding.pressure(z_initial*units.meter)
    t_initial = sol.fun*units.kelvin
    
    def integrand(z_final):
        z_final = z_final*units.meter
        p_final = sounding.pressure(z_final)
        t_final = moist_lapse(
            p_final, t_initial, p_initial, method='integration')
        w_final = mpcalc.saturation_mixing_ratio(p_final, t_final)
        tv_final = mpcalc.virtual_temperature(t_final, w_final)
        
        t_env = sounding.temperature(z_final)
        w_env = mpcalc.mixing_ratio_from_specific_humidity(
            sounding.specific_humidity(z_final))
        tv_env = mpcalc.virtual_temperature(t_env, w_env)
        
        result = 1 - tv_final.m_as(units.kelvin)/tv_env.m_as(units.kelvin)
        return result
    
    # DCAPE: integrate from neutral buoyancy level to min. wetbulb level
    z_dcape = np.linspace(0, z_initial, samples)
    dcape = simps(
        np.maximum(integrand(z_dcape), 0), z_dcape)*units.meter*const.g
    # DCIN: integrate from surface to neutral buoyancy level
    z_dcin = np.linspace(0, z_initial, samples)
    dcin = simps(
        np.minimum(integrand(z_dcin), 0), z_dcin)*units.meter*const.g
    
    return dcape, dcin


def lcl_romps(p, T, q):
    """
    Analytic solution for the LCL (adapted from Romps 2017).
    
    This code is adapted from Romps (2017):
    https://romps.berkeley.edu/papers/pubdata/2016/lcl/lcl.py
    
    Args:
        p: Pressure.
        T: Temperature.
        q: Specific humidity.
        
    Returns:
        (pressure, temperature) at the LCL.
    """
    
    # unit conversions
    rhl = mpcalc.relative_humidity_from_specific_humidity(p, T, q).m
    p = p.m_as(units.pascal)
    T = T.m_as(units.kelvin)
    
    # Parameters
    Ttrip = 273.16     # K
    ptrip = 611.65     # Pa
    E0v   = 2.3740e6   # J/kg
    ggr   = 9.81       # m/s^2
    rgasa = 287.04     # J/kg/K 
    rgasv = 461        # J/kg/K 
    cva   = 719        # J/kg/K
    cvv   = 1418       # J/kg/K 
    cvl   = 4119       # J/kg/K 
    cvs   = 1861       # J/kg/K 
    cpa   = cva + rgasa
    cpv   = cvv + rgasv

    # The saturation vapor pressure over liquid water
    def pvstarl(T):
        return (ptrip * (T/Ttrip)**((cpv-cvl)/rgasv)
            * np.exp((E0v - (cvv-cvl)*Ttrip) / rgasv * (1/Ttrip - 1/T)))
   
    # Calculate pv from rh, rhl, or rhs
    pv = rhl * pvstarl(T)

    # Calculate lcl_liquid and lcl_solid
    qv = rgasa*pv / (rgasv*p + (rgasa-rgasv)*pv)
    rgasm = (1-qv)*rgasa + qv*rgasv
    cpm = (1-qv)*cpa + qv*cpv
    aL = -(cpv-cvl)/rgasv + cpm/rgasm
    bL = -(E0v-(cvv-cvl)*Ttrip)/(rgasv*T)
    cL = rhl*np.exp(bL)
    T_lcl = bL/(aL*lambertw(bL/aL*cL**(1/aL),-1).real)*T
    p_lcl = p*(T_lcl/T)**(cpm/rgasm)

    return p_lcl/1e2*units.mbar, T_lcl*units.kelvin


def wetbulb_romps(pressure, temperature, specific_humidity):
    """
    Calculates wet bulb temperature using Normand's rule and Romps (2017).
    
    Args:
        p: Pressure.
        T: Temperature.
        q: Specific humidity.
        
    Returns:
        Wet bulb temperature.
    """
    
    lcl_pressure, lcl_temperature = lcl_romps(
        pressure, temperature, specific_humidity)
    return moist_lapse(pressure, lcl_temperature, lcl_pressure)


# ---------- Calculations from Davies-Jones 2008 ----------

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


def _daviesjones_f(Tw, pi, Q=None, kind='pseudo'):
    """
    Evaluates the function f defined in eq. 2.3 of Davies-Jones 2008.

    Args:
        Tw: Wet-bulb temperature in KELVIN.
        pi: Nondimensional pressure.
        Q: total mixing ratio of all phases of water (only needed for
            reversible adiabats).
        kind: 'pseudo' for pseudoadiabats and 'reversible' for
            reversible adiabats.

    Returns:
        The value of f(Tw, pi).
    """

    pressure = 1000.0 * pi**3.504  # in mbar

    # coefficients
    C = 273.15
    if kind == 'pseudo':
        k0 = 3036
        k1 = 1.78
        k2 = 0.448
        nu = 0.2854  # poisson constant for dry air
    elif kind == 'reversible':
        if Q is None:
            raise ValueError(
                'Total water mixing ratio Q must be supplied for '
                'reversible adiabats.')
        L0 = 2.501e6
        L1 = 2.37e3
        cpd = const.dry_air_spec_heat_press.m
        cw = 4190.  # specific heat of liquid water
        k0 = (L0 + L1*C)/(cpd + cw*Q)
        k1 = L1/(cpd + cw*Q)
        k2 = 0
        nu = const.dry_air_gas_constant.m/(cpd + cw*Q)
    else:
        raise ValueError("kind must be 'pseudo' or 'reversible'.")
        
    # saturation mixing ratio and vapour pressure calculated using
    # eq. 10 of Bolton 1980
    rs = mpcalc.saturation_mixing_ratio(
        pressure*units.mbar, Tw*units.kelvin).m_as(units.dimensionless)
    es = mpcalc.saturation_vapor_pressure(Tw*units.kelvin).m_as(units.mbar)

    G = (k0/Tw - k1)*(rs + k2*rs**2)
    f = (C/Tw)**3.504 * (1 - es/pressure)**(3.504*nu) * np.exp(-3.504*G)

    return f


def _daviesjones_fprime(tau, pi, Q=None, kind='pseudo'):
    """
    Evaluates df/dtau (pi fixed) defined in eqs. A.1-A.5 of Davies-Jones 2008.

    Args:
        tau: Temperature in KELVIN.
        pi: Nondimensional pressure.
        Q: total mixing ratio of all phases of water (only needed for
            reversible adiabats).
        kind: 'pseudo' for pseudoadiabats and 'reversible' for
            reversible adiabats.

    Returns:
        The value of f'(Tau, pi) for fixed pi.
    """

    pressure = 1000.0 * pi**3.504  # in mbar

    # coefficients
    C = 273.15
    epsilon = 0.6220
    if kind == 'pseudo':
        k0 = 3036
        k1 = 1.78
        k2 = 0.448
        nu = 0.2854  # poisson constant for dry air
    elif kind == 'reversible':
        if Q is None:
            raise ValueError(
                'Total water mixing ratio Q must be supplied for '
                'reversible adiabats.')
        L0 = 2.501e6
        L1 = 2.37e3
        cpd = const.dry_air_spec_heat_press.m
        cw = 4190.  # specific heat of liquid water
        k0 = (L0 + L1*C)/(cpd + cw*Q)
        k1 = L1/(cpd + cw*Q)
        k2 = 0
        nu = const.dry_air_gas_constant.m/(cpd + cw*Q)
    else:
        raise ValueError("kind must be 'pseudo' or 'reversible'.")

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


def wetbulb(pressure, theta_e, improve=True):
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


def reversible_lapse_daviesjones(
        pressure, initial_temperature, initial_liquid_ratio,
        reference_pressure=None, improve=2):
    """
    Computes temperature along reversible adiabats.
    
    Uses the method of Davies-Jones (2008).
    
    Args:
        pressure: Array of pressures for which the temperature is to
            be found.
        initial_temperature: Initial parcel temperature.
        q_initial: Initial specific humidity.
        initial_liquid_ratio: Initial ratio of liquid mass to total.
        reference_pressure: The pressure corresponding to
            initial_temperature. Optional, defaults to pressure[0].
        improve: Number of iterations of Newton's method to execute.
    
    Returns:
        Array of final temperatures.
    """
    
    pressure = np.atleast_1d(pressure).m_as(units.mbar)
    if reference_pressure is None:
        reference_pressure = pressure[0]
    else:
        reference_pressure = reference_pressure.m_as(units.mbar)
    
    # nondimensional pressure
    reference_pi = (reference_pressure/1000.0)**(1./3.504)
    C = 273.15
    # initial specific humidity is saturated specific humidity
    q_initial = saturation_specific_humidity(
        pressure[0]*units.mbar, initial_temperature).m
    # total mixing ratio (liquid + vapour)
    Q = ((q_initial + initial_liquid_ratio)
         /(1 - q_initial - initial_liquid_ratio))
    if hasattr(Q, 'units'):
        Q = Q.m_as(units.dimensionless)  # make sure Q is a number
        
    # see eq. 5.3 of Davies-Jones 2008
    f_initial = _daviesjones_f(
        initial_temperature.m_as(units.kelvin), reference_pi, Q=Q,
        kind='reversible')
    A1 = f_initial**(-1/3.504)*C/reference_pi
    
    def single(p):
        """Finds the final temperature for a single pressure value."""
        
        # initial guess using pseudoadiabat
        temperature = moist_lapse(
            p*units.mbar, initial_temperature,
            reference_pressure*units.mbar, method='fast',
            improve=False).m_as(units.celsius)

        pi = (p/1000.0)**(1./3.504)
        X = (C/(A1*pi))**3.504
        for i in range(improve):
            # apply iterations of Newton's method (eq. 2.6)
            slope = _daviesjones_fprime(
                temperature + C, pi, Q=Q, kind='reversible')
            fvalue = _daviesjones_f(
                temperature + C, pi, Q=Q, kind='reversible')
            temperature = temperature - (fvalue - X)/slope
        
        return temperature
        
    return ([single(p) for p in pressure] if pressure.size > 1
            else single(pressure.item()))*units.celsius


def reversible_lapse_saunders(
        pressure, t_initial, l_initial, reference_pressure=None, improve=2):
    """
    Calculates temperature along reversible adiabats.
    
    Uses Eq. 3 of Saunders (1957).
    
    Args:
        pressure: Pressure array.
        t_initial: Initial temperature.
        l_initial: Initial ratio of liquid mass to total
        reference_pressure: Pressure corresponding to t_inital.
        improve: Number of Newton's method iterations to use.
        
    Returns:
        Resultant temperature array.
    """
    
    pressure = np.atleast_1d(pressure).m_as(units.mbar)
    if reference_pressure is None:
        reference_pressure = pressure[0]
    else:
        reference_pressure = reference_pressure.m_as(units.mbar)
        
    t_initial = t_initial.m_as(units.kelvin)
    if hasattr(l_initial, 'units'):
        l_initial = l_initial.m_as(units.dimensionless)
    
    # constants
    cp = const.dry_air_spec_heat_press.m
    cw = const.water_specific_heat.m*1e3
    R = const.dry_air_gas_constant.m
    C = 273.15
    e0 = 6.112
    a = 17.67
    b = 243.5
    epsilon = const.epsilon.m
    L0 = 2.501e6
    L1 = 2.37e3
    
    # total vapour + liquid water mixing ratio (invariant)
    q_initial = saturation_specific_humidity(
        reference_pressure*units.mbar, t_initial*units.kelvin).m
    r = (q_initial + l_initial)/(1 - q_initial - l_initial)
    
    def saunders_function(p, t):
        """Evaluates the LHS of Eq. 3 and its derivative w.r.t. temperature"""
        
        # saturation vapour pressure and derivative
        es = e0*np.exp(a*(t - C)/(t - C + b))
        des_dt = a*b/(t - C + b)**2 * es
        
        # saturation (vapour) mixing ratio and derivative
        rw = epsilon*es/(p - es)
        drw_dt = epsilon*p*des_dt/(p - es)**2
        
        # latent heat of vapourisation of water and derivative
        Lv = L0 - L1*(t - C)
        dLv_dt = -L1
        
        # LHS of Eq. 3 and derivative
        fvalue = (cp + r*cw)*np.log(t) + rw*Lv/t - R*np.log(p - es)
        fprime = ((cp + r*cw)/t + (t*(drw_dt*Lv + rw*dLv_dt) - rw*Lv)/t**2
                  + R*des_dt/(p - es))
        
        return fvalue, fprime
    
    # RHS of Eq. 3
    A, _ = saunders_function(reference_pressure, t_initial)
    
    # initial guess: pseudoadiabatic values
    t_final = moist_lapse(
        pressure*units.mbar, t_initial*units.kelvin,
        reference_pressure*units.mbar).m_as(units.kelvin)
    
    # apply Newton's method
    for i in range(improve):
        fvalue, fprime = saunders_function(pressure, t_final)
        t_final = t_final - (fvalue - A)/fprime
    
    return t_final*units.kelvin
    

# ---------- adiabatic descent calculation ----------

def descend(
        pressure, temperature, specific_humidity, liquid_ratio,
        reference_pressure, improve=2, improve_reversible=2, kind='pseudo'):
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
        improve_reversible: Number of iterations to use for reversible
            adiabat calculation.
        kind: 'pseudo' for pseudoadiabats, 'reversible' for reversible
            adiabats.

    Returns:
        Final temperature, specific humidity and liquid ratio.
    """

    # calculate dry adiabatic value outside if statement since it
    # is needed for the guess in case 2.2
    t_final_dry = mpcalc.dry_lapse(pressure, temperature, reference_pressure)

    if liquid_ratio <= 0:
        # case 1: dry adiabat only
        q_final = specific_humidity
        l_final = 0*units.dimensionless
        return t_final_dry, q_final, l_final
    else:
        # case 2: some moist descent
        if kind == 'pseudo':
            t_final_moist = moist_lapse(
                pressure, temperature, reference_pressure)
        elif kind == 'reversible':
            t_final_moist = reversible_lapse_saunders(
                pressure, temperature, liquid_ratio,
                reference_pressure, improve=improve_reversible)
        else:
            raise ValueError("kind must be 'pseudo' or 'reversible'.")
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
                reference_pressure, temperature, specific_humidity,
                with_units=False)
            if improve == 'exact':
                # iterate until convergence using Newton's method
                def root_function(T):
                    value, slope = theta_e(
                        pressure, T*units.kelvin, q_final, prime=True,
                        with_units=False)
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
                        pressure, t_final*units.kelvin, q_final, prime=True,
                        with_units=False)
                    t_final = t_final - (value - theta_e_initial)/slope
                t_final = t_final*units.kelvin

            return t_final, q_final, l_final


# ---------- entrainment calculations ----------

def mix(parcel, environment, rate, dz):
    """
    Mixes parcel and environment variables (for entrainment).

    Args:
        parcel: Parcel value.
        environment: Environment value.
        rate: Entrainment rate.
        dz: Distance descended.

    Returns:
        Mixed value of the variable.
    """

    return parcel + rate * (environment - parcel) * dz


def equilibrate(
        pressure, t_parcel, q_parcel, l_parcel, t_env, q_env, l_env, rate, dz):
    """
    Finds parcel properties after entrainment and phase equilibration.

    Args:
        pressure: Pressure during the change (constant).
        t_parcel: Initial temperature of the parcel.
        q_parcel: Initial specific humidity of the parcel.
        l_parcel: Initial ratio of liquid mass to parcel mass.
        t_env: Temperature of the environment.
        q_env: Specific humidity of the environment.
        l_env: Liquid ratio of the environment.
        rate: Entrainment rate.
        dz: Distance descended.

    Returns:
        A tuple containing the final parcel temperature, specific
            humidity and liquid ratio.
    """

    # mixing without phase change
    t_mixed = mix(t_parcel, t_env, rate, dz)
    q_mixed = mix(q_parcel, q_env, rate, dz)
    l_mixed = mix(l_parcel, l_env, rate, dz)
    q_mixed_saturated = saturation_specific_humidity(pressure, t_mixed)

    if q_mixed > q_mixed_saturated:
        # we need to condense water vapour
        # ept = theta_e(pressure, t_mixed, q_mixed)
        # t_final = wetbulb(pressure, ept, improve=True)
        t_final = wetbulb_romps(pressure, t_mixed, q_mixed)
        q_final = saturation_specific_humidity(pressure, t_final)
        l_final = l_mixed + q_mixed - q_final
        return (t_final, q_final, l_final)
    elif q_mixed < q_mixed_saturated and l_mixed > 0:
        # we need to evaporate liquid water.
        # if all liquid evaporates:
        t_all_evap = t_mixed + temperature_change(l_mixed)
        q_all_evap_saturated = saturation_specific_humidity(
            pressure, t_all_evap)

        if q_mixed + l_mixed <= q_all_evap_saturated:
            return (t_all_evap, q_mixed + l_mixed, 0*units.dimensionless)
        else:
            # ept = theta_e(pressure, t_mixed, q_mixed)
            # t_final = wetbulb(pressure, ept, improve=True)
            t_final = wetbulb_romps(pressure, t_mixed, q_mixed)
            q_final = saturation_specific_humidity(pressure, t_final)
            l_final = l_mixed + q_mixed - q_final
            return (t_final, q_final, l_final)
    elif q_mixed < q_mixed_saturated and l_mixed <= 0:
        # already in equilibrium, no action needed
        return (t_mixed, q_mixed, 0*units.dimensionless)
    else:
        # parcel is perfectly saturated, no action needed
        return (t_mixed, q_mixed, l_mixed)


# ---------- old functions (to be removed) ----------

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
