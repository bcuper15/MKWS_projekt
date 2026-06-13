"""
Shared helpers for the CH4/H2 blend combustion project.

The fuel blend composition is parametrized by X_H2, the molar fraction of
hydrogen in the fuel mixture (CH4 + H2 = 1):

    X_H2 = 0.0  -> pure methane
    X_H2 = 1.0  -> pure hydrogen

The oxidizer is always air, represented as O2:1, N2:3.76 (molar).
"""

import cantera as ct

MECHANISM = "gri30.yaml"
OXIDIZER = "O2:1, N2:3.76"


def fuel_composition(x_h2: float) -> str:
    """Return a Cantera composition string for a CH4/H2 fuel blend.

    Parameters
    ----------
    x_h2 : float
        Molar fraction of H2 in the fuel blend, 0 <= x_h2 <= 1.
    """
    if not 0.0 <= x_h2 <= 1.0:
        raise ValueError("x_h2 must be between 0 and 1")
    x_ch4 = 1.0 - x_h2
    return f"CH4:{x_ch4}, H2:{x_h2}"


def make_gas(mechanism: str = MECHANISM) -> ct.Solution:
    """Create a fresh Solution object for the given mechanism."""
    return ct.Solution(mechanism)


def set_mixture(gas: ct.Solution, phi: float, x_h2: float,
                T: float = 300.0, P: float = ct.one_atm) -> ct.Solution:
    """Set the gas state to a CH4/H2/air mixture at the given conditions.

    Parameters
    ----------
    gas : ct.Solution
        Gas object to modify in place.
    phi : float
        Equivalence ratio.
    x_h2 : float
        Molar fraction of H2 in the fuel blend (0 = pure CH4, 1 = pure H2).
    T : float
        Initial temperature [K].
    P : float
        Initial pressure [Pa].
    """
    gas.TP = T, P
    gas.set_equivalence_ratio(phi, fuel_composition(x_h2), OXIDIZER)
    return gas


def lhv_mass(x_h2: float) -> float:
    """Lower heating value of the CH4/H2 fuel blend on a mass basis [MJ/kg].

    Computed from the standard enthalpy of the complete combustion reaction
    (products: CO2 and H2O(g)) using the gri30 mechanism thermo data, mixed
    on a mass-weighted basis for the given molar blend composition.
    """
    gas = make_gas()
    gas.TP = 298.15, ct.one_atm
    gas.X = fuel_composition(x_h2)
    m_fuel = gas.mean_molecular_weight  # kg/kmol, for 1 kmol of fuel blend
    h_fuel_total = gas.enthalpy_mole  # J, enthalpy of 1 kmol of fuel blend

    # Stoichiometric combustion products for 1 kmol of fuel blend
    x_ch4 = 1.0 - x_h2
    n_co2 = x_ch4
    n_h2o = 2 * x_ch4 + x_h2
    n_o2 = 2 * x_ch4 + 0.5 * x_h2

    products = make_gas()
    products.TPX = 298.15, ct.one_atm, {"CO2": n_co2, "H2O": n_h2o}
    h_products_total = products.enthalpy_mole * (n_co2 + n_h2o)

    reactants_o2 = make_gas()
    reactants_o2.TPX = 298.15, ct.one_atm, {"O2": 1.0}
    h_o2_total = reactants_o2.enthalpy_mole * n_o2

    # LHV = -(H_products - H_reactants) / mass_of_fuel
    delta_h = h_products_total - (h_fuel_total + h_o2_total)
    return -delta_h / m_fuel / 1e6  # MJ/kg fuel
