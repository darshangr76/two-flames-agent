"""
ZENITHFORGE - Unit Integrity Layer
Ensures all data flowing through the engine preserves canonical NASA units.
"""

def standardize_units(o2_pct, pressure_val, pressure_unit, temp_val, temp_unit):
    """
    Converts inputs to the canonical NASA standard:
    - Oxygen: % (0-100)
    - Pressure: kPa
    - Temperature: Kelvin (K)
    - Time: Seconds (s)
    - Burning Rate: mm/s (or mm^2/s as defined by NASA TP)
    """
    # 1. Pressure Conversion
    if pressure_unit.lower() == "atm":
        pressure_kpa = pressure_val * 101.325
    elif pressure_unit.lower() == "mmhg":
        pressure_kpa = pressure_val * 0.133322
    elif pressure_unit.lower() == "psi":
        pressure_kpa = pressure_val * 6.89476
    else:
        pressure_kpa = pressure_val # Assume kPa
        
    # 2. Temperature Conversion
    if temp_unit.lower() == "c":
        temp_k = temp_val + 273.15
    elif temp_unit.lower() == "f":
        temp_k = (temp_val - 32) * 5/9 + 273.15
    else:
        temp_k = temp_val # Assume Kelvin
        
    # 3. Oxygen Mole Fraction (Dimensionless to %)
    o2_vol_pct = o2_pct * 100 if o2_pct <= 1.0 else o2_pct
    
    return {
        "oxygen_pct": round(o2_vol_pct, 2),
        "pressure_kpa": round(pressure_kpa, 2),
        "temperature_k": round(temp_k, 2)
    }

if __name__ == "__main__":
    print("Testing Unit Integrity Layer:")
    print("1 atm ->", standardize_units(0.21, 1.0, "atm", 25, "C"))
    print("760 mmHg ->", standardize_units(0.21, 760, "mmHg", 298, "K"))
    print("14.7 psi ->", standardize_units(0.21, 14.7, "psi", 77, "F"))
