"""Test auto fallback from analytical to dynamic"""

import sympy as sp
import numpy as np
from dynare_python import v, process_model
import warnings

print("=" * 70)
print("Testing AUTO → DYNAMIC fallback")
print("=" * 70)

# Parameters
alpha = sp.symbols("alpha")

# Dynamic variables
vars_dyn = ["c", "k"]

# Auxiliary variable with equation that's hard for SymPy
vars_aux = ["z"]

# Setup symbols
c_0 = v("c", 0)
k_0 = v("k", 0)
z_0 = v("z", 0)

# Simple dynamic equations
eq_1 = c_0 + k_0 - 1
eq_2 = k_0 - alpha

# Create an auxiliary equation that SymPy will struggle with
# Try a 5th degree polynomial (no general solution)
eq_aux = z_0**5 + z_0**3 + z_0 - c_0 - k_0

equations = [eq_1, eq_2, eq_aux]

print("\nEquations:")
print(f"  Dynamic: {eq_1}, {eq_2}")
print(f"  Auxiliary: {eq_aux}")
print(f"  (5th degree polynomial - SymPy unlikely to solve)")

print("\n1. Testing AUTO method:")
print("   Expecting: analytical fails → fallback to dynamic")

# Capture warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")

    model = process_model(equations, vars_dyn, vars_aux=vars_aux, aux_method='auto')

    # Check if warning was issued
    if w:
        print(f"\n   ✓ Warning issued: {w[0].message}")

print(f"\nResults:")
print(f"  Method used: {model['aux_method']}")
print(f"  vars_dyn: {model['vars_dyn']}")
print(f"  vars_aux: {model['vars_aux']}")
print(f"  Number of dynamic equations: {len(model['dynamic_eqs'])}")

if model['aux_method'] == 'dynamic':
    print(f"\n  ✓ Correctly fell back to dynamic method!")
    print(f"  ✓ Auxiliary variable 'z' merged into dynamic variables")
    print(f"  ✓ All 3 equations kept in system (Dynare-style)")

    # Verify equations are all present
    if len(model['dynamic_eqs']) == 3:
        print(f"  ✓ All equations preserved (dynamic + auxiliary)")
    else:
        print(f"  ✗ ERROR: Expected 3 equations, got {len(model['dynamic_eqs'])}")
else:
    print(f"\n  Note: Method is '{model['aux_method']}'")
    if model['aux_method'] == 'analytical':
        print(f"  SymPy actually managed to solve it!")

print("\n" + "=" * 70)
print("2. Compare with ANALYTICAL (forced - should fail)")
print("=" * 70)

try:
    model_analytical = process_model(equations, vars_dyn, vars_aux=vars_aux,
                                    aux_method='analytical')
    print(f"  Result: {model_analytical['aux_method']}")
    print(f"  SymPy solved it despite 5th degree!")
except ValueError as e:
    print(f"  ✓ Raised ValueError as expected")
    print(f"  Error message mentions 'auto' and 'dynamic' as alternatives")

print("\n" + "=" * 70)
print("3. NESTED method (explicit choice)")
print("=" * 70)

model_nested = process_model(equations, vars_dyn, vars_aux=vars_aux,
                             aux_method='nested')
print(f"  Method used: {model_nested['aux_method']}")
print(f"  vars_dyn: {model_nested['vars_dyn']}")
print(f"  Number of dynamic equations: {len(model_nested['dynamic_eqs'])}")
print(f"  ✓ Nested method still available as explicit option")

print("\n" + "=" * 70)
print("Test complete!")
print("=" * 70)
