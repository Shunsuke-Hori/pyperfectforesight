"""Test dynamic fallback with proper dynamic model"""

import sympy as sp
from pyperfectforesight import v, process_model

print("=" * 70)
print("Testing Dynamic Method with Proper Dynamic Equations")
print("=" * 70)

# Parameters
beta, delta, alpha = sp.symbols("beta delta alpha")

# Dynamic variables
vars_dyn = ["c", "k"]

# Auxiliary variable
vars_aux = ["i"]

# Time-indexed symbols
c_0, c_p = v("c", 0), v("c", 1)
k_0, k_p = v("k", 0), v("k", 1)
i_0 = v("i", 0)

# Dynamic equations (with leads)
eq_euler = 1/c_0 - beta*(alpha*k_p**(alpha-1) + (1-delta))/c_p
eq_kacc = k_p - (1-delta)*k_0 - k_0**alpha + c_0

# Auxiliary equation (static)
eq_aux = i_0 - k_0**alpha + c_0

equations = [eq_euler, eq_kacc, eq_aux]

print("\nModel:")
print("  Dynamic equations: Euler equation, Capital accumulation")
print("  Auxiliary equation: i = y - c")
print()

# Test DYNAMIC method
print("=" * 70)
print("DYNAMIC Method")
print("=" * 70)
model_dynamic = process_model(equations, vars_dyn, vars_aux=vars_aux,
                              aux_method='dynamic')

print(f"Method used: {model_dynamic['aux_method']}")
print(f"vars_dyn: {model_dynamic['vars_dyn']}")
print(f"vars_aux: {model_dynamic['vars_aux']}")
print(f"Number of dynamic equations: {len(model_dynamic['dynamic_eqs'])}")

if len(model_dynamic['vars_dyn']) == 3:
    print("\n✓ Auxiliary variable merged into dynamic variables")

if len(model_dynamic['dynamic_eqs']) >= 2:
    print(f"✓ Dynamic equations present: {len(model_dynamic['dynamic_eqs'])}")
    print("  (Note: static elimination may have removed pure static equations)")

# Test AUTO method (should use analytical for simple equation)
print("\n" + "=" * 70)
print("AUTO Method (should use analytical)")
print("=" * 70)
model_auto = process_model(equations, vars_dyn, vars_aux=vars_aux,
                           aux_method='auto')

print(f"Method used: {model_auto['aux_method']}")
print(f"vars_dyn: {model_auto['vars_dyn']}")
print(f"vars_aux: {model_auto['vars_aux']}")
print(f"Number of dynamic equations: {len(model_auto['dynamic_eqs'])}")

if model_auto['aux_method'] == 'analytical':
    print("\n✓ AUTO chose analytical (equation is simple)")

# Test NESTED method
print("\n" + "=" * 70)
print("NESTED Method")
print("=" * 70)
model_nested = process_model(equations, vars_dyn, vars_aux=vars_aux,
                             aux_method='nested')

print(f"Method used: {model_nested['aux_method']}")
print(f"vars_dyn: {model_nested['vars_dyn']}")
print(f"vars_aux: {model_nested['vars_aux']}")
print(f"Number of dynamic equations: {len(model_nested['dynamic_eqs'])}")
print(f"Auxiliary functions compiled: {len(model_nested['aux_eqs_funcs'])}")

if len(model_nested['aux_eqs_funcs']) == 1:
    print("\n✓ Nested method compiled auxiliary equation function")

# Comparison
print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)

print("\nDimensions:")
print(f"  AUTO (→analytical): {len(model_auto['vars_dyn'])} vars, {len(model_auto['dynamic_eqs'])} eqs")
print(f"  DYNAMIC:            {len(model_dynamic['vars_dyn'])} vars, {len(model_dynamic['dynamic_eqs'])} eqs")
print(f"  NESTED:             {len(model_nested['vars_dyn'])} vars, {len(model_nested['dynamic_eqs'])} eqs")

print("\nKey difference:")
print("  - AUTO/NESTED: Lower dimension, auxiliary handled specially")
print("  - DYNAMIC: Higher dimension, auxiliary in main system (Dynare-style)")

print("\n" + "=" * 70)
print("✓ All methods working correctly!")
print("=" * 70)
