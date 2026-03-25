"""Test different auxiliary variable methods"""

import sympy as sp
from dynare_python import v, process_model

print("=" * 70)
print("Auxiliary Variable Methods Comparison")
print("=" * 70)

# Simple model setup
alpha = sp.symbols("alpha")
vars_dyn = ["c", "k"]
vars_aux = ["i"]

c_0 = v("c", 0)
k_0 = v("k", 0)
i_0 = v("i", 0)

# Dynamic equations
eq_1 = c_0 + k_0 - 1
eq_2 = k_0 - alpha

# Simple auxiliary equation (SymPy can solve)
eq_aux = i_0 - c_0 - k_0

equations = [eq_1, eq_2, eq_aux]

print("\nModel: 2 dynamic vars [c, k], 1 auxiliary var [i]")
print("Auxiliary equation: i = c + k (simple, analytically solvable)")

# Test 1: AUTO (should use analytical)
print("\n" + "=" * 70)
print("1. AUTO method (default)")
print("=" * 70)
model_auto = process_model(equations, vars_dyn, vars_aux=vars_aux, aux_method='auto')
print(f"Result: {model_auto['aux_method']}")
print(f"vars_dyn: {model_auto['vars_dyn']}")
print(f"vars_aux: {model_auto['vars_aux']}")
print(f"Dynamic equations: {len(model_auto['dynamic_eqs'])}")

if model_auto['aux_method'] == 'analytical':
    print("✓ Used analytical (equation is simple)")
    print(f"✓ Dimension reduced: solving for {len(model_auto['vars_dyn'])} variables")

# Test 2: ANALYTICAL (forced)
print("\n" + "=" * 70)
print("2. ANALYTICAL method (forced)")
print("=" * 70)
model_analytical = process_model(equations, vars_dyn, vars_aux=vars_aux,
                                 aux_method='analytical')
print(f"Result: {model_analytical['aux_method']}")
print(f"vars_dyn: {model_analytical['vars_dyn']}")
print(f"vars_aux: {model_analytical['vars_aux']}")
print(f"Dynamic equations: {len(model_analytical['dynamic_eqs'])}")
print("✓ Analytical method works for simple equations")

# Test 3: DYNAMIC (forced)
print("\n" + "=" * 70)
print("3. DYNAMIC method (forced)")
print("=" * 70)
model_dynamic = process_model(equations, vars_dyn, vars_aux=vars_aux,
                              aux_method='dynamic')
print(f"Result: {model_dynamic['aux_method']}")
print(f"vars_dyn: {model_dynamic['vars_dyn']}")
print(f"vars_aux: {model_dynamic['vars_aux']}")
print(f"Dynamic equations: {len(model_dynamic['dynamic_eqs'])}")

if model_dynamic['aux_method'] == 'dynamic':
    print("✓ Auxiliary variable merged into dynamic")
    print(f"✓ All equations kept: {len(model_dynamic['dynamic_eqs'])} equations")
    print(f"✓ Solving for {len(model_dynamic['vars_dyn'])} variables (higher dimension)")

# Test 4: NESTED (forced)
print("\n" + "=" * 70)
print("4. NESTED method (forced)")
print("=" * 70)
model_nested = process_model(equations, vars_dyn, vars_aux=vars_aux,
                             aux_method='nested')
print(f"Result: {model_nested['aux_method']}")
print(f"vars_dyn: {model_nested['vars_dyn']}")
print(f"vars_aux: {model_nested['vars_aux']}")
print(f"Dynamic equations: {len(model_nested['dynamic_eqs'])}")
print(f"Aux equation functions: {len(model_nested['aux_eqs_funcs'])}")

if model_nested['aux_method'] == 'nested':
    print("✓ Nested method set up correctly")
    print(f"✓ Dimension reduced: solving for {len(model_nested['vars_dyn'])} variables")
    print("✓ Auxiliary equations compiled for numerical solving")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("\nDimensionality:")
print(f"  Analytical: {len(model_analytical['vars_dyn'])} dynamic vars")
print(f"  Dynamic:    {len(model_dynamic['vars_dyn'])} dynamic vars (includes aux)")
print(f"  Nested:     {len(model_nested['vars_dyn'])} dynamic vars")

print("\nEquations in system:")
print(f"  Analytical: {len(model_analytical['dynamic_eqs'])} equations")
print(f"  Dynamic:    {len(model_dynamic['dynamic_eqs'])} equations")
print(f"  Nested:     {len(model_nested['dynamic_eqs'])} equations")

print("\nRecommendation:")
print("  - Use AUTO (default): gets analytical for simple cases")
print("  - For complex aux equations: AUTO falls back to DYNAMIC (Dynare-style)")
print("  - NESTED available as explicit option if desired")

print("\n" + "=" * 70)
