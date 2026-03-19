# Auxiliary Variables Implementation

## Overview

The package now supports **four methods** for handling static/auxiliary variables, with smart defaults that balance speed and robustness.

## Methods

### 1. `'auto'` (DEFAULT) - Best of Both Worlds
```python
process_model(..., aux_method='auto')  # or just omit parameter
```

**Behavior:**
1. First tries **analytical** method (fast symbolic solving)
2. If SymPy fails → automatically falls back to **dynamic** method (Dynare-style: include in system)
3. Issues a warning when fallback occurs

**When to use:** Most cases - you get speed when possible, robustness when needed

**Example:**
```python
# Simple equation: i = y - c - g
model_funcs = process_model(equations, vars_dyn, vars_aux=['i'])
# Result: Uses analytical (fast!)

# Complex equation: z^5 + z^3 + z = x + y
model_funcs = process_model(equations, vars_dyn, vars_aux=['z'])
# Result: Falls back to dynamic (single optimization, Dynare-style!)
```

---

### 2. `'analytical'` - Force Symbolic Solving
```python
process_model(..., aux_method='analytical')
```

**Behavior:**
- Solves auxiliary equations symbolically using SymPy
- **Fails with ValueError** if SymPy can't solve
- No fallback

**When to use:**
- When you know equations are simple (linear, polynomial)
- When you want to ensure no nested optimization overhead
- When you want explicit failure if equations are too complex

**Pros:** Fastest method, reduces solver dimensionality
**Cons:** Limited to analytically solvable equations, can hang on complex cases

---

### 3. `'nested'` - Force Numerical Solving
```python
process_model(..., aux_method='nested')
```

**Behavior:**
- Skips analytical attempt entirely
- After `solve_perfect_foresight` returns, solves auxiliary equations numerically
  for each time period in sequence (post-solve, not inline at each Newton iteration)
- Uses scipy.optimize.root with warm starting across periods

**When to use:**
- Complex auxiliary equations (implicit, transcendental, coupled)
- When you know analytical will fail or hang
- When robustness is more important than speed

**Pros:** Works for ANY auxiliary equations, never hangs
**Cons:** Slower due to nested optimization

---

### 4. `'dynamic'` - Treat as Jump Variables
```python
process_model(..., aux_method='dynamic')
```

**Behavior:**
- No special handling for auxiliary variables
- Treats them as regular dynamic (jump) variables
- Increases solver dimensionality but avoids nested solving

**When to use:**
- When you prefer simpler code over performance
- When auxiliary equations are very cheap to solve alongside dynamics
- When you want to avoid any nested optimization overhead

**Pros:** Simplest approach, no nested loops
**Cons:** Higher dimensionality, no computational savings

---

## Complete Example

```python
import sympy as sp
import numpy as np
from dynare_python import v, process_model, solve_perfect_foresight

# Parameters
beta, delta, alpha = sp.symbols("beta delta alpha")

# Dynamic variables (have leads/lags)
vars_dyn = ["c", "k"]

# Auxiliary variables (static, derived)
vars_aux = ["i"]

# Exogenous variables
vars_exo = ["g"]

# Define equations
c_0, c_p = v("c", 0), v("c", 1)
k_0, k_p = v("k", 0), v("k", 1)
i_0 = v("i", 0)
g_0 = v("g", 0)
y_0 = k_0**alpha

eq_euler = 1/c_0 - beta*(alpha*k_p**(alpha-1) + (1-delta))/c_p
eq_kacc = k_p - (1-delta)*k_0 - y_0 + c_0 + g_0
eq_i = y_0 - c_0 - i_0 - g_0  # Auxiliary equation

equations = [eq_euler, eq_kacc, eq_i]

# Process model (uses 'auto' by default)
model_funcs = process_model(equations, vars_dyn,
                           vars_exo=vars_exo,
                           vars_aux=vars_aux)

print(f"Method used: {model_funcs['aux_method']}")
# Output: "analytical" (simple equation, SymPy solved it)

# Solve
params = {beta: 0.96, delta: 0.08, alpha: 0.36}
ss = np.array([1.2, 5.4])  # Steady state for [c, k]
T = 100
X0 = np.tile(ss, (T, 1))
exog_path = np.full((T, 1), 0.2)

sol = solve_perfect_foresight(T, X0, params, ss, model_funcs, vars_dyn,
                              exog_path=exog_path)

# Access results
X_dyn = sol.x.reshape(T, -1)  # Dynamic variables [c, k]
X_aux = sol.x_aux              # Auxiliary variables [i]

c_path = X_dyn[:, 0]
k_path = X_dyn[:, 1]
i_path = X_aux[:, 0]  # Computed automatically!
```

---

## Implementation Details

### How 'auto' Works

1. **Identify auxiliary equations**: Static equations containing auxiliary variables
2. **Try analytical**: Call `sympy.solve(aux_eqs, aux_vars)`
   - Success → Compile closed-form functions, remove equations from system
   - Failure → Issue warning, proceed to step 3
3. **Fallback to dynamic**: Merge auxiliary variables into dynamic variables, keep auxiliary equations in system
4. **Return**: Model with appropriate method recorded (`'analytical'` or `'dynamic'`)

### How 'nested' Works at Runtime

After `solve_perfect_foresight` converges on the dynamic variables, auxiliary
variables are solved in a separate post-processing pass:

```python
# After solver converges on [c, k] paths:
for t in range(T):
    # Given converged c[t], k[t]
    # Solve: find i[t] such that y[t] - c[t] - i[t] - g[t] = 0
    i[t] = scipy.optimize.root(aux_residual, guess)
    guess = i[t]  # Warm start next period
```

Uses warm starting: `i[t]` as initial guess for `i[t+1]`

### How 'analytical' Works at Runtime

After solving for [c, k] paths:

```python
# Post-processing: evaluate closed-form solution
for t in range(T):
    i[t] = y(k[t]) - c[t] - g[t]  # Direct evaluation, no iteration
```

### How 'dynamic' Works at Runtime

Auxiliary variables treated as dynamic variables (Dynare-style):

```python
# Single optimization: solve for [c, k, i] paths together
# Auxiliary equation becomes part of the system:
# F1: Euler equation for c
# F2: Capital accumulation for k
# F3: Auxiliary equation: y - c - i - g = 0

# Solver finds [c, k, i] paths such that all residuals = 0
# No nested optimization, single Jacobian, but higher dimensional
```

**Key difference from nested**: Dynamic method solves one large system alongside the main Newton iterations; nested solves auxiliary variables in a separate post-processing pass after the main solver has converged.

---

## Performance Comparison

| Method | Dimensionality | Speed | Robustness | Use Case |
|--------|---------------|-------|------------|----------|
| **auto** (→analytical) | Low (n_dyn) | ⚡⚡⚡ Very Fast | ✓✓✓ High | Default choice |
| **auto** (→dynamic) | High (n_dyn+n_aux) | ⚡⚡ Fast | ✓✓✓ High | Complex equations |
| **analytical** (forced) | Low (n_dyn) | ⚡⚡⚡ Very Fast | ✓ Limited | Simple equations only |
| **nested** (forced) | Low (n_dyn) | ⚡⚡ Fast | ✓✓✓ High | Explicit nested solving |
| **dynamic** (forced) | High (n_dyn+n_aux) | ⚡⚡ Fast | ✓✓✓ High | Dynare-style approach |

---

## Recommendations

### For Most Users
```python
# Just use the default!
model_funcs = process_model(equations, vars_dyn, vars_aux=vars_aux)
```
The 'auto' method intelligently chooses the best approach.

### For Simple Auxiliary Equations
```python
# e.g., i = y - c - g, z = x^2 + y
model_funcs = process_model(equations, vars_dyn, vars_aux=vars_aux,
                           aux_method='analytical')
```
Guarantees no nested overhead, fails fast if equations are complex.

### For Complex Auxiliary Equations
```python
# e.g., z^5 + z^3 + z = x + y, x*y + sin(x) = c
# Option 1: Use auto (will fallback to dynamic)
model_funcs = process_model(equations, vars_dyn, vars_aux=vars_aux)

# Option 2: Force dynamic if you know analytical won't work
model_funcs = process_model(equations, vars_dyn, vars_aux=vars_aux,
                           aux_method='dynamic')

# Option 3: Explicit nested solving (if preferred over dynamic)
model_funcs = process_model(equations, vars_dyn, vars_aux=vars_aux,
                           aux_method='nested')
```
Auto will try analytical, then fallback to dynamic (Dynare-style). Or force your preferred method.

### For Maximum Simplicity
```python
# Treat everything as dynamic variables
model_funcs = process_model(equations, vars_dyn + vars_aux,
                           aux_method='dynamic')
```
Higher dimensional but conceptually simpler.

---

## Design Philosophy

1. **Smart defaults**: 'auto' tries fast method first, falls back to robust method
2. **User control**: Force specific method when you know what's best
3. **Graceful degradation**: Analytical → Dynamic fallback (matching Dynare's approach)
4. **Clear feedback**: Method used is recorded in `model_funcs['aux_method']`
5. **Dynare compatibility**: When analytical fails, uses Dynare-style treatment (include in system)

This design gives you the best of both worlds: speed when possible, robustness when needed, and full control when desired. The fallback to dynamic method follows Dynare's philosophy of using a single optimization over all time periods with block-recursive structure when analytical elimination is not possible.
