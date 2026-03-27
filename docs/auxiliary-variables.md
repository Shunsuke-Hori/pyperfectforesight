# Auxiliary Variables

## Overview

Auxiliary (static) variables are variables that appear only at the current period — they have no leads or lags. Examples include investment $i_t = y_t - c_t - g_t$ or any other purely static relationship.

`pyperfectforesight` supports **four methods** for handling auxiliary variables, with a smart default that balances speed and robustness.

## Declaring auxiliary variables

Pass `vars_aux` and the auxiliary equations to `process_model`:

```python
from pyperfectforesight import process_model

# vars_aux lists the auxiliary variable names
model_funcs = process_model(
    equations,          # list of all equations (dynamic + auxiliary)
    vars_dyn,           # list of dynamic variable names
    vars_exo=vars_exo,  # list of exogenous variable names (optional)
    vars_aux=vars_aux,  # list of auxiliary variable names
    aux_method='auto',  # how to handle them (default)
)
```

After the solver returns, auxiliary variable paths are available on `sol.x_aux`:

```python
sol = solve_perfect_foresight(T, params, ss, model_funcs, vars_dyn, ...)
X_dyn = sol.x.reshape(T, -1)   # dynamic variables
X_aux = sol.x_aux               # auxiliary variables, shape (T, n_aux)
```

---

## Methods

### `'auto'` (default) — best of both worlds

```python
process_model(..., aux_method='auto')  # or just omit the parameter
```

**Behavior:**

1. First tries the **analytical** method (fast symbolic solving via SymPy)
2. If SymPy fails, automatically falls back to the **dynamic** method (Dynare-style: include in the main system)
3. Issues a `UserWarning` when fallback occurs

The method actually used is recorded in `model_funcs['aux_method']`.

**When to use:** Most cases. You get speed when possible, robustness when needed.

```python
# Simple equation: i = y - c - g  ->  uses analytical (fast)
model_funcs = process_model(equations, vars_dyn, vars_aux=['i'])
print(model_funcs['aux_method'])   # 'analytical'

# Complex equation: z^5 + z^3 + z = x + y  ->  falls back to dynamic
model_funcs = process_model(equations, vars_dyn, vars_aux=['z'])
print(model_funcs['aux_method'])   # 'dynamic'
```

---

### `'analytical'` — force symbolic solving

```python
process_model(..., aux_method='analytical')
```

**Behavior:** Solves auxiliary equations symbolically using SymPy. Raises `ValueError` if SymPy cannot find a closed-form solution (no fallback).

**When to use:**
- Equations are simple (linear, polynomial)
- You want to guarantee no nested optimization overhead
- You want explicit failure if equations are too complex

**Pros:** Fastest method; reduces solver dimensionality to `n_dyn`.
**Cons:** Limited to analytically solvable equations; can hang on complex symbolic expressions.

---

### `'nested'` — force post-solve numerical solving

```python
process_model(..., aux_method='nested')
```

**Behavior:** After `solve_perfect_foresight` converges on the dynamic variables, auxiliary equations are solved numerically in a post-processing pass — one period at a time, with warm starting across periods:

```python
# After solver converges on [c, k] paths:
for t in range(T):
    # Solve: find i[t] such that y[t] - c[t] - i[t] - g[t] = 0
    i[t] = scipy.optimize.root(aux_residual, guess)
    guess = i[t]   # warm start next period
```

**Structural requirements** (raises `ValueError` if violated):
- The auxiliary system must be **square**: number of auxiliary equations must equal the number of auxiliary variables
- Auxiliary variables must **only appear in auxiliary equations** — they cannot appear in any non-auxiliary model equation

**When to use:** Complex auxiliary equations (implicit, transcendental, coupled) that form a self-contained square subsystem.

**Pros:** Handles many nonlinear auxiliary systems without enlarging the main Newton system; avoids SymPy hangs.
**Cons:** Slower due to the post-solve numerical pass; raises `ValueError` if the auxiliary block violates the structural requirements — use `'dynamic'` in those cases.

---

### `'dynamic'` — treat as jump variables

```python
process_model(..., aux_method='dynamic')
```

**Behavior:** No special handling. Auxiliary variables are merged into `vars_dyn` and treated as regular jump variables. The auxiliary equations become part of the main Newton system.

**When to use:**
- Simpler code is preferred over performance
- Auxiliary equations are cheap and `'nested'` structural requirements are not met
- You want to avoid any nested optimization overhead

**Pros:** Conceptually simplest; no nested loops; single Jacobian.
**Cons:** Higher solver dimensionality (`n_dyn + n_aux`); no dimensional reduction.

---

## Complete example

```python
import sympy as sp
import numpy as np
from pyperfectforesight import v, process_model, solve_perfect_foresight

# Parameters
beta, delta, alpha = sp.symbols("beta delta alpha")

vars_dyn = ["c", "k"]    # dynamic variables (have leads/lags)
vars_aux = ["i"]          # auxiliary variable (static: i = y - c - g)
vars_exo = ["g"]          # exogenous government spending

c_0, c_p = v("c", 0), v("c", 1)
k_0, k_p = v("k", 0), v("k", 1)
i_0      = v("i", 0)
g_0      = v("g", 0)
y_0      = k_0**alpha

eq_euler = 1/c_0 - beta*(alpha*k_p**(alpha-1) + (1-delta))/c_p
eq_kacc  = k_p - (1-delta)*k_0 - y_0 + c_0 + g_0
eq_i     = y_0 - c_0 - i_0 - g_0   # auxiliary equation

equations = [eq_euler, eq_kacc, eq_i]

# Process model (uses 'auto' by default)
model_funcs = process_model(
    equations, vars_dyn,
    vars_exo=vars_exo,
    vars_aux=vars_aux,
)
print(f"Method used: {model_funcs['aux_method']}")
# Output: "analytical" (simple linear equation, SymPy solved it instantly)

# Solve
params = {beta: 0.96, delta: 0.08, alpha: 0.36}
ss = np.array([1.2, 5.4])   # steady state for [c, k]
T = 100
X0 = np.tile(ss, (T, 1))
exog_path = np.full((T, 1), 0.2)   # constant government spending

sol = solve_perfect_foresight(T, params, ss, model_funcs, vars_dyn,
                              exog_path=exog_path)

# Access results
X_dyn = sol.x.reshape(T, -1)   # dynamic variables [c, k]
X_aux = sol.x_aux               # auxiliary variables [i]

c_path = X_dyn[:, 0]
k_path = X_dyn[:, 1]
i_path = X_aux[:, 0]            # computed automatically
```

---

## Method comparison

| Method | Solver dimensionality | Speed | Robustness | Best for |
|---|---|---|---|---|
| `auto` -> analytical | Low (`n_dyn`) | Very fast | High | Default choice |
| `auto` -> dynamic | High (`n_dyn + n_aux`) | Fast | High | Complex equations |
| `analytical` (forced) | Low (`n_dyn`) | Very fast | Limited | Simple equations only |
| `nested` (forced) | Low (`n_dyn`) | Fast | High | Explicit nested solving |
| `dynamic` (forced) | High (`n_dyn + n_aux`) | Fast | High | Dynare-style approach |

---

## Recommendations

### For most users

```python
# Just use the default — 'auto' chooses the best approach automatically.
model_funcs = process_model(equations, vars_dyn, vars_aux=vars_aux)
```

### For simple auxiliary equations

```python
# e.g., i = y - c - g,  z = x^2 + y
model_funcs = process_model(equations, vars_dyn, vars_aux=vars_aux,
                            aux_method='analytical')
```

Guarantees no nested overhead; fails fast if equations are too complex.

### For complex auxiliary equations

```python
# e.g., z^5 + z^3 + z = x + y

# Option 1: Use auto (will fall back to dynamic automatically)
model_funcs = process_model(equations, vars_dyn, vars_aux=vars_aux)

# Option 2: Force dynamic directly
model_funcs = process_model(equations, vars_dyn, vars_aux=vars_aux,
                            aux_method='dynamic')

# Option 3: Explicit nested (requires square, self-contained subsystem)
model_funcs = process_model(equations, vars_dyn, vars_aux=vars_aux,
                            aux_method='nested')
```

### For maximum simplicity

```python
# Include everything in vars_dyn — no aux handling at all.
model_funcs = process_model(equations, vars_dyn + vars_aux,
                            aux_method='dynamic')
```

---

## Design philosophy

1. **Smart defaults**: `'auto'` tries the fast method first, falls back to the robust method.
2. **User control**: Force a specific method when you know what is best.
3. **Graceful degradation**: Analytical to Dynamic fallback matches Dynare's approach.
4. **Clear feedback**: The method used is recorded in `model_funcs['aux_method']`.
5. **Dynare compatibility**: When analytical fails, the package uses Dynare-style treatment — include all variables in a single system.
