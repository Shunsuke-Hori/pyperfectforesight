# Initial Guess

The perfect foresight solver requires an initial guess `X0` — a `(T, n)` array whose rows are the starting estimates for each period's endogenous variables. A good initial guess reduces Newton iterations and avoids convergence failure, especially for large shocks.

## `make_initial_guess`

```{eval-rst}
.. autofunction:: pyperfectforesight.make_initial_guess
```

## Methods

`make_initial_guess` supports three interpolation methods between `ss_initial` (the starting point) and `ss_terminal` (the terminal steady state):

`linear` (default)
: Linearly interpolates from `ss_initial` at $t=0$ to `ss_terminal` at $t=T-1$. This matches Dynare's default `perfect_foresight_setup` behaviour when both `initval` and `endval` are supplied.

`exponential`
: Geometric convergence: $x(t) = \texttt{ss\_terminal} + (\texttt{ss\_initial} - \texttt{ss\_terminal}) \cdot \texttt{decay}^t$. The path closes most of the gap early and flattens near `ss_terminal`, mimicking the saddle-path dynamics typical of DSGE models. The `decay` parameter (default `0.9`) controls the convergence speed — smaller values (e.g. `0.5`) close the gap faster.

`constant`
: Returns `ss_terminal` repeated for all `T` periods. Equivalent to the common idiom `np.tile(ss, (T, 1))`.

## Usage examples

### Replacing `np.tile`

The simplest starting point is a constant path at the terminal steady state. `make_initial_guess` with `method='constant'` is a drop-in replacement:

```python
import numpy as np
from pyperfectforesight import make_initial_guess

# Old idiom
X0 = np.tile(ss, (T, 1))

# Equivalent with make_initial_guess
X0 = make_initial_guess(T, ss_initial=ss, ss_terminal=ss, method='constant')
```

### Linear interpolation (transition between two steady states)

When you know the economy starts at `ss_old` and ends at `ss_new`, a linear interpolation is a natural warm start:

```python
X0 = make_initial_guess(T, ss_initial=ss_old, ss_terminal=ss_new)
# method='linear' is the default
```

### Exponential interpolation for saddle-path models

For models with saddle-path dynamics, the exponential method often gives a better warm start because the true solution also closes most of the gap early:

```python
X0 = make_initial_guess(
    T,
    ss_initial=ss,
    ss_terminal=ss,
    method='exponential',
    decay=0.85,   # faster convergence than default 0.9
)
```

### Combining with `solve_perfect_foresight`

```python
import numpy as np
from pyperfectforesight import (
    v, process_model, solve_perfect_foresight, make_initial_guess,
)

ALPHA, BETA = 0.36, 0.99

eq_euler = v("c", 0)**(-1) - BETA * ALPHA * v("k", 0)**(ALPHA-1) * v("c", 1)**(-1)
eq_kacc  = v("k", 0) - v("k", -1)**ALPHA + v("c", 0)

vars_dyn = ["c", "k"]
model_funcs = process_model([eq_euler, eq_kacc], vars_dyn)

K_SS = (ALPHA * BETA) ** (1 / (1 - ALPHA))
C_SS = K_SS**ALPHA - K_SS
ss = np.array([C_SS, K_SS])

T = 100
k_neg1 = np.array([K_SS * 1.1])

# Build initial guess: exponential path from perturbed SS back to SS
ss_perturbed = np.array([C_SS, K_SS * 1.1])   # approximate period-0 values
X0 = make_initial_guess(T, ss_initial=ss_perturbed, ss_terminal=ss,
                        method='exponential', decay=0.9)

sol = solve_perfect_foresight(
    T, X0, {}, ss, model_funcs, vars_dyn,
    initial_state=k_neg1,
    stock_var_indices=[1],
)
print(f"Converged: {sol.success}")
```

## The `decay` parameter

The `decay` parameter only affects `method='exponential'`. It must be in $(0, 1)$:

| `decay` | Convergence speed | Notes |
|---|---|---|
| `0.5` | Fast — half the gap closed each period | Good for very persistent models |
| `0.85` | Medium-fast | Reasonable default for most DSGE models |
| `0.9` | Medium (default) | Matches typical AR(1) persistence |
| `0.99` | Slow — nearly linear | Use `method='linear'` instead |

Values outside $(0, 1)$ raise `ValueError`.
