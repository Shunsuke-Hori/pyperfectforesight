# Initial Guess

The perfect foresight solver requires an initial guess `X0` — a `(T, n)` array whose rows are the starting estimates for each period's endogenous variables. A good initial guess reduces Newton iterations and avoids convergence failure, especially for large shocks.

## `make_initial_guess`

```{eval-rst}
.. autofunction:: pyperfectforesight.make_initial_guess
   :no-index:
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

### Combining with `Model.solve`

```python
import numpy as np
from pyperfectforesight import Model, make_initial_guess

m = Model()
m.endog("c k")
m.params("alpha beta")

eq_euler = m.c[0]**(-1) - m.beta * m.alpha * m.k[0]**(m.alpha - 1) * m.c[1]**(-1)
eq_kacc  = m.k[0] - m.k[-1]**m.alpha + m.c[0]
m.build([eq_euler, eq_kacc])

PARAMS = {m.alpha: 0.36, m.beta: 0.99}
ss = m.steady_state(PARAMS)

T = 100
k_neg1 = np.array([ss[1] * 1.1])

# Build initial guess: exponential path from perturbed SS back to SS
ss_perturbed = np.array([ss[0], ss[1] * 1.1])   # approximate period-0 values
X0 = make_initial_guess(T, ss_initial=ss_perturbed, ss_terminal=ss,
                        method='exponential', decay=0.9)

sol = m.solve(T, PARAMS, X0=X0, initial_state=k_neg1, endval=ss)
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
