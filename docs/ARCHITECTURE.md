# Project Architecture

This file records the project structure and the role of each layer. It is meant to be a quick map before reading the source code.

## High-Level Flow

```text
Product -> Model -> Pricer -> PricingResult -> Notebook
```

- `Product`: defines the derivative payoff and contract identity.
- `Model`: defines the market dynamics and model-specific capabilities.
- `Pricer`: applies a numerical or analytic method to one product under one model.
- `PricingResult`: stores price, standard error, runtime, method name, and diagnostics.
- `Notebook`: teaches the theory, runs experiments, and discusses numerical behaviour.

## Source Tree

```text
src/derivlab/
├── core/
│   ├── market.py                   # Market inputs: spot, rate, dividend, volatility
│   ├── results.py                  # PricingResult container
│   ├── enums.py                    # Shared option/model/method enums
│   ├── validation.py               # Parameter parsing and validation helpers
│   ├── random.py                   # Random number generator helper
│   └── distributions.py            # Distribution wrappers, currently normal CDF/PDF/PPF
├── products/
│   ├── european.py                 # European call/put product
│   ├── american.py                 # American option product
│   ├── asian.py                    # Arithmetic/geometric Asian product
│   └── barrier.py                  # Single-barrier product identity and path logic
├── models/
│   ├── black_scholes.py            # Black-Scholes model and path simulation
└── pricers/
    ├── analytic/
    │   └── black_scholes.py        # Closed-form BS prices, European, geometric Asian, single barriers
    ├── monte_carlo/
    │   ├── engine.py               # Plain Monte Carlo
    │   ├── variance_reduction.py   # controlled variate, antithetic variate and stratified sampling
    │   ├── lsmc.py                 # Longstaff-Schwartz American pricing
    │   └── brownian_bridge.py      # Brownian bridge correction for barrier pricing
    └── pde/
        ├── black_scholes_pde.py    # solving Black-Scholes pde and construct tridiagonal matrix
        ├── boundary_conditions.py  # helper to define boundaries for PDE
        ├── grids.py                # setting for PDE grids
        └── solver.py               # tridiagonal matrix solver
```

## Notebook Layer

```text
notebooks/
├── 01_european_options.ipynb
├── 02_american_options.ipynb
├── 03_asian_options.ipynb
└── 04_barrier_options.ipynb

```

The notebooks are option-centric. Each notebook starts with the product, then compares relevant methods for that product.

## Documentation Layer
- Result discussion.
- Code architecture for clear workflow presentation and quick reference.
- Supplementary note for methodologies of numerical schemes implemented.
- Reference list.
