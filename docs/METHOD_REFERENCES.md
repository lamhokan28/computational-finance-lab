# Method References

This is a friendly map from methods in the project to the sources. It is not meant to be a formal bibliography only; it should help readers find the methodology quickly. The references selected are not just for formulation reference but for learning the intuitions behind the methodologies employed.

## Black-Scholes Analytic Formulas

1. European vanilla option
    - Shreve, S. E. (2004). Stochastic calculus for finance II: Continuous-time models. Springer (Chapter 1-5)

2. Geometric Asian option
    - Advanced Stochastic Modelling course notes by Prof. Aleš Černý
    - Shreve, S. E. (2004). Stochastic calculus for finance II: Continuous-time models. Springer (Chapter 7.5)
    - Wikipedia

3. Single barrier option
    - Advanced Stochastic Modelling course notes by Prof. Aleš Černý
    - Shreve, S. E. (2004). Stochastic calculus for finance II: Continuous-time models. Springer (Chapter 3.7, 7.3)
    - Rubinstein, M., & Reiner, E. (1991). Breaking down the barriers. Risk, 4(8), 28–35.

Project location:

- `src/derivlab/pricers/analytic/black_scholes.py`

## Monte Carlo Simulation

1. Euler scheme
    - Simulation Techniques and Financial Modelling course notes by Prof. Laura Ballotta
    - Mikosch, T. (1998). Elementary stochastic calculus with finance in view (Section 3.4)
    - Glasserman, P. (2004). Monte Carlo Methods in Financial Engineering (Chapter 1-3)

2. Variance reduction techniques
    - Simulation Techniques and Financial Modelling course notes by Prof. Laura Ballotta
    - Glasserman, P. (2004). Monte Carlo Methods in Financial Engineering (Chapter 4)

3. Least-square Monte Carlo
    - Longstaff, F. A. and Schwartz, E. S. (2001). Valuing American Options by Simulation: A Simple Least-Squares Approach.

4. Barrier crossing correction via Brownian Bridge
    - Shreve, S. E. (2004). Stochastic calculus for finance II: Continuous-time models. Springer (Section 3.7)
    - Glasserman, P. (2004). Monte Carlo Methods in Financial Engineering.
    - Glasserman, Paul & Staum, Jeremy. (2000). Conditioning on One-Step Survival for Barrier Option Simulations. Operations Research. 49. 10.1287/opre.49.6.923.10018. 
    - Broadie, M., Glasserman, P. and Kou, S. G. (1997). A Continuity Correction for Discrete Barrier Options.

Project location:

- `src/derivlab/pricers/monte_carlo/engine.py`
- `src/derivlab/pricers/monte_carlo/variance_reduction.py`
- `src/derivlab/pricers/monte_carlo/lsmc.py`
- `src/derivlab/pricers/monte_carlo/brownian_bridge.py`

## Finite Difference PDE Methods

1. Explicit Euler/Implicit Euler/Crank-Nicolson Schemes under Black-Scholes PDE
    - Advanced Stochastic Modelling course notes by Prof. Aleš Černý

2. Thomas Algorithm
    - Youtube

Project location:

- `src/derivlab/pricers/pde/black_scholes_pde.py`
- `src/derivlab/pricers/pde/solver.py`