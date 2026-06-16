# Computational Finance Lab

This research lab studies various numerical methods for pricing four types of options: European vanilla, American, Asian and Barrier options. Together, these products provide a comprehensive framework for investigating key challenges in derivative pricing, including early exercise decisions, lifetime path dependency and point-wise path dependency. 

The first phase of the study begins with the Black-Scholes model, which remains one of the most influential frameworks in computational finance. Despite its simplifying assumptions, the model provides elegant closed-form solutions and embeds many of the fundamental concepts of continuous-time finance, including risk-neutral valuation, replication and no-arbitrage pricing. However, as recorded in the literature, Black-Scholes has obvious limitations - failing to capture stylised facts of underlying process such as fat tails, volatility and time-varying interest rates. Therefore, the later phases of this study break the assumptions of Black-Scholes one by one and observe the impact and complications to the derivative pricing. The final objective is to have a complete toolkit to build a custom hybrid stochastic model powerful enough to price complex derivatives such as autocallables and target redemptions. 

Monte Carlo simulation, partial differential equation (PDE) and numerical integration (coming soon) are important numerical approaches in quantitative finance and they will be key tools to evaluate option prices throughout this study.
- PDE methods solve directly for the option value function and are highly efficient and accurate for low-dimensional problems, particularly for products with early exercise features. 
- Monte Carlo methods, in contrast, estimate the risk-neutral expectation through simulation and are naturally suited to path-dependent products and high-dimensional models.
- Numerical integration is particularly useful when characteristic functions are known, enabling efficient Fourier-based pricing and the numerical treatment of PIDEs in models with jumps or stochastic volatility.

## Project Map

| Section | Description |
|---|---|
| [Result Discussion](docs/RESULT_DISCUSSION.md) | Main numerical findings and discussion |
| [Method Supplementary Note](docs/METHOD_SUPPLEMENTARY_NOTE.md) | Explanation of methodologies implemented |
| [Method References](docs/METHOD_REFERENCES.md) | Where each methodology comes from |
| [Architecture](docs/ARCHITECTURE.md) | Code structure and design flow |

## Notebooks

| Notebook | Numerical Methods Tested |
|---|---|
| [`01_european_options.ipynb`](notebooks/01_european_options.ipynb) | MC(with control, antithetic and stratified), PDE |
| [`02_american_options.ipynb`](notebooks/02_american_options.ipynb) | LSMC, PDE |
| [`03_asian_options.ipynb`](notebooks/03_asian_options.ipynb) | MC(with control variates) |
| [`04_barrier_options.ipynb`](notebooks/04_barrier_options.ipynb) | MC, barrier corrected MC, PDE |