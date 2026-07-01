# Method Supplementary Note

This note records the methodology behind the numerical methods implemented in the project. It is written as a GitHub-readable supplement to the notebooks and source code.

## 1. Monte Carlo Simulation

### 1.1 Ordinary Monte Carlo
For generic SDE models, Monte Carlo often relies on Euler-Maruyama discretisation. In the Black-Scholes case, however, the log-price dynamics are available in closed form, so our implementation simulates the exact lognormal transition.

The Euler--Maruyama discretisation scheme divides the time interval $[0,T]$ into $n$ timesteps of length $\Delta t=\frac{T}{n}$ and generates values of the SDE at time $t_{j+1}$, where $j=0,1,\ldots,n-1$, from known value(s) at $t_j$. For example, a one-dimensional, diffusion based SDE writes:

```math
X_{j+1}
=
X_j + f(t_j, X_{t_j})\,\Delta t
+
g(t_j, X_{t_j})
\left(W_{t_{j+1}} - W_{t_j}\right)
```

### 1.2 Monte Carlo Variance Reduction

#### 1.2.1 Control Variate
The control variate method gives the Monte Carlo estimator extra information by adding a control variable. 

Let $X$ and $Y$ be random variables such that $E(X)$ is known.
Given a sequence $\{(X_j,Y_j)\}_{j=1}^M$ of i.i.d. samples from the joint
distribution of $(X,Y)$, define

```math
Y_j(b)
=
Y_j - b\bigl(X_j - E(X)\bigr),
\qquad
b \in \mathbb{R},
\qquad
j = 1,\ldots,M.
```

The equation should resemble an OLS regression model. The variance-minimising coefficient is:

```math
b^* = \frac{\mathrm{Cov}(X,Y)}{\mathrm{Var}(X)}.
```

Using this value minimises the variance of the adjusted estimator, so the role of $b^*$ is similar to the role of a regression beta coefficient.

Two classic examples from Glasserman (2004) were implemented:
- The discounted stock price was used as a control variate for the valuation of a European call option.
- The geometric Asian option was used as a control variate for the valuation of an arithmetic Asian option.

#### 1.2.2 Antithetic Variate
The antithetic variates method seeks to reduce variance by inducing negative dependence between pairs of simulation replications. The most common implementation is based on the observation that if $U \sim U(0,1)$, then $1-U \sim U(0,1)$. Consequently, paths can be generated using both $U$ and $1-U$ as inputs without altering the underlying distribution of the simulation.

The random variables $U_j$ and $1-U_j$ form an antithetic pair, since a large value of one is associated with a small value of the other. As a result, unusually large or small outputs produced by one path may be offset by the corresponding outputs from its antithetic counterpart. Averaging the two outputs therefore tends to reduce the variance of the estimator while preserving its unbiasedness.

In general, antithetic variates can be constructed through the inverse CDF method. Suppose the simulation requires a random variable $X$ with distribution function $F$. If $U \sim \mathcal{U}(0,1)$, then $X = F^{-1}(U)$ has distribution $F$. Since $1-U$ is also uniformly distributed on $(0,1)$, the antithetic counterpart can be generated as $X^A = F^{-1}(1-U)$.

For the standard normal case, this construction simplifies. Let $\Phi$ denote the standard normal CDF. If $Z = \Phi^{-1}(U)$, then the antithetic normal shock is $Z^A = \Phi^{-1}(1-U)$. By symmetry of the standard normal distribution, $\Phi^{-1}(1-U) = -\Phi^{-1}(U)$, so $Z^A = -Z$.

Therefore, in the Black-Scholes Monte Carlo implementation, explicitly drawing from a uniform distribution and applying the inverse CDF is not needed. Instead, drawing standard normal shocks directly and pairing each shock with its negative is more efficient.

#### 1.2.3 Stratified Sampling
Stratified sampling refers broadly to any sampling mechanism that constrains the fraction of observations drawn from specific subsets, or strata, of the sample space. It is particularly useful when we want to set specific targets to sampling.

In the code, the shocks used in the path simulation are stratified. The unit interval is first divided into $L$ equal-probability strata,

```math
\left[\frac{j}{L}, \frac{j+1}{L}\right), \qquad j=0,\dots,L-1.
```

Within each stratum, draw

```math
U_j \sim \mathcal{U}\left(\frac{j}{L}, \frac{j+1}{L}\right),
```

then transform it into a shock using the inverse CDF.

The stratified estimator averages within each stratum and combines the stratum means:

```math
\hat{V}_{\text{strat}}
=
\sum_{j=1}^{L} p_j \bar{Y}_j.
```

For equal-probability strata, $p_j=1/L$, so

```math
\hat{V}_{\text{strat}}
=
\frac{1}{L}\sum_{j=1}^{L}\bar{Y}_j.
```

And, therefore, the variance of stratified Monte Carlo estimator is

```math
\mathrm{Var}(\hat{V}_{\text{strat}})
=
\sum_{j=1}^{L}
\frac{p_j^2}{n_j}
\mathrm{Var}(Y\mid U\in A_j).
```

The stratified variance formula is used only when each payoff has one clear stratum, i.e., `n_steps = 1`. For multi-step path simulations, the ordinary sample standard error is reported even when shocks are stratified marginally.

### 1.3 Least-squares Monte Carlo
For American options, the key objective is to solve for the stopping rule (i.e., when to terminate the option). The intuition of least-square Monte Carlo ("LSMC") is to simulate the trajectories of the state variable with ordinary Monte Carlo, and then at each time point, compare the conditional expectation of continuation value with the early termination value. This section illustrates the steps of implementing LSMC in the code.

1. Simulate trajectories of the state variable with the ordinary Monte Carlo engine and compute the discount factor for each time interval with $df(r; \Delta t)=e^{-r\Delta t}$.
2. Compute the realised future cashflow from continuing beyond time $t$, then discount it back to time $t$. This represents the value of not exercising at time $t$ and is used as the dependent variable $Y$. To improve efficiency, only the in-the-money paths at time $t$ are included in the regression analysis, since these are the paths where immediate exercise has positive value and must be compared against continuation.
3. Set up a regression with the simulated state variable at time $t$ as independent variable $X$:

```math
    E[Y|X]=\alpha+\beta_1X+\beta_2X^2
```

The fitted conditional expectation is the estimated continuation value. In the code, the default polynomial degree is 2, but the user can choose to increase the degree to improve accuracy.

4. Compare the continuation value and the early termination value. The value of the option at a particular gird point is $\max(\text{termination}, \text{continuation})$.

5. Repeat the above procedure for all time points until we arrive at time $t=0$.

### 1.4 Specific applications

#### 1.4.1 Barrier crossing correction via Brownian Bridge
For ordinary Monte Carlo, discrete points are simulated to generate trajectories of the state variable. If a barrier option is monitored continuously, there is a chance of barrier breach in between the simulated points. This may over(under)value the knock-out(in) options since the option would have been knocked out(in) but it was not monitored in MC. We can make adjustment to the simulation by computing the probability of barrier crossing between time $t$ and time $t+1$, where $t+1=t+\Delta t$. This done by using the concept of Brownian Bridge (without actually constructing one) and the reflection principle of Brownian motion. 

The key idea is to use the Brownian bridge distribution between two simulated log-price points, without explicitly simulating the intermediate path. Conditional on the endpoints $X_t=x$ and $X_{t+\Delta t}=y$, the path between them is distributed as a Brownian bridge: its conditional mean is the straight line connecting $x$ and $y$, and the remaining fluctuation depends on a zero-mean Gaussian bridge process. In this conditional distribution, the original drift term $\mu$ of the log-price process is absorbed by the fixed endpoint $y$. The Brownian bridge itself should not be described as a martingale. The reflection principle is applied to the underlying driftless Gaussian fluctuation around the straight-line bridge.

We are interested in the probability which the barrier is crossed between time $t$ and time $t+1$. This is best explain via an example. For an "up" barrier option, there are 2 subsets(scenarios):

```math
\begin{cases}
P\left(\max_{u\in[t,t+\Delta t]}X_u\ge B
   \mid X_t=x, X_{t+\Delta t}=y\right), & X_{t} < B   \text{and}   X_{t+1} < B,\\
1, & \text{otherwise}.
\end{cases}
```

Recall that if $X$ is a continuous Gaussian martingale and $\tau_B$ is the first hitting time of the barrier level $B$, then the process reflected after $\tau_B$,

```math
X'_t =
\begin{cases}
X_t, & t \leq \tau_B,\\
2B-X_t, & t > \tau_B,
\end{cases}
```

is equal to $X$ in law. It is much harder to evaluate directly the density of paths that travel from $X_t$ to $B$ and then back to $X_{t+\Delta t}$. The reflection principle replaces this path-dependent event with an unrestricted transition from $X_t$ to the reflected endpoint $X'_{t+\Delta t}=2B-X_{t+\Delta t}$.

We want to compute the conditional probability that the process hits the barrier
between the two monitoring dates:

```math
P\left(
\max_{u\in[t,t+\Delta t]}X_u \geq B
\mid X_t=x,\ X_{t+\Delta t}=y
\right).
```

Let

```math
\tau_B=\inf\{u\geq t:X_u=B\}
```

be the first hitting time of the barrier. The event of interest is

```math
\{\tau_B\leq t+\Delta t\}.
```

Using conditional probability in density form,

```math
P\left(
\tau_B\leq t+\Delta t
\mid X_t=x,\ X_{t+\Delta t}=y
\right)
=
\frac{
P\left(\tau_B\leq t+\Delta t,\ X_{t+\Delta t}=y
\mid X_t=x\right)
}{
P\left(X_{t+\Delta t}=y\mid X_t=x\right)
}.
```

The denominator is the unrestricted transition density from $x$ to $y$:

```math
p(\Delta t,x,y)
=
\frac{1}{\sigma\sqrt{2\pi\Delta t}}
\exp\left(
-\frac{(y-x)^2}{2\sigma^2\Delta t}
\right).
```

For the numerator, we use the reflection principle. Any path starting at $x<B$
that hits $B$ before $t+\Delta t$ and ends at $y<B$ can be reflected after its
first hitting time $\tau_B$. The reflected terminal value is

```math
X'_{t+\Delta t}=2B-y.
```

Therefore, paths from $x$ to $y$ that hit $B$ are in one-to-one correspondence
with unrestricted paths from $x$ to $2B-y$. Hence,

```math
P\left(\tau_B\leq t+\Delta t,\ X_{t+\Delta t}=y
\mid X_t=x\right)
=
p(\Delta t,x,2B-y).
```

Thus,

```math
P\left(
\tau_B\leq t+\Delta t
\mid X_t=x,\ X_{t+\Delta t}=y
\right)
=
\frac{p(\Delta t,x,2B-y)}{p(\Delta t,x,y)}.
```

Substituting the Gaussian transition densities,

```math
P\left(
\tau_B\leq t+\Delta t
\mid X_t=x,\ X_{t+\Delta t}=y
\right)
=
\frac{
\exp\left(
-\frac{((2B-y)-x)^2}{2\sigma^2\Delta t}
\right)
}{
\exp\left(
-\frac{(y-x)^2}{2\sigma^2\Delta t}
\right)
}.
```

After simplification, 

```math
P\left(
\tau_B\leq t+\Delta t
\mid X_t=x,\ X_{t+\Delta t}=y
\right)
=
\exp\left(
-\frac{2(B-x)(B-y)}{\sigma^2\Delta t}
\right)=\hat{p}_i^{\text{hit}}.
```

Finally, the conditional expectation of a knock-out barrier option with payoff function $k(\hat{X})$ is:

```math
\begin{aligned}
&\mathbb{E}\left[
k(\hat{X}(n))
\prod_{i=0}^{n-1}
\mathbf{1}_{\{\tau_B^{(i)} > t_{i+1}\}}
\;\middle|\;
\hat{X}(0),\hat{X}(1),\ldots,\hat{X}(n)
\right] \\
&= k(\hat{X}(n))
\prod_{i=0}^{n-1}
P\left(
\tau_B^{(i)} > t_{i+1}
\mid
\hat{X}(i),\hat{X}(i+1)
\right) \\
&= k(\hat{X}(n))
\prod_{i=0}^{n-1}
\left(1-p_i^{\mathrm{hit}}\right).
\end{aligned}
```

## 2. Finite Difference Method for PDEs and PIDEs

### 2.1 The Black--Scholes PDE
The Black--Scholes PDE decomposes into a *time* component and a
*spatial* component:

```math
  \underbrace{\frac{\partial f}{\partial t}}_{\text{Time}}
  + \underbrace{\frac{\partial f}{\partial S}(r-q)S
    + \frac{1}{2}\frac{\partial^{2}f}{\partial S^{2}}\sigma^{2}S^{2}
    - rf}_{\text{Spatial}}
  = 0.
```

#### 2.1.1 Transform Spatial Terms into a 3-State Equation

Denote $V_{\tau,i}=f(T-\tau,S_i)$, where $\tau=T-t$ and define the spatial operator

```math
  \mathcal{L}_\tau(V_{\tau,i}) = a_i\,V_{\tau,i-1} + b_i\,V_{\tau,i} + c_i\,V_{\tau,i+1}.
```

Using central finite differences for the spatial derivatives:

```math
  \frac{\partial f}{\partial S} \approx \frac{V_{i+1}-V_{i-1}}{2\Delta S},
  \qquad
  \frac{\partial^{2}f}{\partial S^{2}} \approx
    \frac{V_{i+1}-2V_{i}-V_{i-1}}{\Delta S^{2}}.
```

Substituting into the spatial part of the Black--Scholes PDE:

```math
\begin{aligned}
  \mathcal{L}_\tau(V_{i})
  &= \frac{V_{i+1}-V_{i-1}}{2\Delta S}(r-q)S
     + \frac{1}{2}\cdot\frac{V_{i+1}-2V_{i}-V_{i-1}}{\Delta S^{2}}\,\sigma^{2}S^{2}
     - rV_{i} \\[6pt]
  &= (r-q)\frac{S}{2\Delta S}(V_{i+1}-V_{i-1})
     + \frac{\sigma^{2}S^{2}}{2\Delta S^{2}}(V_{i+1}-2V_{i}-V_{i-1})
     - rV_{i}.
\end{aligned}
```

Collecting terms by node gives:

```math
  \mathcal{L}_\tau(V_{i})
  = \underbrace{\left[\frac{S^{2}\sigma^{2}}{2\Delta S^{2}}
      - (r-q)\frac{S}{2\Delta S}\right]}_{a_i}\!V_{i-1}
  + \underbrace{\left[-r - \frac{S^{2}\sigma^{2}}{\Delta S^{2}}\right]}_{b_i}\!V_{i}
  + \underbrace{\left[\frac{S^{2}\sigma^{2}}{2\Delta S^{2}}
      + (r-q)\frac{S}{2\Delta S}\right]}_{c_i}\!V_{i+1}.
```

#### 2.1.2 Variable Transformation: Time Domain

The PDE is solved *backwards* from the terminal condition, so $t$ is
descending.  It is more natural to work with *time to maturity*
$\tau = T - t$, which is ascending.  Since

```math
  \frac{\partial f}{\partial t} = -\frac{\partial f}{\partial\tau},
```

the PDE becomes

```math
  -\frac{\partial f}{\partial\tau} + \mathcal{L}_\tau(V_{i}) = 0
   \Longrightarrow 
  \frac{\partial f}{\partial\tau} = \mathcal{L}_\tau(V_{i}).
```

Discretising in $\tau$:

```math
  \frac{\partial f}{\partial\tau}
  \approx \frac{V_{\tau+1,i}-V_{\tau,i}}{\Delta t}
   \Longrightarrow 
  \frac{V_{\tau+1,i}-V_{\tau,i}}{\Delta t} = \mathcal{L}_\tau(V_{i}).
```

#### 2.1.3 Choose a Time-Stepping Scheme
**Explicit scheme  ($\theta=0$)**

Reference time point at $\tau$:

```math
      \frac{V_{\tau+1,i}-V_{\tau,i}}{\Delta t} = \mathcal{L}_\tau(V_{\tau,i}).
```

**Implicit scheme  ($\theta=1$)**

Reference time point at $\tau+1$:

```math
      \frac{V_{\tau+1,i}-V_{\tau,i}}{\Delta t} = \mathcal{L}_\tau(V_{\tau+1,i}).
```

**Crank--Nicolson  ($\theta=0.5$)**

Reference time point at $\tau+\tfrac{1}{2}$:

```math
      \frac{V_{\tau+1,i}-V_{\tau,i}}{\Delta t}
      = \tfrac{1}{2}\!\left[\mathcal{L}_\tau(V_{\tau,i})+\mathcal{L}_\tau(V_{\tau+1,i})\right].
```

Letting the reference time point be $\tau+\theta$ yields the **general $\theta$-scheme**:

```math
  \frac{V_{\tau+1,i}-V_{\tau,i}}{\Delta t}
  = (1-\theta)\,\mathcal{L}_\tau(V_{\tau,i}) + \theta\,\mathcal{L}_\tau(V_{\tau+1,i}).
```

Rearranging the theta-scheme:

```math
  V_{\tau+1,i} - \Delta t\,\theta\,\mathcal{L}_\tau(V_{\tau+1,i})
  = V_{\tau,i} + \Delta t(1-\theta)\,\mathcal{L}_\tau(V_{\tau,i}).
```

Substituting $\mathcal{L}_\tau(V_{\tau,i}) = a_i\,V_{\tau,i-1}+b_i\,V_{\tau,i}+c_i\,V_{\tau,i+1}$
on both sides:

**LHS** (unknowns at $\tau+1$):

```math
  (-\Delta t\,\theta\,a_i)\,V_{\tau+1,i-1}
  + (1-\Delta t\,\theta\,b_i)\,V_{\tau+1,i}
  + (-\Delta t\,\theta\,c_i)\,V_{\tau+1,i+1}
```

**RHS** (known values at $\tau$):

```math
  (\Delta t(1-\theta)\,a_i)\,V_{\tau,i-1}
  + (1+\Delta t(1-\theta)\,b_i)\,V_{\tau,i}
  + (\Delta t(1-\theta)\,c_i)\,V_{\tau,i+1}
```

Assembling across all interior nodes $i = 1,\dots,N-1$, the system takes the
matrix form

```math
  A\,\mathbf{V}_{\tau+1} = B\,\mathbf{V}_{\tau},
```

where $\mathbf{V}_{\tau} = (V_{\tau,1},\dots,V_{\tau,N-1})^{\!\top}$ and the
two $(N-1)\times(N-1)$ tridiagonal matrices are

```math
  A =
  \begin{pmatrix}
    1-\Delta t\theta b_i & -\Delta t\theta c_i      &                  &        \\
    -\Delta t\theta a_i  & 1-\Delta t\theta b_i     & -\Delta t\theta c_i   &        \\
                    & \ddots              & \ddots           & \ddots \\
                    &                     & -\Delta t\theta a_i   & 1-\Delta t\theta b_i
  \end{pmatrix},
```

```math
  B =
  \begin{pmatrix}
    1+\Delta t(1-\theta)b_i   & \Delta t(1-\theta)c_i   &                      &        \\
    \Delta t(1-\theta)a_i     & 1+\Delta t(1-\theta)b_i & \Delta t(1-\theta)c_i     &        \\
                         & \ddots             & \ddots               & \ddots \\
                         &                    & \Delta t(1-\theta)a_i     & 1+\Delta t(1-\theta)b_i
  \end{pmatrix}.
```

Boundary corrections (from 2.1.4) are added to the first and last entries of
$B\,\mathbf{V}_{\tau}$ as needed.

#### 2.1.4 Impose Boundary Conditions

If a boundary (e.g. $S=0$ or $S=S_{\max}$) is hit, the state value is no
longer evaluated via the PDE.  Instead, it is replaced by an **analytical formulation** appropriate to the payoff.

For knock-out barrier options, one more consideration is needed when determining the boundaries. When the barrier is hit, the option deactivates and the payoff is zero (assuming no rebate). Therefore, we can further truncate the grid space with lower barrier payoff = 0 for down-and-out options and upper boundary payoff = 0 for up-and-out options at the barrier level. For knock-in options, the option becomes a vanilla option once the barrier is hit so the grid truncation is not as trivial. Therefore, the code only solves for knock-out options and uses the parity relationship (i.e., knock-in = vanilla - knock-out) to evaluate knock-in options for numerical efficiency.

#### 2.1.5 Projection
For American (also applicable to Bermudan) options, the early exercise decision must be considered in the finite difference scheme. This is done by projecting the values of each grid points to the function $\max(\text{termination}, \text{continuation})$, where the option holder must compare whether the immediate payoff (i.e., termination) or the expected continuation value is more valuable.

#### 2.1.6 Solve the Tridiagonal Matrix

The tridiagonal system is solved at each time step for
$\mathbf{V}_{\tau+1}$:

```math
  \mathbf{V}_{\tau+1} = A^{-1}\!\left(B\,\mathbf{V}_{\tau}
    + \mathbf{g}_{\tau,\tau+1}\right),
```

where $\mathbf{g}_{\tau,\tau+1}$ collects any boundary correction terms.  Because $A$ is tridiagonal, the solve is carried out in $\mathcal{O}(N)$ operations via the **Thomas algorithm**.
