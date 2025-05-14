# Portfolio Optimization (Markowitz)

## Overview

This project is an application of **Modern Portfolio Theory (MPT)** concepts to construct an optimal portfolio using French stocks, based on historical data and portfolio analysis tools.

## Features

- Download historical price data for a set of French stocks.
- Calculate **annualized returns**, **annualized volatilities**, **Sharpe ratio**, and **correlation matrix**
- Build the **efficient frontier** (return vs risk) with positioning of key portfolios.
    - Implement the Markowitz model to identify:
        - The **Maximum Sharpe Ratio (MSR)** portfolio.
        - The **Global Minimum Variance (GMV)** portfolio.

- Compare optimized portfolios with an **equally-weighted (EW)** portfolio used as a benchmark.
- Visualize the results to better understand diversification and optimization.

---

### Project Structure

**Function module file**

- Used to centralize analysis functions

**Optimization notebook:**  

- Used to compute optimal allocations based on expected returns and covariances

