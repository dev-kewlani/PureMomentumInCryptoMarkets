# Pure Momentum Cryptocurrency Strategy

This repository contains a pair of Python scripts that implement and analyze a pure momentum cryptocurrency trading strategy using QuantConnect’s Lean engine framework. The repository is organized as follows:

- **research.py**  
  A research script that:
  - Uses QuantConnect’s `QuantBook` to pull hourly cryptocurrency data.
  - Calculates lagged returns (1 to 48 hours) and generates several visualizations including a heatmap of regression coefficients, time evolution of the lag 24 effect, a trading strategy simulation, and a comparative analysis across multiple cryptocurrencies.
  - Outputs key performance metrics such as the lag-24 coefficient, strategy annualized return, and Sharpe ratio.
  
- **main.py**  
  A Lean algorithm script that:
  - Implements the pure momentum strategy as a Lean algorithm.
  - Contains a custom alpha model (`PureMomentumAlphaModel`) that triggers long signals based on a delayed rate-of-change (ROC).
  - Uses a portfolio construction model (`NegativeReturnRankingPortfolioConstructionModel`) to allocate cash based on recent 24-hour performance.
  - Includes a risk management model (`SimpleStopTakeRiskManagementModel`) which liquidates positions if stop loss or take profit levels are reached.
  
- **README.md**  
  This file explains the project, the purpose behind it, and how to run both the research and algorithm code locally.

## Prerequisites

To run this project, you will need to setup an account on QuantConnect and create a new project with these two files:
- **main.py**
- **research.py/ipynb**