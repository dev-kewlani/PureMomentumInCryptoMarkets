# research.py
# This script performs a pure momentum analysis on cryptocurrency hourly data
# using QuantConnect's Research API (QuantBook). It calculates lag effects,
# generates several visualization charts, and aggregates key metrics across coins.

# Ensure required libraries are imported:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

# Instantiate the QuantBook instance for Lean research.
qb = QuantBook()

def enhanced_coin_analysis(ticker):
    # Set start and end dates correctly
    start = datetime(2019, 1, 1)
    end = datetime(2024, 12, 31)
    qb.SetStartDate(start)  # Ensure start is set instead of the end date
    # Request hourly data from Coinbase (using minute resolution, then aggregating to hour)
    symbol = qb.AddCrypto(ticker, Resolution.MINUTE, Market.COINBASE).Symbol
    data = qb.History(symbol, start, end, Resolution.HOUR)
    
    # Select necessary columns and reset index
    data = data[['close', 'volume']]
    data.reset_index(inplace=True)
    if 'symbol' in data.columns:
        data.drop(columns='symbol', inplace=True)

    # Calculate hourly simple returns (not log returns)
    data['1_period_return'] = data['close'] / data['close'].shift(1) - 1
    data['time'] = pd.to_datetime(data['time'])
    
    # Create time features
    data['dayofweek'] = data['time'].dt.dayofweek
    data['dayofyear'] = data['time'].dt.dayofyear
    data['day'] = data['time'].dt.day
    data['month'] = data['time'].dt.month
    data['year'] = data['time'].dt.year
    data['hour'] = data['time'].dt.hour
    data['quarter'] = data['time'].dt.quarter

    # Create lagged hourly returns for lags 1 to 48 and set the target as the next hour's return
    for lag in range(1, 49):
        data[f'{lag+1}_period_return'] = data['1_period_return'].shift(lag)
    data['target'] = data['1_period_return'].shift(-1)
    
    # Drop rows with missing values
    data.dropna(inplace=True)

    # VISUALIZATION 1: Lag Coefficient Heatmap with Significance
    # Run individual regressions for each lag and store coefficients and p-values
    results = {}
    p_values = {}
    for lag in range(1, 49):
        X = sm.add_constant(data[f'{lag}_period_return'])
        model = sm.OLS(data['target'], X)
        result = model.fit()
        results[lag] = result.params[1]  # Store the coefficient
        p_values[lag] = result.pvalues[1]  # Store the p-value

    plt.figure(figsize=(14, 8))
    ax = plt.subplot2grid((2, 3), (0, 0), colspan=2)
    colors = ['green' if p_values[lag] < 0.05 else 'lightgray' for lag in range(1, 49)]
    bars = ax.bar(range(1, 49), [results[lag] for lag in range(1, 49)], color=colors)
    if 24 in results:
        ax.bar(24, results[24], color='red')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('Lag (hours)')
    ax.set_ylabel('Coefficient')
    ax.set_title(f'Effect of Lagged Hourly Returns on Next Hour Return for {ticker}')
    ax.set_xticks(range(0, 49, 6))
    ax.grid(axis='y', alpha=0.3)
    
    # Add significance legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Significant (p<0.05)'),
        Patch(facecolor='lightgray', label='Not Significant'),
        Patch(facecolor='red', label='Lag 24 (Focus of Study)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # VISUALIZATION 4: Lag 24 Effect by Year (Time Evolution)
    ax4 = plt.subplot2grid((2, 3), (1, 1), colspan=2)
    year_effects = {}
    years = sorted(data['year'].unique())
    for year in years:
        year_data = data[data['year'] == year]
        if len(year_data) > 100:
            X = sm.add_constant(year_data['24_period_return'])
            model = sm.OLS(year_data['target'], X)
            result = model.fit()
            year_effects[year] = result.params[1]
    ax4.bar(year_effects.keys(), year_effects.values())
    ax4.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Lag 24 Coefficient')
    ax4.set_title('Evolution of Lag 24 Effect Over Time')
    ax4.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.suptitle(f'Pure Momentum Analysis for {ticker}', fontsize=16, y=1.02)
    plt.show()

    # VISUALIZATION 5: Trading Strategy Simulation
    plt.figure(figsize=(14, 7))
    data['signal'] = np.where(data['24_period_return'] > 0, -1, 1)
    data['strategy_return'] = data['signal'] * data['target']
    data['cum_target'] = ((1 + data['target']).cumprod() - 1) * 100
    data['cum_strategy'] = (1 + data['strategy_return']).cumprod() - 1
    plt.plot(data['time'], data['cum_target'], label=f'{ticker} Buy & Hold', alpha=0.7)
    plt.plot(data['time'], data['cum_strategy'], label='Pure Momentum Strategy', linewidth=1.5)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.2)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.title(f'Pure Momentum Strategy vs Buy & Hold for {ticker}')
    plt.legend()
    plt.grid(alpha=0.3)
    strategy_return = data['strategy_return'].mean() * 24 * 365 * 100
    strategy_vol = data['strategy_return'].std() * np.sqrt(24 * 365) * 100
    strategy_sharpe = strategy_return / strategy_vol if strategy_vol > 0 else 0
    buy_hold_return = data['target'].mean() * 24 * 365 * 100
    buy_hold_vol = data['target'].std() * np.sqrt(24 * 365) * 100
    buy_hold_sharpe = buy_hold_return / buy_hold_vol if buy_hold_vol > 0 else 0
    plt.annotate(f'Strategy: Return={strategy_return:.1f}%, Sharpe={strategy_sharpe:.2f}',
                 xy=(0.02, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    plt.annotate(f'Buy & Hold: Return={buy_hold_return:.1f}%, Sharpe={buy_hold_sharpe:.2f}',
                 xy=(0.02, 0.89), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    plt.show()
    
    # VISUALIZATION 6: Comparative Analysis Across Multiple Lags
    plt.figure(figsize=(14, 7))
    key_lags = [1, 24, 25, 48]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for i, lag in enumerate(key_lags):
        data[f'{lag}_period_quintile'] = pd.qcut(data[f'{lag}_period_return'], 2, labels=False)
        quintile_returns = data.groupby(f'{lag}_period_quintile')['target'].mean() * 100 
        axes[i].bar(quintile_returns.index, quintile_returns)
        axes[i].set_xlabel(f'Lag {lag} Return Quintile')
        axes[i].set_ylabel('Average Next-Hour Return (%)')
        axes[i].set_title(f'Lag {lag}: Return Predictability')
        axes[i].grid(axis='y', alpha=0.3)
        axes[i].set_xticks(range(2))
        axes[i].set_xticklabels(['Low', 'High'])
    plt.tight_layout()
    plt.suptitle(f'Return Predictability Across Key Lags for {ticker}', fontsize=16, y=1.02)
    plt.show()
    
    # Return key statistics for further comparison
    return {
        'ticker': ticker,
        'lag_24_coef': results.get(24, 0),
        'lag_24_pvalue': p_values.get(24, 1),
        'strategy_return': strategy_return,
        'strategy_sharpe': strategy_sharpe
    }

# Analyze a list of tickers (example uses the 7 most liquid cryptocurrencies)
tickers = ['BTCUSD', 'ETHUSD', 'ADAUSD', 'DOGEUSD', 'SOLUSD', 'LTCUSD', 'DASHUSD']
all_results = []

for ticker in tickers:
    try:
        result = enhanced_coin_analysis(ticker)
        all_results.append(result)
        print(f"Completed analysis for {ticker}")
    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")

# VISUALIZATION 7: Cross-Coin Comparison
if all_results:
    comparison_df = pd.DataFrame(all_results)
    plt.figure(figsize=(14, 10))
    
    # Plot lag 24 coefficient across coins
    ax1 = plt.subplot(2, 2, 1)
    bars = ax1.bar(comparison_df['ticker'], comparison_df['lag_24_coef'])
    for i, p_value in enumerate(comparison_df['lag_24_pvalue']):
        bars[i].set_color('green' if p_value < 0.05 else 'lightgray')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.set_xlabel('Cryptocurrency')
    ax1.set_ylabel('Lag 24 Coefficient')
    ax1.set_title('Pure Momentum Effect Strength Across Cryptocurrencies')
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot strategy returns
    ax2 = plt.subplot(2, 2, 2)
    ax2.bar(comparison_df['ticker'], comparison_df['strategy_return'])
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Cryptocurrency')
    ax2.set_ylabel('Annualized Return (%)')
    ax2.set_title('Strategy Performance Across Cryptocurrencies')
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot Sharpe ratios
    ax3 = plt.subplot(2, 2, 3)
    ax3.bar(comparison_df['ticker'], comparison_df['strategy_sharpe'])
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.set_xlabel('Cryptocurrency')
    ax3.set_ylabel('Sharpe Ratio')
    ax3.set_title('Risk-Adjusted Performance Across Cryptocurrencies')
    ax3.grid(axis='y', alpha=0.3)
    
    # Scatter plot: Coefficient vs. Annualized Return
    ax4 = plt.subplot(2, 2, 4)
    ax4.scatter(comparison_df['lag_24_coef'], comparison_df['strategy_return'])
    for i, ticker in enumerate(comparison_df['ticker']):
        ax4.annotate(ticker, 
                     (comparison_df['lag_24_coef'].iloc[i], comparison_df['strategy_return'].iloc[i]),
                     xytext=(5, 5), textcoords='offset points')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax4.set_xlabel('Lag 24 Coefficient')
    ax4.set_ylabel('Annualized Return (%)')
    ax4.set_title('Relationship Between Effect Strength and Performance')
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Cross-Cryptocurrency Comparison of Pure Momentum Strategy', fontsize=16, y=1.02)
    plt.show()
