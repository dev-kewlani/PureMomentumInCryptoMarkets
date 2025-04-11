# main.py
# This Lean algorithm implements a pure momentum strategy based on delayed Rate of Change Percent.
# It leverages QuantConnect's framework to provide:
#   - A custom alpha model (PureMomentumAlphaModel) that triggers long insights
#   - A portfolio construction model (NegativeReturnRankingPortfolioConstructionModel) that allocates cash based on 24-hour performance
#   - A risk management model (SimpleStopTakeRiskManagementModel) that manages positions using stop loss and take profit parameters.

from AlgorithmImports import *

class PureMomentumStrategy(QCAlgorithm):

    def Initialize(self):
        # Retrieve optimization parameters or use defaults:
        self.freePortfolioValuePercentage = float(self.GetParameter("FreePortfolioValuePercentage") or 0.1)
        self.portfolioAllocation = float(self.GetParameter("PortfolioAllocation") or 0.1)
        self.lagPeriod = int(self.GetParameter("LagPeriod") or 24)
        self.thresholdLong = float(self.GetParameter("thresholdLong") or 3.0)
        self.totalLongWeight = float(self.GetParameter("TotalLongWeight") or 0.1)
        self.stopLoss = float(self.GetParameter("stopLoss") or 2 * self.thresholdLong)
        self.takeProfit = float(self.GetParameter("takeProfit") or 2 * self.thresholdLong)

        # Backtest period and initial cash
        self.SetStartDate(2018, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)
        
        # Set Coinbase brokerage model for crypto cash account
        self.SetBrokerageModel(BrokerageName.Coinbase, AccountType.Cash)
        
        # Adjust settings for smaller trade sizes
        self.Settings.FreePortfolioValuePercentage = self.freePortfolioValuePercentage
        self.Settings.MinimumOrderMarginPortfolioPercentage = 0

        # Define the cryptocurrencies to trade
        symbols = ['BTCUSD', 'ETHUSD', 'USDTUSD', 'XRPUSD', 'ADAUSD', 'SOLUSD',
                   'DOGEUSD', 'AVAXUSD', 'DOTUSD', 'LTCUSD', 'DAIUSD', 'SHIBUSD', 'MATICUSD',
                   'CROUSD', 'ATOMUSD', 'ALGOUSD', 'APEUSD', 'BITUSD', 'MANAUSD', 'HBARUSD',
                   'FILUSD', 'GRTUSD', 'ICPUSD', 'EGLDUSD', 'VETUSD', 'AUDIOUSD', 'LDOUSD',
                   'SUSHIUSD', 'NEARUSD', 'LRCUSD', 'ZECUSD', 'BATUSD', 'KAVAUSD',
                   'ROSEUSD', 'ENJUSD', 'CHZUSD', 'SANDUSD', 'GALAUSD', 'MKRUSD', 'COMPUSD',
                   'CRVUSD', 'SNXUSD', 'EOSUSD', 'UNIUSD', 'YFIUSD', 'AAVEUSD', 'USTUSD', 'KSMUSD',
                   'STXUSD', 'BATUSD', 'MINAUSD', 'DASHUSD', 'ZENUSD',
                   'ZRXUSD', 'CVCUSD', 'PUNDIXUSD', 'BANDUSD', 'STORJUSD', 'RENUSD', 'COTIUSD',
                   'BALUSD', 'OCEANUSD', 'ANKRUSD', 'SKLUSD', 'GNOUSD', 'GLMUSD', 'AMPUSD',
                   'FETUSD', 'RLCUSD', 'INJUSD', '1INCHUSD', 'ENSUSD', 'MASKUSD',
                   'LPTUSD', 'API3USD', 'SNTUSD', 'HNTUSD', 'KNCUSD', 'MTLUSD', 'NMRUSD',
                   'PERPUSD', 'PLAUSD', 'POWRUSD', 'QNTUSD', 'REQUSD', 'VTHOUSD', 'ZRXUSD',
                   'ALICEUSD', 'ASTUSD', 'BNTUSD', 'CTSIUSD', 'FARMUSD', 'GTCUSD', 'IMXUSD',
                   'JASMYUSD', 'LOOMUSD', 'LRCUSD', 'NKNUSD', 'OXTUSD', 'POLSUSD', 'RADUSD']

        for symbol in symbols:
            self.AddCrypto(symbol, Resolution.Hour, Market.Coinbase)

        # Add the alpha model (pure momentum based on delayed ROC)
        self.AddAlpha(PureMomentumAlphaModel(
            lag_period=self.lagPeriod, 
            threshold_long=self.thresholdLong,
            total_long_weight=self.totalLongWeight))

        # Disable automatic rebalancing based on insight expiration and security changes.
        self.Settings.RebalancePortfolioOnInsightChanges = False
        self.Settings.RebalancePortfolioOnSecurityChanges = False

        # Set the custom portfolio construction model
        self.SetPortfolioConstruction(NegativeReturnRankingPortfolioConstructionModel(
            allocation=self.portfolioAllocation))
        
        # Set the risk management model to exit positions based on stop loss and take profit.
        self.SetRiskManagement(SimpleStopTakeRiskManagementModel(
            stopLoss=self.stopLoss, takeProfit=self.takeProfit))
        
        # Warm up indicators
        self.SetWarmup(25, Resolution.Hour)
        self.Debug("Initialized with cash: " + str(self.Portfolio.Cash))


#########################################
# Portfolio Construction Model
#########################################
class NegativeReturnRankingPortfolioConstructionModel(PortfolioConstructionModel):
    def __init__(self, allocation=0.1):
        self.allocation = allocation

    def CreateTargets(self, algorithm, insights):
        active_insights = [i for i in insights if i.Direction == InsightDirection.Up]
        if not active_insights:
            return []
            
        symbol_returns = {}
        for insight in active_insights:
            symbol = insight.Symbol
            history = algorithm.History([symbol], 24, Resolution.Hour)
            if history.empty:
                continue
            try:
                symbol_history = history.loc[symbol].sort_index()
            except Exception:
                continue
            first_price = symbol_history.iloc[0]['close']
            last_price  = symbol_history.iloc[-1]['close']
            ret = (last_price - first_price) / first_price
            if ret < 0:
                symbol_returns[symbol] = abs(ret)
            else:
                symbol_returns[symbol] = 0
                
        total_negative = sum(symbol_returns.values())
        if total_negative == 0:
            return []
        
        available_cash = algorithm.Portfolio.Cash
        total_allocation_cash = available_cash * self.allocation
        targets = []
        for insight in active_insights:
            symbol = insight.Symbol
            if not algorithm.Portfolio[symbol].Invested:
                weight_factor = symbol_returns.get(symbol, 0)
                allocation_cash = total_allocation_cash * (weight_factor / total_negative)
                price = algorithm.Securities[symbol].Price
                if price <= 0:
                    continue
                quantity = allocation_cash / price
                targets.append(PortfolioTarget(symbol, quantity))
                
        return targets


#########################################
# Risk Management Model
#########################################
class SimpleStopTakeRiskManagementModel(RiskManagementModel):
    def __init__(self, stopLoss, takeProfit):
        self.stopLoss = stopLoss
        self.takeProfit = takeProfit

    def ManageRisk(self, algorithm, targets):
        risk_adjusted_targets = []
        for kvp in algorithm.Portfolio:
            holding = kvp.Value
            if not holding.Invested:
                continue
            avg_price = holding.AveragePrice
            current_price = algorithm.Securities[holding.Symbol].Price
            if avg_price <= 0 or current_price <= 0:
                continue
            if current_price <= avg_price * (1 - self.stopLoss) or current_price >= avg_price * (1 + self.takeProfit):
                risk_adjusted_targets.append(PortfolioTarget(holding.Symbol, 0))
                algorithm.Debug(f"Risk triggered for {holding.Symbol}: Current Price = {current_price}, Average Price = {avg_price}. Liquidating position.")
        return risk_adjusted_targets


#########################################
# Alpha Model: Pure Momentum with Delayed ROC
#########################################
class PureMomentumAlphaModel(AlphaModel):
    def __init__(self, lag_period=24, threshold_long=2.0, total_long_weight=0.1):
        self.lag_period = lag_period
        self.threshold_long = threshold_long
        self.total_long_weight = total_long_weight
        self.symbol_data = {}

    def Update(self, algorithm, data):
        insights = []
        for symbol, symbol_data in self.symbol_data.items():
            if not symbol_data.IsReady:
                continue
            delayed_roc = symbol_data.DelayedROC
            if delayed_roc < -self.threshold_long:
                insights.append(
                    Insight.Price(
                        symbol,
                        timedelta(days=7),
                        InsightDirection.Up,
                        None, None, None,
                        self.total_long_weight
                    )
                )
        return insights

    def OnSecuritiesChanged(self, algorithm, changes):
        for security in changes.AddedSecurities:
            symbol = security.Symbol
            self.symbol_data[symbol] = SymbolData(algorithm, symbol, self.lag_period)
        for security in changes.RemovedSecurities:
            symbol = security.Symbol
            algorithm.Liquidate(symbol)
            if symbol in self.symbol_data:
                del self.symbol_data[symbol]


#########################################
# Helper Class: SymbolData
#########################################
class SymbolData:
    def __init__(self, algorithm, symbol, lag_period):
        self.algorithm = algorithm
        self.symbol = symbol
        self.consolidator = algorithm.ResolveConsolidator(symbol, Resolution.Hour)
        self.roc = RateOfChangePercent(1)
        delay = Delay(lag_period)
        self.delayed_roc = IndicatorExtensions.Of(delay, self.roc)
        algorithm.RegisterIndicator(symbol, self.roc, self.consolidator)

    @property
    def DelayedROC(self):
        return self.delayed_roc.Current.Value

    @property
    def IsReady(self):
        return self.delayed_roc.IsReady
