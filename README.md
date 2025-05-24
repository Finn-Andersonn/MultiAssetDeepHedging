# Deep Hedging for Deep OTM Options: A Multi-Asset Approach in Bates Jump-Diffusion Environments

## Abstract

Hedging remains an ever-complicating problem in the financial securities industry, especially for deep out-of-the money (OTM) options 
where traditional models fail. We present a framework for hedging a portfolio of deep OTM European Calls and Puts, considering transactions 
costs, liquidity constraints, cross-correlations, and jumps. Under a market generated from a Bates stochastic volatility model, we find that 
a deep Reinforcement Learning (RL) agent significantly outperforms the classic Black–Scholes delta-neutral hedging strategy by 56.89% in VaR, 
74.82% in CVaR, 76.79% in max drawdown, 89.41% in P&L, and 81.62% in utility.

## Findings
First, we answer our initial topic question by establishing that deep hedging can be effectively calibrated to cryptocurrency markets, which 
serve as excellent proxies for extreme market conditions. Second, by incorporating cross-correlations, in addition to transaction costs and 
liquidity constraints, our framework addresses real trading limitations that are often ignored in theoretical models. Third, our approach 
demonstrates that neural networks can learn optimal hedging strategies that adapt to cross-asset correlations and sudden market jumps without 
relying on distributional assumptions.
