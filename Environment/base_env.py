# base_env.py
# ----------------------------------------------------------
# Multi-Asset Deep Hedging Environment
# ----------------------------------------------------------
# This file defines a discrete-time environment that can be
# used by a PyTorch-based RL agent. 
# 
# The environment logic is:
#    - We hold a "portfolio" z_t
#    - We see a "market state" m_t (e.g. spots, vol, jumps)
#    - We choose an action a_t in R^n (hedge trades)
#    - The environment transitions to (z_{t+1}, m_{t+1})
#      via a user-supplied BatesModel, and we compute
#      an immediate reward = daily PnL - transactionCost.
#
# ----------------------------------------------------------

import numpy as np
from config.black_scholes import black_scholes_call, black_scholes_put
from data.calibration_bates import MultiAssetBatesModelCrossCorr, MultiAssetBatesModel
from config.transaction_costs import spot_scale_cost_fn_factory, bid_ask_cost_fn_factory, approximate_option_spread
from data.historical import calibrate_bates_multi_asset

class MultiAssetHedgingEnv:
    """
    MultiAssetHedgingEnv implements a daily-step environment
    for an RL agent to learn hedging in a multi-asset context.

    State:
      (z_t, m_t), where
         z_t in R^d_portfolio   # portfolio holdings or risk measures
         m_t in R^d_market      # market data (spot, vol, jump intensity, etc.)

    Action:
      a_t in R^n  # each dimension = #units traded in a hedging instrument.

    Next State:
      z_{t+1} = z_t + DeltaPortfolio(a_t, m_t)
      m_{t+1} ~ BatesModel.step(m_t, rng)

    Reward:
      reward_t = immediate_pnl(z_t, z_{t+1}, m_t, m_{t+1}) - cost(a_t, m_t)

    This environment is not directly a "tf.keras.Model". 
    It is a standard Python class the RL code can query.
    We penalize trades that exceed a specified liquidity limit by adding a large cost to them.
    """

    def __init__(self,
                 d_portfolio,
                 d_market,
                 n_hedges,
                 max_steps,
                 transaction_cost_fn=None,
                 bates_model = None,
                 use_corr_model = True,
                 seed=42, 
                 liquidity_limit=100.0,
                 big_penalty=1e6,
                 synthetic = True,
                 ):
        """
        Constructor for MultiAssetHedgingEnv.

        Parameters
        ----------
        d_portfolio : int
            Dimension of the portfolio representation (z_t).
            E.g. number of underlying assets.
        d_market : int
            Dimension of the market data vector (m_t).
            E.g. spot prices, implied vol, jump parameters, etc.
        n_hedges : int
            Dimension of action space (R^n). Each coordinate is
            how many units of a particular hedge to buy/sell.
        max_steps : int
            Episode length in days (or time steps).
        bates_model : object
            Must have:
              - sample_initial_state(d_market, rng)
              - step(m_t, rng) -> returns next market state m_{t+1}.
        transaction_cost_fn : callable
            A function transaction_cost_fn(a_t, m_t) -> cost_value
            giving cost of trades "a_t" in market state "m_t".
        seed : int
            Random seed for environment.
        liquidity_limit : float
            If |action[i]| > this limit for any i, we add a big penalty to the cost.
        big_penalty : float
            The cost penalty we add per instrument that violates the liquidity limit.
        
        """
        self.d_portfolio = d_portfolio
        self.d_market = d_market
        self.n_hedges = n_hedges
        self.max_steps = max_steps
        self.bates_model = bates_model
        self.rng = np.random.RandomState(seed)

        if transaction_cost_fn is not None:
            self.transaction_cost_fn = transaction_cost_fn(self)
        else:
            # fallback or default
            self.transaction_cost_fn = lambda a, m: 0.001 * np.sum(np.abs(a))

        # reward tracking
        self.reward_count = 0
        self.reward_running_mean = 0.0
        self.reward_running_var = 0.0

        # internal environment state
        self.current_step = 0
        self.state = None  # will hold (z_t, m_t)
        self.done = False

        # Liquidity parameters
        self.liquidity_limit = liquidity_limit
        self.big_penalty = big_penalty

        # Decide if we're using synthetic or historical data
        self.synthetic = synthetic

        if use_corr_model:
            N, alpha, beta, volofvol, lam_j, mu_j, sigma_j, S0_list, V0_list, r_, drift_spot, corr_2N  = calibrate_bates_multi_asset()
            self.bates_model = MultiAssetBatesModelCrossCorr(
                N=N, # Number of underlying assets
                alpha=alpha,
                beta=beta,
                sigma=volofvol,
                mu=drift_spot,
                lambda_j=lam_j,
                mu_j=mu_j,
                sigma_j=sigma_j,
                # 2N=6 => correlation matrix shape (6,6). 
                corr_2N = corr_2N,
                S0=S0_list, # Spot prices!
                V0=V0_list,
                r=r_,
                tau=1.0,
                dt=1.0/252,
                seed=seed                
                )
        else:
            # If not using correlation, maybe single-asset or a simpler multi-asset model
            from data.calibration_bates import MultiAssetBatesModel
            self.bates_model = MultiAssetBatesModel()

        # Instruments to hedge
        self.instruments = [
            # Underlying #0 -> BTC
            {"option_type": "call", "asset_index": 0, "strike": 91585.0},
            {"option_type": "put",  "asset_index": 0, "strike": 91585.0},
            {"option_type": "call", "asset_index": 0, "strike": 101226.0},
            {"option_type": "put",  "asset_index": 0, "strike": 101226.0},
            # Underlying #1 -> ETH
            {"option_type": "call", "asset_index": 1, "strike": 2586.0},
            {"option_type": "put",  "asset_index": 1, "strike": 2586.0},
            {"option_type": "call", "asset_index": 1, "strike": 2859.0},
            {"option_type": "put",  "asset_index": 1, "strike": 2859.0},
            # Underlying #2 -> LTC
            {"option_type": "call", "asset_index": 2, "strike": 128.0},
            {"option_type": "put",  "asset_index": 2, "strike": 128.0},
            {"option_type": "call", "asset_index": 2, "strike": 141.5},
            {"option_type": "put",  "asset_index": 2, "strike": 141.5},
        ]

        # len(instruments) == d_portfolio == n_hedges for a direct 1:1 mapping
        assert len(self.instruments) == d_portfolio == n_hedges, (
            "Mismatch in dimension. Adjust d_portfolio, n_hedges, or instrument list."
        )


    def reset(self):
        """
        Resets the environment to start a new episode.

        Returns
        -------
        state : (z_0, m_0)
            z_0: np.array of shape (d_portfolio,)
            m_0: np.array of shape (d_market,)
        """
        self.current_step = 0
        self.done = False

        self.reward_count = 0
        self.reward_running_mean = 0.0
        self.reward_running_var = 0.0

        # market initial
        m_0 = self.bates_model.sample_initial_state(self.d_market, self.rng)

        # portfolio initial
        z_0 = np.zeros(self.d_portfolio, dtype=np.float32)

        self.state = (z_0, m_0)
        return self.state

    def step(self, action):
        """
        Executes one time step of the environment.

        Parameters
        ----------
        action : np.array of shape (n_hedges,)
            Hedge trades to apply at this step.

        Returns
        -------
        next_state : (z_{t+1}, m_{t+1})
        reward : float
        done : bool
        info : dict
        """
        if self.done:
            raise RuntimeError("Environment has already finished this episode. Call reset().")

        z_t, m_t = self.state

        # 1) Compute transaction cost
        cost_val = self.transaction_cost_fn(action, m_t)

        # 2) Update portfolio
        penalty = 0.0
        for i in range(len(action)):
            if abs(action[i]) > self.liquidity_limit:
                penalty += self.big_penalty
        
        total_cost = cost_val + penalty

        z_tp1 = self._update_portfolio(z_t, action, m_t)

        # 3) Sample next market
        m_tp1 = self.bates_model.step(m_t, self.rng)

        # 4) Compute immediate reward
        #    daily PnL minus cost. We can do e.g. payoff difference:
        reward_val = self._immediate_reward(z_t, z_tp1, m_t, m_tp1, action, total_cost)
        info = {
            "cost_val": total_cost,
            "daily_pnl": (self._mark_to_market(z_tp1, m_tp1)
                        - self._mark_to_market(z_t, m_t))
        }

        # 5) Move time forward
        self.current_step += 1
        self.done = (self.current_step >= self.max_steps)
        next_state = (z_tp1, m_tp1)
        self.state = next_state

        return next_state, reward_val, self.done, info

    def _update_portfolio(self, z_t, action, m_t):
        """
        Defines how trades action 'a_t' translate to portfolio changes.

        For multi-asset deep hedging, we can store each underlying or Greek
        dimension in z_t.
        This function *adds* the new exposures to z_t.

        Returns z_{t+1}.
        For a simple approach, assume each dimension of z_t corresponds 1-to-1
        with each dimension of action. So if action[i] = +5, then we add 5 units
        of instrument i to the portfolio.
        """
        z_tp1 = z_t.copy()
        for i in range(min(len(z_tp1), len(action))):
            z_tp1[i] += action[i]
        return z_tp1

    def _immediate_reward(self, z_t, z_tp1, m_t, m_tp1, action, cost_val):
        """
        The reward for one step: daily PnL from z_t plus the new trades,
        minus the transaction cost. Instead of using a fixed scaling factor, we
        update running statistics to normalize the reward.
        """
        old_val = self._mark_to_market(z_t, m_t)
        new_val = self._mark_to_market(z_tp1, m_tp1)
        daily_pnl = new_val - old_val
        raw_reward = daily_pnl - cost_val

        # Update running statistics using an exponential moving average:
        # Here we use a smoothing parameter beta_sm (e.g. 0.99) for the running average.
        beta_sm = 0.99
        self.reward_count += 1
        if self.reward_count == 1:
            self.reward_running_mean = raw_reward
            self.reward_running_var = 0.0
        else:
            # Update mean and variance (using a simple exponential moving average update)
            self.reward_running_mean = beta_sm * self.reward_running_mean + (1 - beta_sm) * raw_reward
            # Note: update variance using the squared deviation from the new mean.
            self.reward_running_var = beta_sm * self.reward_running_var + (1 - beta_sm) * ((raw_reward - self.reward_running_mean) ** 2)

        std = np.sqrt(self.reward_running_var) if self.reward_running_var > 1e-8 else 1.0
        normalized_reward = (raw_reward - self.reward_running_mean) / std

        # Clip the normalized reward to keep it in a reasonable range:
        clip_threshold = 1000.0  # adjust threshold as needed
        clipped_reward = np.clip(normalized_reward, -clip_threshold, clip_threshold)
        
        return clipped_reward
    
    def _mark_to_market(self, z, m):
        """
        Compute the total mark-to-market (MtM) value of the portfolio z.

        We assume market state m is structured as:
        [S1, vol1, S2, vol2, ..., r, tau]
        where for each asset j:
        S_j    = m[2*j]
        vol_j  = m[2*j + 1]
        and the last two entries are r (risk-free rate) and tau (time to maturity).

        Since your Bates model returns m with length 6 (for N=2 assets),
        we have:
            num_assets = (len(m) - 2) // 2 = (6 - 2)//2 = 2
            index_r = 2 * num_assets = 4
            index_tau = 2 * num_assets + 1 = 5
        """
        num_assets = (len(m) - 2) // 2
        index_r = 2 * num_assets
        index_tau = 2 * num_assets + 1

        r = m[index_r]
        tau = m[index_tau]

        # Use a default dividend (if not provided in m)
        div = 0.0

        total_value = 0.0

        # Loop over each instrument in the portfolio
        for i, quantity in enumerate(z):
            instr = self.instruments[i]
            opt_type    = instr["option_type"]  # "call" or "put"
            asset_index = instr["asset_index"]
            strike      = instr["strike"]

            # Extract underlying data from m: S and vol for asset_index
            S   = m[2 * asset_index + 0]
            vol = m[2 * asset_index + 1]

            # Price the instrument using Black-Scholes (functions assumed defined or imported)
            if opt_type == "call":
                px = black_scholes_call(S, strike, r, vol, tau, div)
            else:
                px = black_scholes_put(S, strike, r, vol, tau, div)

            total_value += quantity * px

        return total_value

    def current_state_info(self):
        """
        Returns the current environment state for debugging
        or logging. 
        """
        z_t, m_t = self.state
        return {
            'step': self.current_step,
            'portfolio': z_t.copy(),
            'market': m_t.copy(),
            'done': self.done
        }
