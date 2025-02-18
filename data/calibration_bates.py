# calibration_bates.py
# ----------------------------------------------------------------
# Multi-Asset Bates model with cross-asset correlation in the
# 2N Brownian increments:
#  W^V_1, W^S_1, ..., W^V_N, W^S_N
# using a (2N x 2N) correlation matrix. Then each asset i uses
# (W^V_i, W^S_i) for variance & spot updates, with jumps.
# ----------------------------------------------------------------

import math
import numpy as np

class MultiAssetBatesModelCrossCorr:
    """
    Multi-Asset Bates model with cross-asset correlation among
    the 2N Brownian increments (variance & spot for each asset).
    For each asset i:

       dV_i(t) = alpha_i*(beta_i - V_i(t)) dt + sigma_i * sqrt(V_i(t)) dW^V_i
       dS_i(t) = S_i(t)* [ (mu_i dt) + sqrt(V_i(t)) dW^S_i + jump_i ]

    Adding correlation across different i in the W^S_j and W^V_j terms.
    We store a (2N x 2N) correlation matrix "corr_2N". We do a
    Cholesky factorization => we can generate correlated normals.

    Jumps: each asset i has intensity lambda_j_i, and jump sizes are
    lognormal( mu_j_i, sigma_j_i ).
    If one or more jumps occur in dt, we multiply them all to get "jump_factor".
    For more advanced modeling you may want a single jump draw if "poisson>0".

    We'll produce a market state m_t of shape 2N+2:
       [ S1, sqrt(V1),  S2, sqrt(V2),  ..., SN, sqrt(VN),  r, tau ]
    or adapt if you want more fields. We'll do no dividend or separate "div" field
    to keep it straightforward. If you want to expand, add them.

    Implementation is daily: dt = 1/252 or something. We call step() once per day.
    """

    def __init__(self,
                 N,
                 alpha,    # array length N
                 beta,
                 sigma,
                 mu,
                 lambda_j,
                 mu_j,
                 sigma_j,
                 # cross-corr matrix => shape (2N, 2N)
                 corr_2N,
                 S0,   # array length N
                 V0,   # array length N
                 r=0.01,
                 tau=1.0,
                 dt=1.0/252,
                 seed=1234):
        """
        N: number of assets

        alpha, beta, sigma, mu, lambda_j, mu_j, sigma_j => each is array length N,
          describing the Bates parameters for each asset i.
        corr_2N: 2N x 2N correlation matrix for Brownian increments
                 We do the CHOL => L, then L @ normal(0,I) to produce correlated draws.
                 The ordering is: [W^V_1, W^S_1, ..., W^V_N, W^S_N].
                 Must be positive semidefinite, etc. No partial disclaimers.

        S0[i], V0[i] => initial spot & variance for asset i.
        r, tau => global interest rate, time to maturity
        dt => step size, e.g. 1/252
        """
        self.N = N
        # store each param as float arrays
        self.alpha    = np.array(alpha,    dtype=float)
        self.beta     = np.array(beta,     dtype=float)
        self.sigma    = np.array(sigma,    dtype=float)
        self.mu       = np.array(mu,       dtype=float)
        self.lambda_j = np.array(lambda_j, dtype=float)
        self.mu_j     = np.array(mu_j,     dtype=float)
        self.sigma_j  = np.array(sigma_j,  dtype=float)

        self.S0   = np.array(S0, dtype=float)
        self.V0   = np.array(V0, dtype=float)
        self.r    = float(r)
        self.tau  = float(tau)
        self.dt   = float(dt)

        # check shape of corr_2N => must be (2N, 2N)
        corr_2N = np.array(corr_2N, dtype=float)
        if corr_2N.shape != (2*N, 2*N):
            raise ValueError(f"corr_2N must be shape (2N,2N). Found {corr_2N.shape}")
        # do a cholesky
        self.L = np.linalg.cholesky(corr_2N)

        # random generator
        self.rng = np.random.RandomState(seed)

    def sample_initial_state(self, d_market, rng):
        """
        Produce a vector of shape d_market = 2N+2:
         [ S1, sqrt(V1), S2, sqrt(V2), ..., SN, sqrt(VN), r, tau ]
        If you want more fields, adapt accordingly.
        """
        required_dim = 2*self.N + 2
        if d_market != required_dim:
            raise ValueError(f"For multi-asset with cross-corr, we define m_t with shape = {required_dim}. Found {d_market}")
        out = np.zeros(d_market, dtype=np.float32)
        for i in range(self.N):
            out[2*i + 0] = self.S0[i]
            out[2*i + 1] = math.sqrt(self.V0[i])
        out[-2] = self.r
        out[-1] = self.tau
        return out

    def step(self, m_t, rng):
        """
        Evolve from m_t => m_{t+1}, same shape (2N+2).
         We parse:
          S_i = m_t[2i], vol_i = m_t[2i+1], 
          r=m_t[-2], tau=m_t[-1].
        Then do a single day Euler for each asset i, using correlated normal draws:
         z ~ N(0,I_{2N}), w = L@z => we separate w in pairs: w^V_i, w^S_i
        Then do the standard Bates update with jump for each asset i.
        Return out.
        """
        dt = self.dt
        out = m_t.copy()
        r   = float(m_t[-2])
        tau = float(m_t[-1])

        # produce 2N uncorrelated normals
        z = rng.normal(0.0,1.0,size=(2*self.N,))
        # produce correlated increments w => shape(2N,)
        w = self.L.dot(z)

        # We'll parse w^V_i = w[2i], w^S_i = w[2i+1] for i in [0..N-1]
        for i in range(self.N):
            S   = float(m_t[2*i + 0])
            vol = float(m_t[2*i + 1])
            V   = vol*vol

            wV_i = w[2*i + 0]
            wS_i = w[2*i + 1]

            alpha_i = self.alpha[i]
            beta_i  = self.beta[i]
            sigma_i = self.sigma[i]
            mu_i    = self.mu[i]
            lam_i   = self.lambda_j[i]
            mu_j_i  = self.mu_j[i]
            sig_j_i = self.sigma_j[i]

            # 1) update variance
            V_next = V + alpha_i*(beta_i - V)*dt + sigma_i*math.sqrt(max(V,0.0))*wV_i*math.sqrt(dt)
            if V_next<0.0:
                V_next=0.0

            # 2) jump part
            jump_count = rng.poisson(lam_i*dt)
            jump_factor=1.0
            if jump_count>0:
                # sample jump_count lognormal draws
                for _ in range(jump_count):
                    Y= rng.normal(mu_j_i, sig_j_i)
                    jump_factor*= math.exp(Y)

            # 3) spot
            # d(log S) = (mu_i - 0.5*V)*dt + sqrt(V)* wS_i sqrt(dt) + log(jump_factor)
            drift     = (mu_i -0.5*V)*dt
            diffusion = math.sqrt(max(V,0.0))*wS_i*math.sqrt(dt)
            logS_next = math.log(S) + drift + diffusion + math.log(jump_factor)
            S_next    = math.exp(logS_next)

            out[2*i + 0] = S_next
            out[2*i + 1] = math.sqrt(V_next)

        # decrement tau
        tau_next = max(tau - dt, 0.0)
        out[-2] = r      # keep interest rate
        out[-1] = tau_next
        return out


class MultiAssetBatesModel:
    """
    Multi-asset Bates model for N assets. Each asset i has:
      dV_i = alpha_i*(beta_i - V_i) dt + sigma_i sqrt(V_i) dW^V_i
      jumps for the spot with intensity lambda_j_i, etc.
    This is in the event we do no cross-correlations among assets.

    We'll store arrays of length N for each parameter. We'll do discrete-time
    daily Euler steps.
    """

    def __init__(self,
                 N,
                 alpha,    # array of length N
                 beta,
                 sigma,
                 rho,      # array of length N for correlation between W^V_i and W^S_i
                 mu,       # drift array length N
                 lambda_j, # jump intensity array length N
                 mu_j,
                 sigma_j,
                 S0,       # array length N
                 V0,
                 r=0.01,
                 tau=1.0,
                 dt=1.0/252,
                 seed=1234):
        """
        Each parameter is an array of length N. E.g. alpha[i] is alpha for asset i.
        S0[i], V0[i] is initial spot, variance for asset i.
        We do no cross-asset correlation. If you want it, define a bigger covariance.

        We'll produce a d_market = 3*N + 2 vector if the environment wants
        (S_i, vol_i, div_i?), i=1..N, plus r and tau. Adjust as needed.
        """
        self.N = N
        self.alpha   = np.array(alpha, dtype=float)
        self.beta    = np.array(beta,  dtype=float)
        self.sigma   = np.array(sigma, dtype=float)
        self.rho     = np.array(rho,   dtype=float)
        self.mu      = np.array(mu,    dtype=float)
        self.lambda_j= np.array(lambda_j,dtype=float)
        self.mu_j    = np.array(mu_j,  dtype=float)
        self.sigma_j = np.array(sigma_j,dtype=float)
        
        self.S0 = np.array(S0, dtype=float)
        self.V0 = np.array(V0, dtype=float)
        self.r  = float(r)
        self.tau= float(tau)
        self.dt = float(dt)

        self.rng = np.random.RandomState(seed)

    def sample_initial_state(self, d_market, rng):
        """
        Produce [S1, vol1, S2, vol2, ..., SN, volN, r, tau] shape d_market=2N+2 
        (or more if you want a div for each asset).
        We'll skip dividend for each asset, or we can store zeros. 
        We'll do exactly 2*N + 2 = d_market.
        """
        if d_market != 2*self.N + 2:
            raise ValueError(f"d_market must be 2*N+2={2*self.N+2}, found {d_market}")
        out = np.zeros(d_market, dtype=np.float32)
        for i in range(self.N):
            out[2*i + 0] = self.S0[i]
            out[2*i + 1] = math.sqrt(self.V0[i])
        out[-2] = self.r
        out[-1] = self.tau
        return out

    def step(self, m_t, rng):
        """
        Evolves each asset i using a daily Euler scheme for variance and jump,
        ignoring cross-asset correlation. 
        We parse m_t => produce m_{t+1} with shape same as m_t.

        m_t has shape [2N+2], i.e.
         [ S1, vol1,  S2, vol2,  ..., SN, volN,   r, tau ]
        """
        dt = self.dt
        out = m_t.copy()
        r   = float(m_t[-2])
        tau = float(m_t[-1])
        
        for i in range(self.N):
            S   = float(m_t[2*i + 0])
            vol = float(m_t[2*i + 1])
            V   = vol*vol

            # correlation within asset i between W^V_i and W^S_i => we do 2 correlated normals for that asset
            zV = rng.normal(0.0,1.0)
            zS = rng.normal(0.0,1.0)
            if abs(self.rho[i])>1e-12:
                zS = self.rho[i]*zV + math.sqrt(1.0 - self.rho[i]**2)*zS

            # update V
            # V_{t+1} = V + alpha_i*(beta_i -V)*dt + sigma_i sqrt(V) * zV * sqrt(dt)
            alpha_i = self.alpha[i]
            beta_i  = self.beta[i]
            sigma_i = self.sigma[i]
            mu_i    = self.mu[i]
            lam_i   = self.lambda_j[i]
            mu_j_i  = self.mu_j[i]
            sig_j_i = self.sigma_j[i]

            V_next = V + alpha_i*(beta_i - V)*dt + sigma_i*math.sqrt(max(V,0.0))*zV*math.sqrt(dt)
            if V_next<0: 
                V_next=0.0

            # Jumps
            jump_count = rng.poisson(lam_i*dt)
            jump_factor=1.0
            if jump_count>0:
                for _ in range(jump_count):
                    Y= rng.normal(mu_j_i, sig_j_i)
                    jump_factor*= math.exp(Y)

            # spot update
            # d(logS)= (mu_i -0.5 V) dt + sqrt(V)*zS sqrt(dt) + log(jump_factor)
            drift     = (mu_i-0.5*V)*dt
            diffusion = math.sqrt(max(V,0.0))*zS*math.sqrt(dt)
            logS_next = math.log(S) + drift + diffusion + math.log(jump_factor)
            S_next    = math.exp(logS_next)

            # store
            out[2*i + 0] = S_next
            out[2*i + 1] = math.sqrt(V_next)

        # update tau
        tau_next = max(tau - dt, 0.0)
        out[-2]  = r        # remain same
        out[-1]  = tau_next
        return out
