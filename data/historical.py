import numpy as np
import pandas as pd
import math
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.fft import fft

#  Characteristic-Function-Based Single-Asset Bates MLE 

def compute_fft_grid(N=64, dt=1/252):
    """
    Compute FFT grid points for characteristic function evaluation.
    """
    dv = 2*np.pi/(N*dt)
    v = np.arange(N)*dv
    dk = 2*np.pi/(N*dv)
    k = -np.pi/dt + np.arange(N)*dk
    return v, k

def bates_characteristic_function(u, t, params, v0):
    """
    Returns the characteristic function phi(u) = E[exp(i*u*log(S_{t+T}/S_t))]
    for a single-asset Bates model over time t.

    The Bates model does not have a simple closed-form PDF for log-returns, but 
    its characteristic function can be derived in closed form.

    Bates model parameters (a typical set):
        kappa    = mean reversion speed (vol process)
        theta    = long-run variance
        sigma    = vol-of-vol
        rho      = correlation between W^S and W^V
        lambda_j = jump intensity
        mu_j     = lognormal jump mean
        sigma_j  = lognormal jump vol
        r        = risk-free rate or drift
        mu       = drift of spot under chosen measure (or 'excess drift')
    v0 = current variance (v_t).
    """
    (kappa, theta, sigma, rho, lambda_j, mu_j, sigma_j, r, mu) = params

    i = complex(0,1)
    a_ = kappa * theta
    b_ = kappa - i*rho*sigma*u

    # Heston part
    uu = u**2 + i*u
    part_rho = - (rho*sigma*i*u)*2.0
    d = np.sqrt((rho*sigma*i*u - kappa)**2 + sigma**2 * (u**2 + i*u*(1-rho**2)))
    g = (b_ - d)/(b_ + d)

    
    def heston_log_phi():
        # "log" of characteristic function from standard known closed-form:
        #   phi(u) = exp( i*u*(mu-r)*t ) * exp( (kappa*theta/sigma^2)*((b_-d)*t - 2*log((1-g*exp(-d*t))/(1-g))) )
        #   * exp( v0 * ( (b_-d)*(1-exp(-d*t))/(sigma^2*(1-g*exp(-d*t))) ) )

        # 1) drift part
        drift_part = i*u*( (mu-r)*t )  # risk-neutral => mu=r if purely risk-neutral

        # 2) main exponent for mean reversion
        h1 = (b_ - d)*t
        # 3) denom = 1 - g*exp(-d*t)
        denom = (1 - g*np.exp(-d*t))
        # 4) log denom ratio
        log_ratio = -2.0 * np.log( (1 - g*np.exp(-d*t))/(1 - g) )

        reversion_part = (a_/sigma**2) * ( (b_ - d)*t - log_ratio )

        # 5) v0 portion
        #   "coefficient" = (b_-d)*(1-exp(-d*t))/(sigma^2*(1-g*exp(-d*t)))
        tmp = ((b_ - d)*(1 - np.exp(-d*t))) / (sigma**2 * denom )
        v0_part = v0 * tmp

        return drift_part + reversion_part + v0_part

    heston_exp = heston_log_phi()
    phi_heston = np.exp(heston_exp)

    # 2) Jumps: For Bates, multiply by exp( lambda_j * t * (phi_jump(u) - 1) )
    #    If jump size = lognormal( mu_j, sigma_j^2 ), then E[e^{i*u ln(J)}] = exp(i*u*mu_j - 0.5*sigma_j^2 u^2)
    phi_jump = np.exp(i*u*mu_j - 0.5*sigma_j**2 * (u**2))  # standard MGF of lognormal
    jump_factor = np.exp(lambda_j * t * (phi_jump - 1.0))

    phi_bates = phi_heston * jump_factor

    return phi_bates

def bates_cf_with_fft(params, v0, t, N=64):
    """
    Compute characteristic function values using FFT grid
    """
    v, k = compute_fft_grid(N)
    cf_values = np.array([bates_characteristic_function(u, t, params, v0) for u in v])
    return cf_values, k

def bates_pdf_fft(x, t, params, v0):
    """
    Replace bates_pdf_log_price with FFT-based computation
    """
    cf_values, k = bates_cf_with_fft(params, v0, t, N=64)
    
    # Apply FFT
    pdf_values = np.real(fft(cf_values))
    
    # Interpolate to get PDF at desired point x
    idx = np.searchsorted(k, x)
    if idx == len(k):
        idx = len(k) - 1
    pdf = max(pdf_values[idx], 1e-14)
    
    return pdf

def ensure_pos_semidef(corr_2N):
    """
    Make sure corr_2N is symmetric and fix negative eigenvalues.
    Do NOT forcibly set diag=1.0 afterwards, 
    which can break positive-definiteness again.
    """
    # 1) symmetrize
    corr_2N = 0.5*(corr_2N + corr_2N.T)
    
    # 2) eigen-decompose
    eigvals, eigvecs = np.linalg.eigh(corr_2N)
    
    # 3) clamp negative eigenvalues
    min_eig = np.min(eigvals)
    if min_eig < 0:
        eigvals[eigvals < 0] = 0.0
    
    # 4) reconstruct
    corr_2N_fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    # 5) sym again for numerical reasons
    corr_2N_fixed = 0.5*(corr_2N_fixed + corr_2N_fixed.T)
    
    # No forced diag=1
    return corr_2N_fixed


#  BatesCalibrator class 

class BatesCalibrator:
    """
    A class that calibrates single-asset Bates (Heston + Lognormal Jumps)
    using a characteristic-function-based approach for MLE 
    (no random sampling inside the objective).
    
    Then it extends to multiple assets by:
      - Calibrating each asset separately
      - Estimating cross-correlation from the returns 
        to fill a correlation block in your final Multi-Asset Bates model.
    """
    def __init__(self, dt=1/252):
        self.dt = dt
        self.iter_count = 0

    def negative_log_likelihood_fft(self, params, returns, v0):
        # Updated likelihood computation using FFT
        nll = 0.0
        for r_t in returns:
            p_r = bates_pdf_fft(r_t, self.dt, params, v0)
            nll -= np.log(p_r)
        return nll
    
    def single_asset_calibrate(self, prices, 
                               initial_guess=None,
                               bounds=None):
        """
        1) Compute daily log-returns r_t = ln(S_{t+1}/S_t).
        2) MLE: param = (kappa, theta, sigma, rho, lambda_j, mu_j, sigma_j, r, mu)
        3) The negative log-likelihood uses the bates_characteristic_function.
        4) Return the final parameters + initial variance.
        """
        returns = np.diff(np.log(prices))
        v0 = np.var(returns[:30])  # initial variance guess

        # If no guess is provided:
        if initial_guess is None:
            # (kappa, theta, sigma, rho, lambda_j, mu_j, sigma_j, r, mu)
            kappa_0 = 2.0
            theta_0 = 0.04
            sigma_0 = 0.5
            rho_0   = -0.5
            lam_0   = 0.2
            muj_0   = -0.2
            sigj_0  = 0.3
            r_0     = 0.01
            mu_0    = np.mean(returns) 
            initial_guess = [kappa_0, theta_0, sigma_0, rho_0, lam_0,
                             muj_0, sigj_0, r_0, mu_0]

        if bounds is None:
            bounds = [
                (1e-3, 20),  # kappa
                (1e-3, 1.0), # theta
                (1e-3, 2.0), # sigma
                (-0.99, 0.0),# rho
                (0.0,  5.0), # lambda_j
                (-1.0, 0.0), # mu_j
                (1e-3, 2.0), # sigma_j
                (0.0,  0.2), # r
                (-0.1, 0.1)  # mu
            ]

        def objective(param_vec):
            self.iter_count += 1
            val = self.negative_log_likelihood_fft(param_vec, returns, v0)
            if self.iter_count % 5 == 0:
                print(f"[Calibration] Iter: {self.iter_count}: NLL={val}")
            return val
        
        self.iter_count = 0

        res = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B', options={
                   "maxiter": 50,
                   "maxfun": 50,
                   "maxls": 10,
                   "gtol": 1e-6,
                   "ftol": 1e-9
               })
        final_params = res.x
        return final_params, v0, res.fun

    def build_2N_correlation_matrix(self, prices_dict, window=30):
        """
        Comprehensive 2N correlation matrix construction
        """
        assets = list(prices_dict.keys())
        N = len(assets)
        corr_2N = np.zeros((2*N, 2*N))
        
        # 1. Compute returns and realized variances
        returns_dict = {}
        realized_var_dict = {}
        for asset in assets:
            prices = prices_dict[asset]
            returns = np.diff(np.log(prices))
            returns_dict[asset] = returns
            
            # Compute realized variances using rolling window
            realized_var = np.array([
                np.var(returns[max(0, i-window):i]) 
                for i in range(window, len(returns))
            ])
            realized_var_dict[asset] = realized_var
        
        # 2. Fill spot return correlations (W^S_i and W^S_j block)
        for i in range(N):
            for j in range(N):
                r_i = returns_dict[assets[i]]
                r_j = returns_dict[assets[j]]
                # Ensure equal lengths for correlation
                min_len = min(len(r_i), len(r_j))
                spot_corr = np.corrcoef(r_i[:min_len], r_j[:min_len])[0,1]
                corr_2N[2*i+1, 2*j+1] = spot_corr
        
                # 3. Fill variance process correlations (W^V_i and W^V_j block)
                v_i = realized_var_dict[assets[i]]
                v_j = realized_var_dict[assets[j]]
                # Ensure equal lengths
                min_len = min(len(v_i), len(v_j))
                var_corr = np.corrcoef(v_i[:min_len], v_j[:min_len])[0,1]
                corr_2N[2*i, 2*j] = var_corr
        
        # 4. Fill leverage effects (W^V_i and W^S_i, diagonal blocks)
        for i in range(N):
            returns = returns_dict[assets[i]]
            realized_var = realized_var_dict[assets[i]]
            min_len = min(len(returns)-1, len(realized_var))
            leverage_corr = np.corrcoef(returns[:min_len], realized_var[:min_len])[0,1]
            corr_2N[2*i, 2*i+1] = leverage_corr
            corr_2N[2*i+1, 2*i] = leverage_corr
        
        # 5. Ensure matrix is valid correlation matrix
        corr_2N = (corr_2N + corr_2N.T) / 2  # Ensure symmetry
        
        # Make positive semi-definite if needed
        eigvals, eigvecs = np.linalg.eigh(corr_2N)
        if np.min(eigvals) < 0:
            eigvals[eigvals < 0] = 0
            corr_2N = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
        corr_2N = ensure_pos_semidef(corr_2N)
            
        return corr_2N

    def calibrate_multi_asset(self, prices_dict):
        """
        Updated multi-asset calibration with improved correlation structure
        """
        # 1. Perform single-asset calibrations
        single_calib = {}
        for asset, prices in prices_dict.items():
            params, v0, nll = self.single_asset_calibrate(prices)
            single_calib[asset] = {
                "params": params,
                "v0": v0,
                "nll": nll
            }
        
        # 2. Build full 2N correlation matrix
        corr_2N = self.build_2N_correlation_matrix(prices_dict)
        
        return {
            "single_calibration": single_calib,
            "correlation_2N": corr_2N
        }

def resample_to_3days(df):
    """
    df has columns [timestamp, close], each row is 1 day data.
    We'll group by 3-day intervals. 
    """
    # Convert timestamp to a datetime if needed
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)

    # 15 minute resample: we'll do last known close
    df_3m = df["close"].resample("3D").last().dropna().reset_index()
    return df_3m


def calibrate_bates_multi_asset():
    # Example for N=3 assets, but you can do any number of assets
    calibrator = BatesCalibrator(dt=1/252)

    # load CSV data
    df1_full = pd.read_csv("/Users/finn/Desktop/UCL Masters/Advanced ML/Project/Code/data/token_data/btcusd.csv")
    df2_full = pd.read_csv("/Users/finn/Desktop/UCL Masters/Advanced ML/Project/Code/data/token_data/ethusd.csv")
    df3_full = pd.read_csv("/Users/finn/Desktop/UCL Masters/Advanced ML/Project/Code/data/token_data/ltcusd.csv")

    df1 = resample_to_3days(df1_full)
    df2 = resample_to_3days(df2_full)
    df3 = resample_to_3days(df3_full)

    prices_dict = {
        "asset0": df1["close"].values,
        "asset1": df2["close"].values,
        "asset2": df3["close"].values
    }

    # 1) Single-asset calibrations + correlation among returns
    all_results = calibrator.calibrate_multi_asset(prices_dict)
    single_cal = all_results["single_calibration"]
    corr_2N = all_results["correlation_2N"]
    
    # Print summary
    for asset, info in single_cal.items():
        p = info["params"]
        v0 = info["v0"]
        nll= info["nll"]
        print(f"\nAsset: {asset}")
        (kappa, theta, sigma, rho, lam_j, mu_j, sig_j, r_, mu_) = p
        print("Params:", p)
        print("Initial variance v0=", v0, " Negative log-likelihood=", nll)
    
    # If N=3, results in a 6x6 matrix, structured as [W^V_1, W^S_1, W^V_2, W^S_2, W^V_3, W^S_3].
    assets_list = sorted(list(single_cal.keys()))
    N = len(assets_list)

    # pass them into your MultiAssetBatesModelCrossCorr:

    alpha = []
    beta  = []
    volofvol = []
    drift_spot = []
    lam_j = []
    mu_j = []
    sigma_j = []
    S0_list = []
    V0_list = []
    r_ = 0.01 

    for asset in assets_list:
        p = single_cal[asset]["params"]
        v0= single_cal[asset]["v0"]

        alpha.append(p[0])       # kappa
        beta.append(p[1])        # theta
        volofvol.append(p[2])    # sigma
        lam_j.append(p[4])
        mu_j.append(p[5])
        sigma_j.append(p[6])
        drift_spot.append(p[8])  # mu
        S0_list.append(prices_dict[asset][0])  # e.g. initial price from CSV
        V0_list.append(v0)

    print("\nConstructed a MultiAssetBatesModelCrossCorr with the calibrated parameters.\n")
    print("Alpha (kappa):", alpha)
    print("Beta (theta):", beta) # ensure isnt negative
    print("Vol-of-vol:", volofvol) # ensure isnt too large
    print("Jumps:", lam_j, mu_j, sigma_j)
    print("Correlation matrix (2N x 2N):\n", corr_2N)

    return N, alpha, beta, volofvol, lam_j, mu_j, sigma_j, S0_list, V0_list, r_, drift_spot, corr_2N
