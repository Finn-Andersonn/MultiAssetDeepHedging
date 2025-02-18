import math
import numpy as np

def CDF(X):
    a1 = 0.319381530
    a2 = -0.356563782
    a3 = 1.781477937
    a4 = -1.821255978
    a5 = 1.330274429
    
    x = abs(X)
    k = 1 / (1 + 0.2316419 * x)
    n = (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x)
    N = 1 - n * (a1 * k + a2 * k**2 + a3 * k**3 + a4 * k**4 + a5 * k**5)
    
    return N if X >= 0 else 1 - N

def d1(S, E, r, vol, div, tau):
    # if error, check the formula
    if vol == 0:
        vol = 1e-6
    return (math.log(S / E) + (r - div + 0.5 * vol * vol) * tau) / (vol * math.sqrt(tau))

def d2(S, E, r, vol, div, tau):
    if vol == 0:
        vol = 1e-6
    return (math.log(S / E) + (r - div - 0.5 * vol * vol) * tau) / (vol * math.sqrt(tau))

def d2_2(d1, vol, tau):
    return d1 - vol * tau


def black_scholes_call(S = 100, E = 100, r = 0.05, vol = 0.2, tau = 1, div=0.0):
    tau = max(tau, 1e-6)
    prob1 = d1(S, E, r, vol, div, tau)
    prob2 = d2(S, E, r, vol, div, tau)

    # Used throughout
    discount_factor = math.exp(-r * tau)
    div_discount = math.exp(-div * tau)

    # Standard put & call
    call = (S * div_discount) * CDF(prob1) - E * discount_factor * CDF(prob2)
    return call

def black_scholes_put(S, E, r, vol, tau, div=0.0):
    tau = max(tau, 1e-6)
    c = black_scholes_call(S, E, r, vol, tau, div)
    # Put-Call Parity => put = call - S e^{-div tau} + K e^{-r tau}
    return c - S*math.exp(-div*tau) + E*math.exp(-r*tau)


def compute_baseline_pnl(env):
    """
    A more general baseline that, for each instrument,
    computes the Black-Scholes delta using the underlying spot and volatility.
    The risk-free rate and tau are taken from the last two elements of m_t.
    """
    state = env.reset()
    z_t, m_t = state
    baseline_pnl = 0.0
    steps = env.max_steps

    for _ in range(steps):
        baseline_action = np.zeros(env.n_hedges, dtype=np.float32)
        # Extract r and tau from the end of m_t:
        r = m_t[-2]
        tau = m_t[-1]
        tau = max(tau, 1e-6)
        # For each instrument:
        for i, instr in enumerate(env.instruments):
            opt_type = instr["option_type"]
            asset_index = instr["asset_index"]
            strike = instr["strike"]

            # For the asset, assume S and vol are at positions 2*asset_index and 2*asset_index+1
            S = m_t[2 * asset_index]
            vol = m_t[2 * asset_index + 1]

            # Compute d1 for the instrument
            d_1_val = d1(S, strike, r, vol, 0.0, tau)
            if opt_type == "call":
                delta_i = CDF(d_1_val)
            else:
                delta_i = CDF(d_1_val) - 1.0

            baseline_action[i] = delta_i

        next_state, reward, done, _ = env.step(baseline_action)
        baseline_pnl += reward
        if done:
            break
        _, m_t = next_state
    return baseline_pnl

