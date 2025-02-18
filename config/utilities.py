import torch

def oce_utility(x, lambd=1.0, utility_type="exp"):
    """
    A simple example of an OCE-based utility for risk aversion.
    x: shape [batch_size], the random outcome
    lambd: risk aversion
    utility_type: "exp" or "cvar" or "mean", etc.
    returns: shape [batch_size]
    """
    # simple exponential:
    #   U(x) = (1 - exp(-lambda x))/lambda
    # or a cvar approach, but let's pick exponential for demonstration.
    if utility_type == "mean":
        return x  # no risk aversion
    elif utility_type == "exp":
        # (1 - e^{-lambda x}) / lambda
        return (1.0 - torch.exp(-lambd * x)) / lambd
    else:
        raise NotImplementedError("Utility type not implemented: " + utility_type)


def inverse_oce_utility(u, lambd=1.0, utility_type="exp"):
    """
    Inverse of the OCE utility. For example, for exponential:
      u = (1 - exp(-lambda x))/lambda
      => x = -1/lambda log(1 - lambda*u)
    We'll do that for "exp". If "mean", it's just identity.
    This is used in a risk-averse Bellman update:
      V(s) = sup_a  OCE( gamma * V(s') + r )
      => OCE^-1( V(s) ) = gamma * V(s') + r ...
    """
    if utility_type == "mean":
        return u
    elif utility_type == "exp":
        # solve:
        # u = (1 - exp(-l x))/l
        # => 1 - l u = exp(-l x)
        # => x = -1/l log(1 - l u)
        return -1.0 / lambd * torch.log(1.0 - lambd * u)
    else:
        raise NotImplementedError("inverse OCE not implemented for: " + utility_type)
