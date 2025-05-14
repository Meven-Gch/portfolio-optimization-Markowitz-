import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------------
# Return Analysis and general statistics
# ---------------------------------------------------------------------------------


def get_stk_returns():
    
    """
    Load and format the returns_fr_stocks_portfolio.csv
    """
    stk = pd.read_csv("returns_fr_stocks_portfolio.csv",
                      header=0, index_col=0, parse_dates=True)
    stk.index = stk.index.to_period('M')
    return stk
    
def annualize_rets(r, periods_per_year):
    
    """
    Annualizes a set of returns
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1

def annualize_vol(r, periods_per_year):
    
    """
    Annualizes the vol of a set of returns
    """
    return r.std()*(periods_per_year**0.5)

def sharpe_ratio(r, riskfree_rate, periods_per_year):
    
    """
    Computes the annualized sharpe ratio of a set of returns
    """
    # convert the annual riskfree rate to  a per period rate
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol

# ---------------------------------------------------------------------------------
# Modern Portfolio Theory 
# ---------------------------------------------------------------------------------

def portfolio_return(weights, returns):
    
    """
    Computes the return on a portfolio from constituent returns and weights
    weights are a numpy array or Nx1 matrix and returns are a numpy array or Nx1 matrix
    """
    return weights.T @ returns


def portfolio_vol(weights, covmat):
    
    """
    Computes the vol of a portfolio from a covariance matrix and constituent weights
    weights are a numpy array or N x 1 maxtrix and covmat is an N x N matrix
    """
    return (weights.T @ covmat @ weights)**0.5
    
def ew(er):
    
    """
    Retourne les poids égaux pour chaque actif.
    """
    n = len(er)
    return np.repeat(1/n, n)


from scipy.optimize import minimize

def minimize_vol(target_return, er, cov):
    
    """
    Returns the optimal weights that achieve the target return
    given a set of expected returns and a covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    return_is_target = {'type': 'eq',
                        'args': (er,),
                        'fun': lambda weights, er: target_return - portfolio_return(weights,er)
    }
    weights = minimize(portfolio_vol, init_guess,
                       args=(cov,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,return_is_target),
                       bounds=bounds)
    return weights.x


def msr(riskfree_rate, er, cov):
    
    """
    Returns the weights of the portfolio that gives you the maximum sharpe ratio
    given the riskfree rate and expected returns and a covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n 
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    def neg_sharpe(weights, riskfree_rate, er, cov):
        
        """
        Returns the negative of the sharpe ratio
        of the given portfolio
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate)/vol
    
    result = minimize(neg_sharpe, init_guess,
                       args=(riskfree_rate, er, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    
    
    return result.x  
   

def gmv(cov):
    
    """
    Returns the weights of the Global Minimum Volatility portfolio
    given a covariance matrix
    """
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)


def optimal_weights(n_points, er, cov):
    
    """
    Returns a list of weights that represent a grid of n_points on the efficient frontier
    """
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights


def plot_ef(n_points, er, cov, style='--', legend=True, show_cml=False, riskfree_rate=0, show_ew=False, show_gmv=False, ax=None):
    
    """
    Plots the multi-asset efficient frontier
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets,
        "Volatility": vols
    })

    ax.plot(ef["Volatility"], ef["Returns"], style, color='cadetblue', label="Efficient Frontier")


    if show_cml:
        w_msr = msr(riskfree_rate, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x, cml_y, color='tomato', linestyle='--', linewidth=2, label="CML (Capital Market Line)")
        ax.plot(vol_msr, r_msr, 'x', color='firebrick', markeredgewidth=16, label="MSR Portfolio")

    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_return(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        ax.plot([vol_ew], [r_ew], 'x', color='olive', markeredgewidth=16, label="EW Portfolio")

    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        ax.plot([vol_gmv], [r_gmv], 'x', color='teal', markeredgewidth=16, label="GMV Portfolio") 

    ax.set_xlabel("Volatility")
    ax.set_ylabel("Returns (%)")

    if legend:
        ax.legend(handletextpad=2, labelspacing=1.1)

    ax.grid(True)

    return ax


def portfolio_stat(name, weights, er, cov, riskfree_rate=0.1):

    r = portfolio_return(weights, er)
    vol = portfolio_vol(weights, cov)
    sharpe = (r - riskfree_rate) / vol
    print(f"{name} Portfolio :")
    print(f"  Return:        {r:.2%}")
    print(f"  Volatility:    {vol:.2%}")
    print(f"  Sharpe Ratio:  {sharpe:.2f}")

