import numpy as np
import pandas as pd
from scipy.fft import ifft
from datetime import datetime, timedelta
import yaml
from mediamix.transforms import logistic_function, geometric_adstock_tt


def load_generator_config(config_path):
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
        config['channels'] = list(config['media'].keys())
        config['n'] = (config['end_date'] - config['start_date']).days
        return config


def rescale(x, a, b):
    """Rescale x to be between a and b."""
    u, v = x.min(), x.max()
    x = (x - u) / (v - u)
    return x * (b - a) + a


def generate_base_adspend_signal(n, nfreqs=5, min_freq=2, max_freq=40, min_amp=10, max_amp=200):
    """
    Generate a base time series for use as an adspend signal.
    
    Incorporates some distinct frequencies to (hopefully)

        1. Mimic periodicity in true adspend with low effort.
        2. Make it easier for the sampler to converge.

    """
    # initialize the frequency domain with some (positive) noise
    y = np.abs(np.random.normal(size=n))

    # generate random amplitudes and random frequencies
    for i in range(nfreqs):
        freq = np.random.randint(min_freq, max_freq)
        y[freq] += np.random.randint(min_amp, max_amp)

    # convert to time domain and drop imaginary component
    y = ifft(y).real

    # first element always too high - zero it out
    y[0] = 0

    # return the scaled time domain signal
    return rescale(y, 0, 1)


def get_end_of_prior_month(d: datetime) -> datetime:
    """Get the end of the month before the given date."""
    d = datetime(d.year, d.month, 1)
    return d - timedelta(days=1)


def apply_decay(x, alpha):
    # capture the index since the adstock function loses it
    idx = x.index

    # apply the adstock function from the model
    x = geometric_adstock_tt(x, alpha).eval()

    # restore the index
    x = pd.Series(x, index=idx)

    return x


def create_empty_dataframe(n, channels):
    n_channels = len(channels)
    cols = channels + ['sales']
    today = datetime.now().date()
    end_date = get_end_of_prior_month(today)
    start_date = end_date - timedelta(days=n - 1)
    idx = pd.date_range(start=start_date, end=end_date)
    df = pd.DataFrame(np.zeros((n, n_channels + 1)), columns=cols, index=idx)
    return df
    

def channel_impact(x, beta=None, alpha=None, mu=None, model=None, decay=False, saturation=False, **kwargs):
    """Compute the impact of adspend on a channel on the outcome using our basic model."""
    # copy the series so we aren't updating the one in the dataframe
    x = x.copy()

    # the model parameters are interpreted assuming a scaled input, so rescale
    x = rescale(x, 0, 1)

    # if it's the decay model, then apply the decate
    if decay:
        x = apply_decay(x, alpha)

    # if it includes a saturation component, apply it as logistic with mu
    if saturation: 
        x = logistic_function(x, mu)

    # apply beta
    return beta * x

    
def impulse_pattern(n, hits):
    x = np.zeros(n)
    for a, y in hits:
        x[a] = y
    return x