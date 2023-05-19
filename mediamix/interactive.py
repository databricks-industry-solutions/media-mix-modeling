import ipywidgets as widgets
from matplotlib import pyplot as plt
import numpy as np
import mediamix.transforms as mmt
import mediamix.generator as mmg

def _plot_geometric_adstock_curve(alpha):
    hits = [(5, 1)]
    x = np.arange(25)
    y_baseline = mmg.impulse_pattern(25, hits)
    y_delayed = mmt.geometric_adstock(y_baseline, alpha).eval()
    plt.plot(x, y_baseline, label='baseline', color='blue')
    plt.plot(x, y_delayed, label='delayed', linestyle='--', marker='x', color='orange')
    plt.xlim(0, 25)
    plt.ylim(0, 1)
    plt.title('geometric adstock distribution example')
    plt.xlabel('time')
    plt.ylabel('impact')
    plt.legend()

def _plot_saturation_curve(mu):
    x = np.linspace(0, 5, 200)
    y = mmt.saturation(x, mu).eval()
    plt.plot(x, y)
    plt.xlim(0, 5)
    plt.ylim(0, 1)
    plt.title('saturation example')

def _update_geometric_adstock_and_delay_experiment(
    saturation_mu, 
    adstock_alpha
) -> None:
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    _plot_geometric_adstock_curve(adstock_alpha)
    plt.subplot(122)
    _plot_saturation_curve(saturation_mu)

def display_geometric_adstock_and_delay_interactive():
    style = {'description_width': 'initial'}
    widgets.interact(
        _update_geometric_adstock_and_delay_experiment,
        saturation_mu=widgets.FloatSlider(min=0.1, max=2.0, step=0.1, value=1.0, 
            style=style, layout=widgets.Layout(width='50%')),
        adstock_alpha=widgets.FloatSlider(min=0, max=1, step=0.1, value=0.5,
        style=style, layout=widgets.Layout(width='50%')))