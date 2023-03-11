from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pymc3 as pm
import arviz as az

import yaml
import mlflow

from mediamix.transforms import saturation, geometric_adstock



class Channel:

    def __init__(self, 
        channel_name: str,
        beta_sd_prior: float = 1,
        has_saturation: bool = False,
        saturation_alpha_prior: float = None,
        saturation_beta_prior: float = None,
        has_geometric_adstock: bool = False,
        geometric_adstock_alpha_prior: float = None,
        geometric_adstock_beta_prior: float = None
    ):
        self.channel_name = channel_name
        self.beta_sd_prior = beta_sd_prior

        self.has_saturation = has_saturation
        self.saturation_alpha_prior = saturation_alpha_prior
        self.saturation_beta_prior = saturation_beta_prior

        self.has_geometric_adstock = has_geometric_adstock
        self.geometric_adstock_alpha_prior = geometric_adstock_alpha_prior
        self.geometric_adstock_beta_prior = geometric_adstock_beta_prior
    
    @classmethod
    def from_config_dict(cls, name: str, config: Dict):
        c = Channel(name, config['beta']['sd'])
        if 'geometric_adstock' in config:
            c.has_geometric_adstock = True
            c.geometric_adstock_alpha_prior = config['geometric_adstock']['alpha']
            c.geometric_adstock_beta_prior = config['geometric_adstock']['beta']
        if 'saturation' in config:
            c.has_saturation = True
            c.saturation_alpha_prior = config['saturation']['alpha']
            c.saturation_beta_prior = config['saturation']['beta']
        return c

    def get_contribution(self, obs: np.ndarray):
        out = obs
    
        print(f'Adding channel: {self.channel_name}')
    
        if self.has_geometric_adstock:
            alpha = pm.Beta(
                f'geometric_adstock_{self.channel_name}', 
                alpha=self.geometric_adstock_alpha_prior, 
                beta=self.geometric_adstock_beta_prior)
            out = geometric_adstock(out, alpha)
    
        if self.has_saturation:
            channel_mu = pm.Gamma(
                f'saturation_{self.channel_name}', 
                alpha=self.saturation_alpha_prior, 
                beta=self.saturation_beta_prior)
            out = saturation(out, channel_mu)
    
        channel_b = pm.HalfNormal(
            f'beta_{self.channel_name}', 
            sd=self.beta_sd_prior)
        
        return out * channel_b

    def to_config_dict(self):
        config = { 'beta': { 'sd': self.beta_sd_prior } }
        if self.has_geometric_adstock:
            config['geometric_adstock'] = {
                'alpha': self.geometric_adstock_alpha_prior,
                'beta': self.geometric_adstock_beta_prior }
        if self.has_saturation:
            config['saturation'] = {
                'alpha': self.saturation_alpha_prior,
                'beta': self.saturation_beta_prior }
        return config
        

class ModelConfig:

    def __init__(self,
        outcome_name: str,
        intercept_mu: float,
        intercept_sd: float,
        sigma_beta: float,
        outcome_scale: float,
        channels: List[Channel] = []
    ):
        self.outcome_name = outcome_name
        self.intercept_mu = intercept_mu
        self.intercept_sd = intercept_sd
        self.sigma_beta = sigma_beta
        self.outcome_scale = outcome_scale
        self.channels: List[Channel] = []
        for channel in channels:
            self.add_channel(channel)
        self.control_vars = []
        self.index_vars = []

    def add_channel(self, channel):
        self.channels.append(channel)

    @property
    def channel_names(self):
        return [c.channel_name for c in self.channels]

    def create_model(self, df):
        model = pm.Model()
    
        with model:
            response_mean = []
    
            intercept = pm.Normal(
                'intercept', 
                mu=self.intercept_mu, 
                sd=self.intercept_sd)
            response_mean.append(intercept)
    
            for c in self.channels:
                obs = df[c.channel_name].values
                response_mean.append(c.get_contribution(obs))
    
            # Continuous Control Variables
            for channel_name in self.control_vars:
                x = df[channel_name].values
                
                print(f'Adding Control: {channel_name}')
                
                control_beta = pm.Normal(f'beta_{channel_name}', sd=1)
                channel_contrib = control_beta * x
                response_mean.append(channel_contrib)
                
            # Categorical control variables
            for var_name in self.index_vars:
                shape = len(df[var_name].unique())
                x = df[var_name].values
                
                print(f'Adding Index Variable: {var_name}')
                
                ind_beta = pm.Normal(f'beta_{var_name}',sd=.5,shape=shape)
                channel_contrib = ind_beta[x]
                response_mean.append(channel_contrib)
    
            # Noise level
            sigma = pm.HalfCauchy(
                'sigma', 
                beta=self.sigma_beta)
    
            # Define likelihood
            likelihood = pm.Normal(
                self.outcome_name, 
                mu=sum(response_mean), 
                sd=sigma, 
                observed=df[self.outcome_name].values)
    
        return model

    @classmethod
    def from_config_file(cls, config_path):
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
        return cls.from_config_dict(config)

    @classmethod
    def from_config_dict(cls, config):
        mc = ModelConfig(
            outcome_name=config['outcome']['name'],
            intercept_mu=config['outcome']['intercept']['mu'],
            intercept_sd=config['outcome']['intercept']['sd'],
            sigma_beta=config['outcome']['sigma']['beta'],
            outcome_scale=config['outcome']['scale'])

        for name in config['media'].keys():
            mc.add_channel(Channel.from_config_dict(name, config['media'][name]))

        return mc

    def to_config_dict(self):
        config = {
            'outcome': {
                'name': self.outcome_name,
                'intercept': {
                    'mu': self.intercept_mu,
                    'sd': self.intercept_sd
                },
                'sigma': { 'beta': self.sigma_beta },
                'scale': self.outcome_scale
            },
            'media': { c.channel_name: c.to_config_dict() for c in self.channels },
            'control_vars': self.control_vars,
            'index_vars': self.index_vars
        }
        return config

    def run_inference(self, params, df):
        with mlflow.start_run():
            model = self.create_model(df)
            with model:
                outcome = self.outcome_name
                
                mlflow.log_dict(self.to_config_dict(), 'config.yaml')

                mlflow.log_params(params)

                idata = pm.sample(
                    draws=params['draws'],
                    tune=params['tune'],
                    init=params['init'],
                    return_inferencedata=True)

                self.log_results(df, idata)

                return model, idata


    def log_results(self, df, idata):
        az.plot_trace(idata)
        plt.savefig('trace.png')
        plt.close()
        mlflow.log_artifact('trace.png')

        waic = az.waic(idata)
        for m in ['waic', 'waic_se', 'p_waic']:
            mlflow.log_metric(m, waic[m])

        ppc = pm.sample_posterior_predictive(idata, var_names=[self.outcome_name])
        idata_ppc = az.from_pymc3(posterior_predictive=ppc)
        idata.extend(idata_ppc)

        r2_score = az.r2_score(df[self.outcome_name].values, ppc[self.outcome_name])
        for m in ['r2', 'r2_std']:
            mlflow.log_metric(m, r2_score[m])

        summary_df = az.summary(idata)
        summary_df.to_html('summary.html')
        mlflow.log_artifact('summary.html')

        az.plot_ppc(idata_ppc);
        plt.savefig('ppc.png')
        plt.close()
        mlflow.log_artifact('ppc.png')

        idata.to_netcdf('idata.nc')
        mlflow.log_artifact('idata.nc')

            