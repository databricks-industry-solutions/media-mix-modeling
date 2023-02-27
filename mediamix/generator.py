import numpy as np
import pandas as pd
from scipy.fft import ifft
from datetime import datetime, timedelta
import yaml
from mediamix import transforms as mmt
import pyspark.sql
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, DateType, DoubleType


class Channel:

    def __init__(self, name, **kwargs):
        self.name = name
        self.saturation = kwargs['saturation']

        if self.saturation:
            self.mu = kwargs['mu']

        self.decay = kwargs['decay']
        if self.decay:
            self.alpha = kwargs['alpha']
        
        self.beta = kwargs['beta']
        self.sigma = kwargs['signal']['sigma']
        self.min = kwargs['min']
        self.max = kwargs['max']
    
    def sample(self, n):
        """Generate a basic signal to represent spend on this channel."""
        x = np.abs(np.cumsum(np.random.normal(0, self.sigma, size=n)))
        return rescale(x, self.min, self.max)

    def impact(self, x):
        """Compute the impact of adspend on a channel on the outcome using our basic model."""
    
        # the model parameters are interpreted assuming a scaled input, so rescale
        x = rescale(x, 0, 1)
    
        # if it's the decay model, then apply the decate
        if self.decay:
            x = mmt.geometric_adstock(x, self.alpha).eval()
    
        # if it includes a saturation component, apply it as logistic with mu
        if self.saturation: 
            x = mmt.saturation(x, self.mu).eval()
    
        # apply beta
        return self.beta * x
        

class Generator:

    def __init__(self, 
        start_date: datetime, 
        end_date: datetime,
        outcome_name: str,
        intercept: float,
        sigma: float,
        scale: float
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.n = (self.end_date - self.start_date).days + 1
        self.outcome_name = outcome_name
        self.intercept = intercept
        self.sigma = sigma
        self.scale = scale
        self.channels = {}
    
    def sample(self) -> pd.DataFrame:
        pass

    @classmethod
    def from_config_file(cls, filename: str) -> 'Generator':
        # load the config file
        with open(filename, "r") as config_file:
            config = yaml.safe_load(config_file)

        # create the base generator
        generator = Generator(
            config['start_date'],
            config['end_date'],
            config['outcome']['name'],
            config['outcome']['intercept'],
            config['outcome']['sigma'],
            config['outcome']['scale'])

        # add each of the media channels
        for name in config['media'].keys():
            channel = Channel(name, **config['media'][name])
            generator.add_channel(channel)
        
        return generator
    
    def add_channel(self, channel: Channel):
        self.channels[channel.name] = channel

    def _create_empty_dataframe(self):
        n_channels = len(self.channels)
        cols = list(self.channels.keys()) + [self.outcome_name]
        idx = pd.date_range(start=self.start_date, end=self.end_date)
        df = pd.DataFrame(np.zeros((self.n, n_channels + 1)), columns=cols, index=idx)
        return df

    def sample(self):
        """Generate an adspend dataset based on the given config.
        """
        df = self._create_empty_dataframe()

        # generate the baseline signal at the intercept with some white noise
        outcome = np.random.normal(self.intercept, self.sigma, self.n)
        
        # add the impact for each channel
        for channel in self.channels.values():
            adspend = channel.sample(self.n)
            df[channel.name] = np.round(adspend, 2)
            outcome += channel.impact(adspend)

        # scale the final outcome outcome by scale factor
        df[self.outcome_name] = np.round(outcome * self.scale, 2)

        return df
        

def rescale(x, a, b):
    """Rescale x to be between a and b."""
    u, v = x.min(), x.max()
    x = (x - u) / (v - u)
    return x * (b - a) + a


def impulse_pattern(n, hits):
    """Generate a simple impulse pattern."""
    x = np.zeros(n)
    for a, y in hits:
        x[a] = y
    return x


def convert_to_spark_dataframe(df: pd.DataFrame, schema: StructType) -> pyspark.sql.DataFrame:
    """Convert the generator pandas DataFrame to the expected Spark DataFrame.
    """
    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    return (
        spark.createDataFrame(
            df.reset_index(drop=False)
            .rename({'index': 'date'}), 
            schema=schema))
    