"""
Integration tests for PyMC-Marketing MMM implementation.

These tests validate that the MMM model can successfully recover known
parameters from synthetic data and produce accurate predictions.
"""

import numpy as np
import pytest
import pymc as pm
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

from mediamix import generator as mmg


# Configuration
RANDOM_SEED = 8927
CONFIG_PATH = './config/generator/basic_config.yaml'

# Ground truth parameters from config/generator/basic_config.yaml
GROUND_TRUTH = {
    'saturation': {
        'adwords': 3.1,
        'facebook': 4.2,
        'linkedin': 2.1,
    },
    'adstock': {
        'linkedin': 0.6,  # Only LinkedIn has adstock
    }
}

# Thresholds for validation
THRESHOLDS = {
    'r_squared': 0.95,
    'mape': 2.0,  # 2%
    'rhat': 1.01,
    'ess': 400,
    'param_error': 0.05,  # 5%
}


@pytest.fixture(scope='module')
def synthetic_data():
    """Generate synthetic data for testing."""
    np.random.seed(RANDOM_SEED)
    generator = mmg.Generator.from_config_file(CONFIG_PATH)
    df = generator.sample()
    return df


@pytest.fixture(scope='module')
def fitted_mmm(synthetic_data):
    """Fit MMM model to synthetic data."""
    df = synthetic_data
    df_model = df.reset_index().rename(columns={'index': 'date'})

    channel_columns = ['adwords', 'facebook', 'linkedin']
    X = df_model[['date'] + channel_columns].copy()
    y = df_model['sales'].values

    mmm = MMM(
        date_column='date',
        channel_columns=channel_columns,
        adstock=GeometricAdstock(l_max=12),
        saturation=LogisticSaturation(),
    )

    # Use fewer samples for faster testing
    mmm.fit(X, y, draws=500, tune=500, chains=2, random_seed=RANDOM_SEED)

    return mmm, X, y


class TestMMMInstantiation:
    """Test MMM model instantiation and configuration."""

    def test_mmm_creation(self):
        """Test that MMM can be instantiated with correct configuration."""
        mmm = MMM(
            date_column='date',
            channel_columns=['adwords', 'facebook', 'linkedin'],
            adstock=GeometricAdstock(l_max=12),
            saturation=LogisticSaturation(),
        )

        assert mmm.date_column == 'date'
        assert mmm.channel_columns == ['adwords', 'facebook', 'linkedin']
        assert isinstance(mmm.adstock, GeometricAdstock)
        assert isinstance(mmm.saturation, LogisticSaturation)


class TestMMMFitting:
    """Test MMM model fitting and convergence."""

    def test_model_fits_successfully(self, fitted_mmm):
        """Test that model fitting completes without errors."""
        mmm, X, y = fitted_mmm
        assert hasattr(mmm, 'idata')
        assert mmm.idata is not None

    def test_mcmc_convergence(self, fitted_mmm):
        """Test that MCMC sampling converges properly (R-hat < 1.01)."""
        mmm, X, y = fitted_mmm

        import arviz as az
        # Check convergence for key parameters
        summary = az.summary(mmm.idata, var_names=['intercept', 'y_sigma'], round_to=4)

        max_rhat = summary['r_hat'].max()
        assert max_rhat < THRESHOLDS['rhat'], \
            f"R-hat {max_rhat:.4f} exceeds threshold {THRESHOLDS['rhat']}"

    def test_effective_sample_size(self, fitted_mmm):
        """Test that effective sample size is adequate (ESS > 400)."""
        mmm, X, y = fitted_mmm

        import arviz as az
        summary = az.summary(mmm.idata, var_names=['intercept', 'y_sigma'], round_to=4)

        min_ess = summary['ess_bulk'].min()
        assert min_ess > THRESHOLDS['ess'], \
            f"ESS {min_ess:.0f} below threshold {THRESHOLDS['ess']}"


class TestModelPerformance:
    """Test model fit quality and prediction accuracy."""

    def test_model_fit_rsquared(self, fitted_mmm):
        """Test that model achieves high R-squared (> 0.95)."""
        mmm, X, y = fitted_mmm

        # Sample posterior predictive
        mmm.sample_posterior_predictive(X, extend_idata=True)

        posterior_predictive = mmm.idata.posterior_predictive
        y_pred_mean = posterior_predictive['y'].mean(dim=['chain', 'draw']).values

        # Calculate R-squared
        ss_res = np.sum((y - y_pred_mean) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        assert r_squared > THRESHOLDS['r_squared'], \
            f"R² {r_squared:.4f} below threshold {THRESHOLDS['r_squared']}"

    def test_prediction_accuracy_mape(self, fitted_mmm):
        """Test that predictions have low MAPE (< 2%)."""
        mmm, X, y = fitted_mmm

        # Use existing posterior predictive from previous test
        if 'posterior_predictive' not in mmm.idata:
            mmm.sample_posterior_predictive(X, extend_idata=True)

        posterior_predictive = mmm.idata.posterior_predictive
        y_pred_mean = posterior_predictive['y'].mean(dim=['chain', 'draw']).values

        # Calculate MAPE
        mape = np.mean(np.abs((y - y_pred_mean) / y)) * 100

        assert mape < THRESHOLDS['mape'], \
            f"MAPE {mape:.2f}% exceeds threshold {THRESHOLDS['mape']}%"


class TestParameterRecovery:
    """Test that model recovers ground truth parameters."""

    def test_saturation_parameter_recovery(self, fitted_mmm):
        """Test that saturation parameters match ground truth within 5%."""
        mmm, X, y = fitted_mmm

        posterior = mmm.idata.posterior

        # Extract saturation_lam posterior means
        saturation_lam = posterior['saturation_lam'].mean(dim=['chain', 'draw']).values

        channels = ['adwords', 'facebook', 'linkedin']
        for i, channel in enumerate(channels):
            recovered = saturation_lam[i]
            ground_truth = GROUND_TRUTH['saturation'][channel]

            error = abs(recovered - ground_truth) / ground_truth
            assert error < THRESHOLDS['param_error'], \
                f"{channel} saturation error {error:.4f} exceeds threshold {THRESHOLDS['param_error']}"

    def test_adstock_parameter_recovery(self, fitted_mmm):
        """Test that adstock parameters match ground truth within 5%."""
        mmm, X, y = fitted_mmm

        posterior = mmm.idata.posterior

        # Extract adstock_alpha for linkedin (index 2)
        adstock_alpha = posterior['adstock_alpha'].mean(dim=['chain', 'draw']).values

        linkedin_alpha = adstock_alpha[2]  # linkedin is index 2
        ground_truth = GROUND_TRUTH['adstock']['linkedin']

        error = abs(linkedin_alpha - ground_truth) / ground_truth
        assert error < THRESHOLDS['param_error'], \
            f"linkedin adstock error {error:.4f} exceeds threshold {THRESHOLDS['param_error']}"


class TestModelSavingLoading:
    """Test model persistence."""

    def test_model_save(self, fitted_mmm, tmp_path):
        """Test that model can be saved."""
        mmm, X, y = fitted_mmm

        model_path = tmp_path / "test_mmm.nc"
        mmm.save(str(model_path))

        assert model_path.exists()
        assert model_path.stat().st_size > 0

    def test_model_load(self, fitted_mmm, tmp_path):
        """Test that saved model can be loaded."""
        mmm, X, y = fitted_mmm

        model_path = tmp_path / "test_mmm.nc"
        mmm.save(str(model_path))

        # Load the model
        mmm_loaded = MMM.load(str(model_path))

        assert mmm_loaded is not None
        assert hasattr(mmm_loaded, 'idata')


@pytest.mark.slow
class TestAdvancedFeatures:
    """Test advanced MMM features (marked as slow)."""

    def test_prior_predictive_sampling(self, synthetic_data):
        """Test that prior predictive sampling works."""
        df = synthetic_data
        df_model = df.reset_index().rename(columns={'index': 'date'})

        channel_columns = ['adwords', 'facebook', 'linkedin']
        X = df_model[['date'] + channel_columns].copy()

        mmm = MMM(
            date_column='date',
            channel_columns=channel_columns,
            adstock=GeometricAdstock(l_max=12),
            saturation=LogisticSaturation(),
        )

        # Sample from priors (doesn't require fitting)
        mmm.sample_prior_predictive(X, samples=100)

        assert hasattr(mmm, 'idata')
        assert 'prior' in mmm.idata.groups()


# Parametrized test for different configurations
@pytest.mark.parametrize("l_max", [8, 12, 16])
def test_different_lmax_values(synthetic_data, l_max):
    """Test that model works with different l_max values."""
    df = synthetic_data
    df_model = df.reset_index().rename(columns={'index': 'date'})

    channel_columns = ['adwords', 'facebook', 'linkedin']
    X = df_model[['date'] + channel_columns].copy()
    y = df_model['sales'].values

    mmm = MMM(
        date_column='date',
        channel_columns=channel_columns,
        adstock=GeometricAdstock(l_max=l_max),
        saturation=LogisticSaturation(),
    )

    # Quick fit with minimal samples
    mmm.fit(X, y, draws=100, tune=100, chains=1, random_seed=RANDOM_SEED)

    assert hasattr(mmm, 'idata')
    assert mmm.idata is not None


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])
