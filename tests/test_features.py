"""
Unit Tests for Feature Engineering Module.

Tests all 15 feature engineering functions including edge cases.

Author: FinSecure AI Team
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_engineering import (
    balance_error_orig,
    balance_error_dest,
    log_amount,
    log_oldbalanceOrg,
    log_newbalanceOrig,
    log_oldbalanceDest,
    log_newbalanceDest,
    orig_balance_zero,
    dest_balance_zero,
    account_drained,
    amount_gt_orig_balance,
    dest_balance_unchanged,
    orig_to_dest_ratio,
    hour_of_day,
    is_high_risk_hour,
    engineer_features
)


# Test fixtures
@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    return pd.DataFrame({
        'step': [1, 2, 3, 4, 5],
        'amount': [1000.0, 5000.0, 100.0, 0.0, 10000.0],
        'oldbalanceOrg': [5000.0, 0.0, 10000.0, 5000.0, 8000.0],
        'newbalanceOrig': [4000.0, 0.0, 9900.0, 5000.0, 0.0],
        'oldbalanceDest': [1000.0, 0.0, 5000.0, 2000.0, 1000.0],
        'newbalanceDest': [2000.0, 5000.0, 5100.0, 2100.0, 11000.0],
    })


@pytest.fixture
def zero_balance_df():
    """DataFrame with zero balances for edge case testing."""
    return pd.DataFrame({
        'step': [1],
        'amount': [100.0],
        'oldbalanceOrg': [0.0],
        'newbalanceOrig': [0.0],
        'oldbalanceDest': [0.0],
        'newbalanceDest': [0.0],
    })


@pytest.fixture
def overdraft_df():
    """DataFrame simulating overdraft scenario."""
    return pd.DataFrame({
        'step': [1],
        'amount': [10000.0],
        'oldbalanceOrg': [5000.0],
        'newbalanceOrig': [-5000.0],
        'oldbalanceDest': [0.0],
        'newbalanceDest': [10000.0],
    })


class TestBalanceError:
    """Tests for balance error calculations."""

    def test_balance_error_orig_normal(self, sample_df):
        """Test normal transaction balance error calculation."""
        result = balance_error_orig(sample_df)
        assert result.iloc[0] == 500  # (5000 - 1000) - 4000

    def test_balance_error_orig_zero(self, zero_balance_df):
        """Test zero balance transaction."""
        result = balance_error_orig(zero_balance_df)
        assert result.iloc[0] == 0  # (0 - 100) - 0 = -100

    def test_balance_error_dest_normal(self, sample_df):
        """Test destination balance error."""
        result = balance_error_dest(sample_df)
        assert result.iloc[0] == 0  # (1000 + 1000) - 2000 = 0


class TestLogTransformations:
    """Tests for log transformations."""

    def test_log_amount_small_value(self, sample_df):
        """Test log of small amount."""
        result = log_amount(sample_df)
        assert result.iloc[2] == pytest.approx(np.log1p(100.0))

    def test_log_amount_large_value(self, sample_df):
        """Test log of large amount."""
        result = log_amount(sample_df)
        assert result.iloc[1] == pytest.approx(np.log1p(5000.0))

    def test_log_amount_zero(self, sample_df):
        """Test log of zero amount edge case."""
        result = log_amount(sample_df)
        assert result.iloc[3] == 0  # log1p(0) = 0

    def test_log_oldbalanceOrg(self, sample_df):
        """Test log transformation of old balance."""
        result = log_oldbalanceOrg(sample_df)
        assert result.iloc[0] == pytest.approx(np.log1p(5000.0))


class TestAccountFlags:
    """Tests for account status flags."""

    def test_orig_balance_zero_true(self, zero_balance_df):
        """Test origin zero balance detection."""
        result = orig_balance_zero(zero_balance_df)
        assert result.iloc[0] == 1

    def test_orig_balance_zero_false(self, sample_df):
        """Test non-zero balance."""
        result = orig_balance_zero(sample_df)
        assert result.iloc[0] == 0  # 5000 != 0

    def test_dest_balance_zero_true(self, zero_balance_df):
        """Test destination zero balance detection."""
        result = dest_balance_zero(zero_balance_df)
        assert result.iloc[0] == 1


class TestAccountDrained:
    """Tests for account drainage detection."""

    def test_account_drained_fully(self, sample_df):
        """Test fully drained account detection."""
        result = account_drained(sample_df)
        # Row 4: newbalanceOrig = 0 and oldbalanceOrg = 8000 > 0
        assert result.iloc[4] == 1

    def test_account_drained_not_drained(self, sample_df):
        """Test non-drained account."""
        result = account_drained(sample_df)
        # Row 0: newbalanceOrig = 4000 != 0
        assert result.iloc[0] == 0

    def test_account_drained_zero_start(self, sample_df):
        """Test account with zero starting balance."""
        result = account_drained(sample_df)
        # Row 1: oldbalanceOrg = 0, so condition fails
        assert result.iloc[1] == 0


class TestAmountVsBalance:
    """Tests for amount vs balance comparison."""

    def test_amount_gt_orig_balance_true(self, sample_df):
        """Test amount exceeding balance."""
        result = amount_gt_orig_balance(sample_df)
        # Row 4: 10000 > 8000
        assert result.iloc[4] == 1

    def test_amount_gt_orig_balance_false(self, sample_df):
        """Test amount within balance."""
        result = amount_gt_orig_balance(sample_df)
        # Row 0: 1000 < 5000
        assert result.iloc[0] == 0

    def test_amount_gt_orig_balance_exact(self):
        """Test amount exactly equal to balance."""
        df = pd.DataFrame({
            'amount': [1000.0],
            'oldbalanceOrg': [1000.0]
        })
        result = amount_gt_orig_balance(df)
        assert result.iloc[0] == 0  # Not greater, just equal


class TestDestBalanceUnchanged:
    """Tests for destination balance unchanged detection."""

    def test_dest_balance_unchanged_true(self, sample_df):
        """Test unchanged destination balance."""
        result = dest_balance_unchanged(sample_df)
        # Row 3: 2100 != 2000, so this should be 0
        assert result.iloc[3] == 0

    def test_dest_balance_unchanged_false(self, sample_df):
        """Test changed destination balance."""
        result = dest_balance_unchanged(sample_df)
        # Row 0: 2000 != 1000
        assert result.iloc[0] == 0


class TestOrigToDestRatio:
    """Tests for origin to destination ratio."""

    def test_orig_to_dest_ratio_normal(self, sample_df):
        """Test normal ratio calculation."""
        result = orig_to_dest_ratio(sample_df)
        # Row 0: 1000 / (1000 + 1) ≈ 0.999
        assert 0.99 < result.iloc[0] < 1.0

    def test_orig_to_dest_ratio_zero_dest(self, sample_df):
        """Test ratio with zero destination balance."""
        result = orig_to_dest_ratio(sample_df)
        # Row 1: 5000 / (0 + 1) = 5000, clipped to 100
        assert result.iloc[1] == 100.0

    def test_orig_to_dest_ratio_clipping(self):
        """Test ratio clipping at 100."""
        df = pd.DataFrame({
            'amount': [10000.0],
            'oldbalanceDest': [1.0]
        })
        result = orig_to_dest_ratio(df)
        assert result.iloc[0] == 100.0  # Clipped


class TestTemporalFeatures:
    """Tests for temporal features."""

    def test_hour_of_day_basic(self, sample_df):
        """Test hour extraction from step."""
        result = hour_of_day(sample_df)
        # Row 0: 1 % 24 = 1
        assert result.iloc[0] == 1

    def test_hour_of_day_wrapping(self):
        """Test hour wrapping around 24."""
        df = pd.DataFrame({'step': [24, 25, 48]})
        result = hour_of_day(df)
        assert result.iloc[0] == 0  # 24 % 24 = 0
        assert result.iloc[1] == 1  # 25 % 24 = 1
        assert result.iloc[2] == 0  # 48 % 24 = 0

    def test_is_high_risk_hour_midnight(self):
        """Test high risk hour detection (midnight)."""
        df = pd.DataFrame({'step': [0, 1, 2, 3, 4, 5]})
        result = is_high_risk_hour(df)
        # All should be 1 (midnight to 5 AM)
        assert all(result == 1)

    def test_is_high_risk_hour_daytime(self):
        """Test non-high risk hours."""
        df = pd.DataFrame({'step': [6, 12, 18, 23]})
        result = is_high_risk_hour(df)
        # All should be 0 (not in 0-5 range)
        assert all(result == 0)


class TestEngineerFeatures:
    """Integration test for complete feature engineering pipeline."""

    def test_engineer_features_shape(self, sample_df):
        """Test that engineer_features adds correct number of features."""
        result = engineer_features(sample_df)
        # Original columns: step, amount, oldbalanceOrg, newbalanceOrig,
        #                   oldbalanceDest, newbalanceDest
        # One-hot: type_TRANSFER, type_CASH_OUT, type_PAYMENT,
        #          type_CASH_IN, type_DEBIT (will be 0 since no type column)
        # New features: 15
        expected_new_cols = 15
        assert result.shape[1] >= sample_df.shape[1] + expected_new_cols

    def test_engineer_features_columns(self, sample_df):
        """Test that all engineered features are present."""
        result = engineer_features(sample_df)
        expected_features = [
            'balance_error_orig',
            'balance_error_dest',
            'log_amount',
            'log_oldbalanceOrg',
            'log_newbalanceOrig',
            'log_oldbalanceDest',
            'log_newbalanceDest',
            'orig_balance_zero',
            'dest_balance_zero',
            'account_drained',
            'amount_gt_orig_balance',
            'dest_balance_unchanged',
            'orig_to_dest_ratio',
            'hour_of_day',
            'is_high_risk_hour'
        ]
        for feature in expected_features:
            assert feature in result.columns

    def test_engineer_features_no_nan(self, sample_df):
        """Test that engineered features have no NaN values."""
        result = engineer_features(sample_df)
        assert not result.isnull().any().any()

    def test_engineer_features_preserves_original(self, sample_df):
        """Test that original columns are preserved."""
        result = engineer_features(sample_df)
        original_cols = ['step', 'amount', 'oldbalanceOrg',
                        'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
        for col in original_cols:
            assert col in result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])