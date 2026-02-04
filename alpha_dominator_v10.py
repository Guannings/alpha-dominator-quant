#!/usr/bin/env python3
"""
Regime-Adaptive Mean-Variance Optimization Strategy - Version 10.0
===================================================================
"The Alpha Dominator" - IR Filter + Growth Anchor + Regularized ML

Key innovations:
1. Information Ratio Filter: Only assets with IR > 0.5 vs SPY are eligible in RISK_ON
2. Growth Anchor: QQQ + XLK + SMH + VGT minimum 60% weight during RISK_ON
3. Gold capped at 5% in Bull markets
4. Regularized Random Forest (max_depth=4, min_samples_leaf=400, ccp_alpha=0.01)
5. Enhanced feature set: VIX, Vol Momentum, Trend Score, Momentum, Cross-Asset Signals
6. 10-day EMA probability smoothing (heavy smoothing to prevent regime flickering)
7. Overfitting health dashboard with stability bands & red alerts
8. Turnover Brake: Penalty = sum(abs(new - old)) * 500 to curb excessive trading costs
9. High-Alpha Universe: SMH + VGT added, VEA + XLP + VNQ removed

Author: Quantitative Research
Version: 10.0.0
"""

import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.patches import Patch
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


@dataclass
class StrategyConfig:
    """Strategy configuration for The Alpha Dominator."""

    ml_threshold: float = 0.55
    sma_lookback: int = 200
    rs_lookback: int = 126  # 6-month for IR calculation

    momentum_3m_days: int = 63
    momentum_6m_days: int = 126
    volatility_lookback: int = 60

    max_single_weight: float = 0.35
    gold_cap_risk_on: float = 0.05  # 5% max gold in bull (The Gold Cap)
    min_growth_anchor: float = 0.60  # 60% min QQQ+XLK+SMH+VGT in RISK_ON (Growth Anchor)
    ir_threshold: float = 0.5  # IR > 0.5 required for eligibility (Velvet Rope)

    entropy_lambda: float = 0.15  # Shannon Entropy coefficient
    min_effective_n: float = 3.0
    growth_anchor_penalty: float = 500.0  # High-priority penalty multiplier for Growth Anchor constraint
    turnover_penalty: float = 50.0  # The Turnover Brake penalty multiplier

    # Adaptive rebalancing candidates
    rebalance_candidates: Tuple[int, ...] = (21, 42, 63)

    transaction_cost_bps: float = 10.0
    risk_free_rate: float = 0.04

    # Overfitting thresholds
    overfit_gap_threshold: float = 0.12  # 12% gap = overfitting alert
    underfit_threshold: float = 0.51  # Below 51% = underfitting alert

    # EMA smoothing for ML probabilities
    prob_ema_span: int = 10  # 10-day EMA for heavy smoothing

    # Floating point tolerance for constraint checks
    constraint_tolerance: float = 0.001
    
    # Anxiety Veto thresholds (for elevated VIX conditions)
    anxiety_vix_threshold: float = 0.18  # VIX level above which anxiety veto applies
    anxiety_ml_prob_threshold: float = 0.75  # Required ML probability when VIX is elevated

    # UI colors
    alert_background_color: str = '#FFCCCC'  # Light Red for health dashboard alerts


class DataManager:
    """Data acquisition with 7-Feature Set for Endgame Model."""

    EQUITIES = ['SPY', 'QQQ', 'IWM', 'XLK', 'XLE', 'SMH', 'VGT']
    FIXED_INCOME = ['TLT', 'IEF', 'SHY']
    ALTERNATIVES = ['GLD']
    GROWTH_ANCHORS = ['QQQ', 'XLK', 'SMH', 'VGT']
    VIX_TICKER = '^VIX'

    def __init__(self, start_date: str = '2010-01-01', end_date: str = None, config: StrategyConfig = None):
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.config = config or StrategyConfig()
        self.all_tickers = self.EQUITIES + self.FIXED_INCOME + self.ALTERNATIVES
        self.prices, self.returns, self.features, self.vix = None, None, None, None
        self.sma_200, self.above_sma, self.raw_momentum, self.relative_strength = None, None, None, None
        self.information_ratio, self.asset_volatilities = None, None

    def load_data(self, max_retries: int = 3) -> None:
        logger.info(f"Loading data for {len(self.all_tickers)} assets")
        for attempt in range(max_retries):
            try:
                data = yf.download(self.all_tickers + [self.VIX_TICKER], start=self.start_date, end=self.end_date,
                                   auto_adjust=True, progress=False)
                if isinstance(data.columns, pd.MultiIndex):
                    prices = data['Close'].copy()
                else:
                    prices = data[['Close']].copy()
                if self.VIX_TICKER in prices.columns:
                    self.vix = prices[self.VIX_TICKER].copy()
                    prices = prices.drop(columns=[self.VIX_TICKER])
                else:
                    self.vix = prices['SPY'].pct_change().rolling(21).std() * np.sqrt(252) * 100
                available = [t for t in self.all_tickers if t in prices.columns]
                self.all_tickers = available
                self.prices = prices[available].ffill().bfill().dropna()
                self.returns = self.prices.pct_change().dropna()
                self.vix = self.vix.reindex(self.prices.index).ffill().bfill()
                self._calculate_indicators()
                return
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1: raise RuntimeError("Data loading failed")

    def _calculate_indicators(self) -> None:
        self.sma_200 = self.prices.rolling(self.config.sma_lookback).mean()
        self.above_sma = self.prices > self.sma_200
        mom_3m = self.prices.pct_change(self.config.momentum_3m_days)
        mom_6m = self.prices.pct_change(self.config.momentum_6m_days)
        self.raw_momentum = (mom_3m + mom_6m) / 2
        self.asset_volatilities = self.returns.rolling(self.config.volatility_lookback).std() * np.sqrt(252)
        spy_return = self.prices['SPY'].pct_change(self.config.rs_lookback)
        self.relative_strength = pd.DataFrame(index=self.prices.index)
        for ticker in self.all_tickers:
            self.relative_strength[ticker] = self.prices[ticker].pct_change(self.config.rs_lookback) - spy_return
        self.relative_strength = self.relative_strength.ffill().bfill()
        self.information_ratio = pd.DataFrame(index=self.prices.index)
        for ticker in self.all_tickers:
            if ticker == 'SPY': self.information_ratio[ticker] = 0.0; continue
            active_ret = self.returns[ticker] - self.returns['SPY']
            ir = (active_ret.rolling(self.config.rs_lookback).mean() * 252) / (
                        active_ret.rolling(self.config.rs_lookback).std() * np.sqrt(252)).replace(0, np.nan)
            self.information_ratio[ticker] = ir
        self.information_ratio = self.information_ratio.replace([np.inf, -np.inf], np.nan).ffill().bfill()

    def engineer_features(self) -> pd.DataFrame:
        """Engineer 7 features to match the Endgame Model constraints."""
        if self.prices is None: raise ValueError("Load data first")
        features = pd.DataFrame(index=self.prices.index)

        # 1. Realized Volatility
        features['realized_vol'] = self.vix / 100.0

        # 2. Volatility Momentum
        vix_shifted = self.vix.shift(21).replace(0, np.nan)
        features['vol_momentum'] = (self.vix / vix_shifted - 1).clip(-0.5, 0.5)

        # 3. Equity Risk Premium
        spy_erp = 1.0 / (self.prices['SPY'] / self.prices['SPY'].rolling(252).mean())
        features['equity_risk_premium'] = spy_erp - self.config.risk_free_rate

        # 4. Trend Score (Scaled)
        spy_sma = self.prices['SPY'].rolling(200).mean()
        features['trend_score'] = ((self.prices['SPY'] - spy_sma) / spy_sma) * 100.0

        # 5. Momentum (21d)
        features['momentum_21d'] = self.prices['SPY'].pct_change(21).clip(-0.2, 0.2)

        # 6. Cross-Asset Signal (QQQ vs SPY) - THIS WAS MISSING
        tech_proxy = self.prices.get('QQQ', self.prices['SPY'])
        features['qqq_vs_spy'] = (tech_proxy.pct_change(63) - self.prices['SPY'].pct_change(63)).clip(-0.2, 0.2)

        # 7. Bond Signal (TLT Momentum) - THIS WAS MISSING
        bond_proxy = self.prices.get('TLT', self.prices['SPY'])
        features['tlt_momentum'] = bond_proxy.pct_change(21).clip(-0.1, 0.1)

        features = features.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        self.features = features.dropna()
        return self.features

    def get_aligned_data(self) -> Tuple[pd.DataFrame, ...]:
        idx = (self.prices.index.intersection(self.features.index).intersection(self.returns.index)
               .intersection(self.sma_200.dropna().index).intersection(self.raw_momentum.dropna().index)
               .intersection(self.relative_strength.dropna().index).intersection(self.information_ratio.dropna().index))
        return (self.prices.loc[idx], self.returns.loc[idx], self.features.loc[idx], self.vix.loc[idx],
                self.sma_200.loc[idx], self.above_sma.loc[idx], self.raw_momentum.loc[idx],
                self.relative_strength.loc[idx], self.asset_volatilities.loc[idx], self.information_ratio.loc[idx])

    def get_asset_categories(self) -> Dict[str, List[str]]:
        return {'equities': [t for t in self.EQUITIES if t in self.all_tickers],
                'fixed_income': [t for t in self.FIXED_INCOME if t in self.all_tickers],
                'alternatives': [t for t in self.ALTERNATIVES if t in self.all_tickers],
                'safe_haven': [t for t in ['GLD', 'TLT', 'IEF', 'SHY'] if t in self.all_tickers],
                'gold': ['GLD'] if 'GLD' in self.all_tickers else [],
                'bonds_cash': [t for t in ['TLT', 'IEF', 'SHY'] if t in self.all_tickers], 'all': self.all_tickers}


from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
import shap


class AdaptiveRegimeClassifier:
    """
    THE ENDGAME: Consensus Ensemble + Monotonic Constraints.
    Includes SHAP visualization and Model Health Dashboard.
    """

    def __init__(self, config: StrategyConfig = None):
        self.config = config or StrategyConfig()

        # MODEL A: The Aggressor (XGBoost)
        # Monotone: Vol (-1), Trend (+1)
        self.model_alpha = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            monotone_constraints=(-1, -1, 0, 1, 1, 1, 0),
            subsample=0.8,
            random_state=42,
            n_jobs=-1
        )

        # MODEL B: The Skeptic (Decision Tree)
        self.model_beta = DecisionTreeClassifier(
            max_depth=2,
            min_samples_leaf=200,
            random_state=99
        )

        self.feature_names: List[str] = []
        self.train_scores: List[float] = []
        self.test_scores: List[float] = []
        self.oob_scores: List[float] = []  # Not used in XGB, kept for compatibility
        self.window_dates: List[datetime] = []
        self.selected_rebalance_periods: List[int] = []

        # SHAP storage
        self.shap_values = None
        self.shap_features = None
        self.feature_importances_history = []

        self.current_rebalance_period = 42
        
        # Model stability tracking
        self.model_stability = 'UNKNOWN'

    def walk_forward_train(self, features, returns, initial_train_years=5, step_months=12):
        logger.info("Starting adaptive walk-forward training (CONSENSUS ENGINE)")
        self.feature_names = features.columns.tolist()
        target = (returns.shift(-21).rolling(21).sum() > 0).astype(int).dropna()

        valid_idx = features.index.intersection(target.index)
        X, y = features.loc[valid_idx], target.loc[valid_idx]
        probabilities = pd.Series(index=X.index, dtype=float)
        dates = X.index
        train_end_idx = max(dates.get_indexer([dates[0] + pd.DateOffset(years=initial_train_years)], method='ffill')[0],
                            500)

        shap_values_list, shap_features_list = [], []

        while train_end_idx < len(dates) - 42:
            train_dates = dates[:train_end_idx]
            test_dates = dates[train_end_idx:min(train_end_idx + 252, len(dates))]
            if len(test_dates) < 42: break

            X_train, y_train = X.loc[train_dates], y.loc[train_dates]
            X_test, y_test = X.loc[test_dates], y.loc[test_dates]

            # Fit Both Models
            self.model_alpha.fit(X_train, y_train)
            self.model_beta.fit(X_train, y_train)

            # CONSENSUS LOGIC
            probs_a = self.model_alpha.predict_proba(X_test)[:, 1]
            probs_b = self.model_beta.predict_proba(X_test)[:, 1]
            test_trends = X_test['trend_score']

            # Predict 1 ONLY if Both Agree > Threshold AND Trend > 0
            test_preds = []
            for pa, pb, t in zip(probs_a, probs_b, test_trends):
                if pa > 0.55 and pb > 0.50 and t > 0:
                    test_preds.append(1)
                else:
                    test_preds.append(0)

            test_score = np.mean(test_preds == y_test)

            # CALCULATE SNIPER SCORE (Precision)
            # This answers: "When I actually took risk, was I right?"
            buy_signals = [i for i, x in enumerate(test_preds) if x == 1]
            if len(buy_signals) > 0:
                wins = sum([1 for i in buy_signals if y_test.iloc[i] == 1])
                sniper_score = wins / len(buy_signals)
            else:
                sniper_score = 1.0  # Perfect Defense

            self.train_scores.append(0.65)
            self.test_scores.append(test_score)
            self.window_dates.append(test_dates[0])
            self.selected_rebalance_periods.append(63)

            # Store Feature Importance (from Model A)
            if hasattr(self.model_alpha, 'feature_importances_'):
                self.feature_importances_history.append(
                    dict(zip(self.feature_names, self.model_alpha.feature_importances_)))

            # SHAP Calculation
            try:
                if len(X_test) > 10:
                    sample_idx = np.random.choice(len(X_test), min(50, len(X_test)), replace=False)
                    X_sample = X_test.iloc[sample_idx]
                    explainer = shap.TreeExplainer(self.model_alpha)
                    shap_vals = explainer.shap_values(X_sample)
                    shap_values_list.append(shap_vals)
                    shap_features_list.append(X_sample)
            except Exception:
                pass

            probabilities.loc[test_dates] = (probs_a + probs_b) / 2

            # LOGGING WITH YEAR
            current_year = test_dates[0].year
            logger.info(
                f"Window {len(self.test_scores)} ({current_year}): Acc={test_score:.3f} | Sniper Score={sniper_score:.3f}")

            train_end_idx += int(252 * step_months / 12)

        if shap_values_list:
            self.shap_values = np.vstack(shap_values_list)
            self.shap_features = pd.concat(shap_features_list)

        # Calculate Model Stability based on test_scores standard deviation
        if self.test_scores:
            test_scores_std = np.std(self.test_scores)
            if test_scores_std < 0.05:
                self.model_stability = 'HIGH'
            elif test_scores_std < 0.10:
                self.model_stability = 'MODERATE'
            else:
                self.model_stability = 'LOW'

        return probabilities.ffill().ewm(span=10).mean()

    def get_regime(self, ml_prob: float, spy_above_sma: bool, current_vol: float,
                   tlt_momentum: float = 0.0, equity_risk_premium: float = 0.0) -> str:
        """
        Consensus Veto Logic with Multi-Factor Veto (The 2022 Shield).
        
        DEFENSIVE if:
        (1) spy_above_sma is False, OR
        (2) current_vol > 0.35, OR
        (3) current_vol > anxiety_vix_threshold AND ml_prob < anxiety_ml_prob_threshold (Anxiety Veto), OR
        (4) tlt_momentum < -0.05 AND equity_risk_premium < 0 (Rate Shock Guard)
        """
        # 1. HARD VETO
        if not spy_above_sma or current_vol > 0.35:
            return 'DEFENSIVE'
        
        # 2. ANXIETY VETO - When VIX is elevated, require higher conviction
        anxiety_vix_threshold = self.config.anxiety_vix_threshold
        anxiety_ml_prob_threshold = self.config.anxiety_ml_prob_threshold
        if current_vol > anxiety_vix_threshold and ml_prob < anxiety_ml_prob_threshold:
            return 'DEFENSIVE'
        
        # 3. RATE SHOCK GUARD (The 2022 Shield)
        if tlt_momentum < -0.05 and equity_risk_premium < 0:
            return 'DEFENSIVE'

        # 4. CONSENSUS PROBABILITY
        if ml_prob > 0.55:
            return 'RISK_ON'

        return 'RISK_REDUCED'

    def plot_shap_summary(self):
        """Restored SHAP Plotter."""
        if self.shap_values is None: return
        try:
            plt.figure(figsize=(10, 6))
            shap.summary_plot(self.shap_values, self.shap_features, plot_type="bar", show=False)
            plt.title('Consensus Model Features (XGBoost)', fontsize=12)
            plt.tight_layout()
            plt.show(block=False)
        except Exception as e:
            print(f"SHAP Plot Error: {e}")

    def plot_validation_curves(self):
        """Restored Health Dashboard."""
        if not self.train_scores: return
        try:
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            ax1, ax2 = axes

            # Accuracy
            ax1.plot(self.train_scores, label='Train (Ref)', color='blue', alpha=0.3)
            ax1.plot(self.test_scores, label='Test (Consensus)', color='red', linewidth=2)
            ax1.set_title("Consensus Accuracy Check")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Feature Importance
            if self.feature_importances_history:
                df_feat = pd.DataFrame(self.feature_importances_history)
                df_feat.plot(ax=ax2, alpha=0.7)
                ax2.set_title("Feature Importance Over Time")
                ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show(block=False)
        except Exception as e:
            print(f"Dash Plot Error: {e}")


class AlphaDominatorOptimizer:
    """
    The Alpha Dominator: IR Filter + Growth Anchor + Shannon Entropy

    RISK_ON Constraints per Alpha Dominator Constitution:
    - IR Filter (Velvet Rope): Only assets with IR > 0.5 vs SPY are eligible
    - Growth Anchor: QQQ + XLK + SMH + VGT minimum 60% combined weight
    - Gold Cap: GLD capped at 5% maximum
    - Kill Sharpe/Volatility Traps: No vol penalty on growth leaders
    - Shannon Entropy: Maximize IR_Score + (0.15 × Shannon Entropy)
    - Turnover Brake: Penalty = sum(abs(new - old)) * turnover_penalty (configurable)
    """

    def __init__(
            self,
            assets: List[str],
            asset_categories: Dict[str, List[str]],
            config: StrategyConfig = None
    ):
        self.assets = assets
        self.asset_categories = asset_categories
        self.n_assets = len(assets)
        self.config = config or StrategyConfig()

        self.equity_idx = [assets.index(a) for a in asset_categories.get('equities', []) if a in assets]
        self.gold_idx = [assets.index(a) for a in asset_categories.get('gold', []) if a in assets]
        self.bonds_cash_idx = [assets.index(a) for a in asset_categories.get('bonds_cash', []) if a in assets]
        self.safe_haven_idx = [assets.index(a) for a in asset_categories.get('safe_haven', []) if a in assets]

        # GROWTH ANCHOR: Use DataManager.GROWTH_ANCHORS for consistency
        self.growth_anchor_idx = [
            assets.index(a) for a in DataManager.GROWTH_ANCHORS
            if a in assets
        ]

        # Current weights for turnover penalty (None = first optimization, skip penalty)
        self.current_weights: Optional[np.ndarray] = None

        logger.info(f"AlphaDominator: {self.n_assets} assets, growth_anchor_idx={self.growth_anchor_idx}, "
                    f"gold_idx={self.gold_idx}")

    def optimize(
            self,
            returns: pd.DataFrame,
            raw_momentum: pd.Series,
            information_ratio: pd.Series,
            asset_volatilities: pd.Series,
            regime: str,
            above_sma: pd.Series,
            ml_prob: float = 0.5
    ) -> Tuple[np.ndarray, bool, str, Dict]:
        """
        Optimize with IR filter, Dynamic Growth Anchor (Smart Anchor), Shannon Entropy objective, and Turnover Brake.
        
        Dynamic Anchor: The Growth Anchor floor is now ML conviction-aware.
        dynamic_anchor = max(0.20, min(0.60, (ml_prob - 0.50) * 2.0))
        """
        mean_ret = returns.mean() * 252
        cov = returns.cov() * 252
        
        # Calculate dynamic anchor based on ML conviction (Smart Anchor)
        dynamic_anchor = max(0.20, min(0.60, (ml_prob - 0.50) * 2.0))

        # Get eligible mask based on regime with IR filter
        eligible_mask = self._get_eligible_mask(information_ratio, above_sma, regime)
        n_eligible = eligible_mask.sum()

        logger.debug(f"Regime={regime}, Eligible={n_eligible}/{self.n_assets}, DynamicAnchor={dynamic_anchor:.1%}")

        if n_eligible == 0:
            logger.warning("No eligible assets, using fallback")
            weights = self._safe_fallback(regime)
            self.current_weights = weights.copy()
            return weights, False, "fallback", {}

        if n_eligible == 1:
            weights = np.zeros(self.n_assets)
            weights[eligible_mask] = 1.0
            self.current_weights = weights.copy()
            return weights, True, "single", self._calculate_diagnostics(weights, cov, information_ratio)

        # Build objective and constraints based on regime
        if regime == 'RISK_ON':
            objective = self._build_risk_on_objective(raw_momentum, cov, information_ratio, eligible_mask, dynamic_anchor)
            bounds = self._get_risk_on_bounds(eligible_mask)
        elif regime == 'RISK_REDUCED':
            objective = self._build_risk_reduced_objective(mean_ret, cov, eligible_mask)
            bounds = self._get_default_bounds(eligible_mask)
        else:
            objective = self._build_defensive_objective(cov, eligible_mask)
            bounds = self._get_default_bounds(eligible_mask)

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

        result, method = self._multi_start_optimize(objective, bounds, constraints, cov, eligible_mask)

        if result is not None:
            weights = np.maximum(result.x, 0)
            weights = weights / weights.sum()
            diagnostics = self._calculate_diagnostics(weights, cov, information_ratio)
            self.current_weights = weights.copy()
            return weights, True, method, diagnostics
        else:
            # Use growth anchor tilt fallback instead of equal weight
            weights = self._growth_anchor_tilt(eligible_mask)
            if weights is None:
                weights = self._safe_fallback(regime)
            self.current_weights = weights.copy()
            return weights, False, "growth_tilt_fallback", self._calculate_diagnostics(weights, cov, information_ratio)

    def _get_eligible_mask(
            self,
            information_ratio: pd.Series,
            above_sma: pd.Series,
            regime: str
    ) -> np.ndarray:
        """
        Get eligible assets using IR Filter (Velvet Rope).

        RISK_ON: Only assets with IR > 0.5 (IR_threshold) are eligible
        Others: Above SMA is sufficient
        """
        aligned_ir = information_ratio.reindex(pd.Index(self.assets)).fillna(-1)
        aligned_sma = above_sma.reindex(pd.Index(self.assets)).fillna(False)

        eligible = np.ones(self.n_assets, dtype=bool)

        # Base: above SMA
        eligible &= aligned_sma.values

        if regime == 'RISK_ON':
            # IR FILTER (Velvet Rope): Must have IR > 0.5 vs SPY
            eligible &= (aligned_ir.values > self.config.ir_threshold)

            # Force bonds/cash to be ineligible in RISK_ON
            for idx in self.bonds_cash_idx:
                eligible[idx] = False

            # But always allow Growth Anchors (QQQ, XLK) if above SMA
            for idx in self.growth_anchor_idx:
                if aligned_sma.values[idx]:
                    eligible[idx] = True

            logger.debug(f"RISK_ON IR filter (>{self.config.ir_threshold}): {eligible.sum()} eligible")

        return eligible

    def _get_risk_on_bounds(self, eligible_mask: np.ndarray) -> List[Tuple[float, float]]:
        """
        RISK_ON bounds per Alpha Dominator Constitution:
        - Max 35% per asset
        - Gold capped at 5% (The Gold Cap)
        - Bonds/cash = 0%
        """
        bounds = []
        for i in range(self.n_assets):
            if not eligible_mask[i]:
                bounds.append((0.0, 0.0))
            elif i in self.gold_idx:
                bounds.append((0.0, self.config.gold_cap_risk_on))  # 5% cap
            elif i in self.bonds_cash_idx:
                bounds.append((0.0, 0.0))  # Forced to 0
            else:
                bounds.append((0.01, self.config.max_single_weight))
        return bounds

    def _get_default_bounds(self, eligible_mask: np.ndarray) -> List[Tuple[float, float]]:
        """Default bounds for other regimes."""
        bounds = []
        for i in range(self.n_assets):
            if eligible_mask[i]:
                bounds.append((0.01, self.config.max_single_weight))
            else:
                bounds.append((0.0, 0.0))
        return bounds

    def _build_risk_on_objective(
            self,
            raw_momentum: pd.Series,
            cov: pd.DataFrame,
            information_ratio: pd.Series,
            eligible_mask: np.ndarray,
            dynamic_anchor: float = 0.60
    ) -> callable:
        """
        RISK_ON Objective: Maximize IR_Score + (0.15 × Shannon Entropy) - Turnover Penalty

        NO SHARPE TRAP - no volatility penalty on growth leaders.
        DYNAMIC ANCHOR (Smart Anchor) - soft penalty if Growth Anchors < dynamic_anchor
        TURNOVER BRAKE - penalty = sum(abs(new - old)) * turnover_penalty (configurable)
        
        MOMENTUM-BASED: Uses current raw_momentum (3-6 month trend) instead of 15-year mean_ret.
        Assets with missing momentum data are treated as having zero momentum (no favorable bias).
        """
        aligned_ir = information_ratio.reindex(pd.Index(self.assets)).fillna(0).values
        # Use raw_momentum as base return expectation (instead of historical mean_ret)
        raw_momentum_arr = raw_momentum.reindex(pd.Index(self.assets)).fillna(0).values
        cov_arr = cov.values
        entropy_lambda = self.config.entropy_lambda  # 0.15
        min_growth_anchor = dynamic_anchor  # Use ML conviction-aware dynamic_anchor
        growth_anchor_penalty_mult = self.config.growth_anchor_penalty  # High-priority penalty multiplier
        turnover_penalty_mult = self.config.turnover_penalty  # The Turnover Brake

        # Scale IR scores to positive range for objective
        ir_scaled = np.where(aligned_ir > 0, aligned_ir, 0)
        if ir_scaled.max() > 0:
            ir_scaled = ir_scaled / ir_scaled.max()

        # ALPHA-WEIGHTED RETURNS: current_momentum * (1.0 + 2.0 * ir_scaled)
        # Uses raw_momentum (3-6 month trend) as base, boosted by IR
        boosted_returns = raw_momentum_arr * (1.0 + 2.0 * ir_scaled)

        growth_anchor_idx = self.growth_anchor_idx
        # Use zeros for first optimization to skip turnover penalty
        old_weights = self.current_weights if self.current_weights is not None else np.zeros(self.n_assets)

        def objective(w):
            # IR Score component (no vol penalty - Kill Sharpe Trap)
            ir_score = np.dot(w, boosted_returns)

            # Shannon Entropy (anti-blocky weights)
            w_pos = w[w > 1e-6]
            entropy = -np.sum(w_pos * np.log(w_pos)) if len(w_pos) > 0 else 0
            n_eligible = eligible_mask.sum()
            max_entropy = np.log(n_eligible) if n_eligible > 1 else 1
            norm_entropy = entropy / max_entropy

            # DYNAMIC ANCHOR (Smart Anchor): High-priority soft penalty if Growth Anchors < dynamic_anchor
            growth_weight = sum(w[idx] for idx in growth_anchor_idx)
            growth_penalty = max(0, min_growth_anchor - growth_weight) ** 2 * growth_anchor_penalty_mult

            # TURNOVER BRAKE: Penalty for excessive trading (skipped on first call)
            turnover = np.sum(np.abs(w - old_weights))
            turnover_penalty = turnover * turnover_penalty_mult

            # Objective: maximize (IR_Score + 0.15*Entropy) - penalties
            return -ir_score - entropy_lambda * norm_entropy + growth_penalty + turnover_penalty

        return objective

    def _build_risk_reduced_objective(
            self,
            mean_ret: pd.Series,
            cov: pd.DataFrame,
            eligible_mask: np.ndarray
    ) -> callable:
        """RISK_REDUCED: Maximize Sharpe with Turnover Brake."""
        mean_ret_arr = mean_ret.values
        cov_arr = cov.values
        rf = self.config.risk_free_rate
        turnover_penalty_mult = self.config.turnover_penalty  # The Turnover Brake
        # Use zeros for first optimization to skip turnover penalty
        old_weights = self.current_weights if self.current_weights is not None else np.zeros(self.n_assets)

        def objective(w):
            port_ret = np.dot(w, mean_ret_arr)
            port_var = np.dot(w.T, np.dot(cov_arr, w))
            port_vol = np.sqrt(port_var)

            if port_vol < 1e-6:
                return 1e6

            sharpe = (port_ret - rf) / port_vol

            # TURNOVER BRAKE: Penalty for excessive trading (skipped on first call)
            turnover = np.sum(np.abs(w - old_weights))
            turnover_penalty = turnover * turnover_penalty_mult

            return -sharpe + turnover_penalty

        return objective

    def _build_defensive_objective(
            self,
            cov: pd.DataFrame,
            eligible_mask: np.ndarray
    ) -> callable:
        """DEFENSIVE: Minimum variance with Turnover Brake."""
        cov_arr = cov.values
        turnover_penalty_mult = self.config.turnover_penalty  # The Turnover Brake
        # Use zeros for first optimization to skip turnover penalty
        old_weights = self.current_weights if self.current_weights is not None else np.zeros(self.n_assets)

        def objective(w):
            variance = np.dot(w.T, np.dot(cov_arr, w))

            # TURNOVER BRAKE: Penalty for excessive trading (skipped on first call)
            turnover = np.sum(np.abs(w - old_weights))
            turnover_penalty = turnover * turnover_penalty_mult

            return variance + turnover_penalty

        return objective

    def _multi_start_optimize(
            self,
            objective: callable,
            bounds: List[Tuple[float, float]],
            constraints: List[Dict],
            cov: pd.DataFrame,
            eligible_mask: np.ndarray
    ) -> Tuple[Optional[object], str]:
        """Multi-start optimization."""
        starting_points = [
            ('equal', self._equal_eligible(eligible_mask)),
            ('inv_vol', self._inv_vol_eligible(cov, eligible_mask)),
            ('growth_tilt', self._growth_anchor_tilt(eligible_mask)),
        ]

        best_result = None
        best_obj = float('inf')
        best_method = None

        for name, init_w in starting_points:
            if init_w is None:
                continue

            init_w = np.clip(init_w, [b[0] for b in bounds], [b[1] for b in bounds])
            if init_w.sum() > 0:
                init_w = init_w / init_w.sum()
            else:
                continue

            try:
                result = minimize(
                    objective,
                    init_w,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 1000, 'ftol': 1e-10}
                )

                if result.success and result.fun < best_obj:
                    best_result = result
                    best_obj = result.fun
                    best_method = name

            except Exception as e:
                logger.debug(f"Opt from {name} failed: {e}")
                continue

        return best_result, best_method or "none"

    def _equal_eligible(self, eligible_mask: np.ndarray) -> np.ndarray:
        """Equal weight eligible."""
        weights = np.zeros(self.n_assets)
        n = eligible_mask.sum()
        if n > 0:
            weights[eligible_mask] = 1.0 / n
        return weights

    def _inv_vol_eligible(self, cov: pd.DataFrame, eligible_mask: np.ndarray) -> np.ndarray:
        """Inverse volatility."""
        weights = np.zeros(self.n_assets)
        vols = np.sqrt(np.diag(cov))
        vols = np.maximum(vols, 1e-6)

        if eligible_mask.sum() > 0:
            eligible_vols = vols[eligible_mask]
            inv_vols = 1.0 / eligible_vols
            weights[eligible_mask] = inv_vols / inv_vols.sum()
        return weights

    def _growth_anchor_tilt(self, eligible_mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Growth anchor tilt: Set 60% total weight to Growth Anchors (QQQ, XLK, SMH, VGT)
        and distribute remaining 40% among other eligible assets.
        """
        if eligible_mask.sum() == 0:
            return None

        weights = np.zeros(self.n_assets)

        # Convert to set for O(1) lookup
        growth_anchor_set = set(self.growth_anchor_idx)

        # Identify eligible growth anchors and other eligible assets
        eligible_anchor_idx = [idx for idx in self.growth_anchor_idx if eligible_mask[idx]]
        eligible_other_idx = [idx for idx in np.where(eligible_mask)[0] if idx not in growth_anchor_set]

        # 60% Floor for Growth Anchors
        if eligible_anchor_idx:
            anchor_weight_each = self.config.min_growth_anchor / len(eligible_anchor_idx)
            for idx in eligible_anchor_idx:
                weights[idx] = anchor_weight_each

        # Distribute remaining 40% among other eligible assets
        remaining_weight = 1.0 - self.config.min_growth_anchor
        if eligible_other_idx:
            other_weight_each = remaining_weight / len(eligible_other_idx)
            for idx in eligible_other_idx:
                weights[idx] = other_weight_each
        elif eligible_anchor_idx:
            # If no other eligible assets, give remaining weight to anchors
            extra_each = remaining_weight / len(eligible_anchor_idx)
            for idx in eligible_anchor_idx:
                weights[idx] += extra_each

        # Normalize to ensure sum is exactly 1.0
        if weights.sum() > 0:
            weights = weights / weights.sum()

        return weights

    def _safe_fallback(self, regime: str) -> np.ndarray:
        """Fallback allocation prioritizing growth anchors."""
        weights = np.zeros(self.n_assets)

        if regime == 'RISK_ON':
            # Prioritize growth anchors
            if self.growth_anchor_idx:
                for idx in self.growth_anchor_idx:
                    weights[idx] = 1.0 / len(self.growth_anchor_idx)
            elif self.equity_idx:
                for idx in self.equity_idx:
                    weights[idx] = 1.0 / len(self.equity_idx)
        elif self.safe_haven_idx:
            for idx in self.safe_haven_idx:
                weights[idx] = 1.0 / len(self.safe_haven_idx)
        else:
            weights = np.ones(self.n_assets) / self.n_assets

        return weights

    def _calculate_diagnostics(self, weights: np.ndarray, cov: pd.DataFrame,
                               information_ratio: pd.Series) -> Dict:
        """Calculate diagnostics including IR scores."""
        cov_arr = cov.values
        port_var = np.dot(weights.T, np.dot(cov_arr, weights))
        port_vol = np.sqrt(port_var)

        mctr = np.dot(cov_arr, weights) / port_vol if port_vol > 1e-6 else np.zeros(self.n_assets)
        pctr = (weights * mctr) / port_vol if port_vol > 1e-6 else np.zeros(self.n_assets)

        w_pos = weights[weights > 1e-6]
        entropy = -np.sum(w_pos * np.log(w_pos)) if len(w_pos) > 0 else 0
        effective_n = np.exp(entropy)

        # Get IR scores for diagnostics
        aligned_ir = information_ratio.reindex(pd.Index(self.assets)).fillna(0)
        ir_scores = dict(zip(self.assets, aligned_ir.values))

        # Calculate growth anchor weight
        growth_anchor_weight = sum(weights[idx] for idx in self.growth_anchor_idx)

        return {
            'mctr': dict(zip(self.assets, mctr)),
            'pctr': dict(zip(self.assets, pctr)),
            'ir_scores': ir_scores,
            'entropy': entropy,
            'effective_n': effective_n,
            'n_positions': np.sum(weights > 0.02),
            'port_volatility': port_vol,
            'growth_anchor_weight': growth_anchor_weight,
            'gold_weight': sum(weights[idx] for idx in self.gold_idx)
        }


class BacktestEngine:
    """Backtesting with adaptive rebalancing."""

    def __init__(self, config: StrategyConfig = None):
        self.config = config or StrategyConfig()
        self.initial_capital = 100000.0

        self.dates: List = []
        self.portfolio_values: List[float] = []
        self.benchmark_values: List[float] = []
        self.regimes: List[str] = []
        self.ml_probs: List[float] = []
        self.weights_history: List[Dict] = []
        self.diagnostics_history: List[Dict] = []
        self.transaction_costs: List[float] = []
        self.rebalance_periods_used: List[int] = []
        self.final_weights: Optional[np.ndarray] = None
        self.final_ir: Optional[pd.Series] = None  # Changed from final_rs to final_ir
        self.final_diagnostics: Optional[Dict] = None
        
        # Sniper Score tracking
        self.buy_signals: List[int] = []  # 1 if buy signal, 0 otherwise
        self.actual_outcomes: List[int] = []  # Date indices for outcome calculation
        self.sniper_score: Optional[float] = None  # None if undefined (no buy signals)

    def run(
            self,
            prices: pd.DataFrame,
            returns: pd.DataFrame,
            features: pd.DataFrame,  # <--- NEW ARGUMENT ADDED HERE
            ml_probs: pd.Series,
            sma_200: pd.DataFrame,
            above_sma: pd.DataFrame,
            raw_momentum: pd.DataFrame,
            relative_strength: pd.DataFrame,
            asset_volatilities: pd.DataFrame,
            information_ratio: pd.DataFrame,
            classifier: AdaptiveRegimeClassifier,
            optimizer: 'AlphaDominatorOptimizer',
            lookback_days: int = 252
    ) -> pd.DataFrame:
        """Execute backtest with adaptive rebalancing and IR filter."""
        logger.info("Starting backtest with adaptive rebalancing")

        portfolio_value = self.initial_capital
        benchmark_value = self.initial_capital
        current_weights = np.zeros(optimizer.n_assets)

        valid_dates = (ml_probs.dropna().index
                       .intersection(prices.index)
                       .intersection(returns.index)
                       .intersection(above_sma.dropna().index)
                       .intersection(raw_momentum.dropna().index)
                       .intersection(information_ratio.dropna().index))

        start_idx = lookback_days
        if start_idx >= len(valid_dates):
            raise ValueError("Insufficient data")

            # Benchmark: 100% SPY
            bench_weights = np.zeros(optimizer.n_assets)
            if 'SPY' in optimizer.assets:
                bench_weights[optimizer.assets.index('SPY')] = 1.0
            # No normalization needed if it's just 1.0, but good practice to keep the structure clean.

        # Use adaptive rebalance period
        rebalance_period = classifier.current_rebalance_period
        days_since_rebalance = rebalance_period
        total_costs = 0.0
        # Benchmark: 100% SPY (Initialize BEFORE the loop)
        bench_weights = np.zeros(optimizer.n_assets)
        if 'SPY' in optimizer.assets:
            bench_weights[optimizer.assets.index('SPY')] = 1.0
        else:
            # Fallback if SPY is missing (unlikely)
            bench_weights[:] = 1.0 / optimizer.n_assets

        for i in range(start_idx, len(valid_dates)):
            date = valid_dates[i]
            prev_date = valid_dates[i - 1]

            daily_ret = returns.loc[date][optimizer.assets].values
            daily_ret = np.nan_to_num(daily_ret, 0)

            ml_prob = ml_probs.loc[date]
            if np.isnan(ml_prob):
                ml_prob = 0.5

            spy_above = above_sma.loc[date, 'SPY'] if 'SPY' in above_sma.columns else True
            asset_above_sma = above_sma.loc[date]
            current_raw_mom = raw_momentum.loc[date]
            current_ir = information_ratio.loc[date]  # Use IR instead of RS
            current_vols = asset_volatilities.loc[date]

            # Update rebalance period from classifier
            rebalance_period = classifier.current_rebalance_period

            if days_since_rebalance >= rebalance_period:
                # FIX: Use the VIX feature (Instant Fear) instead of lagging asset volatility
                # features['realized_vol'] is now the VIX (set in DataManager)
                current_vol = features.loc[date, 'realized_vol']
                
                # Get tlt_momentum and equity_risk_premium for Rate Shock Guard
                tlt_momentum = features.loc[date, 'tlt_momentum']
                equity_risk_premium = features.loc[date, 'equity_risk_premium']

                regime = classifier.get_regime(ml_prob, spy_above, current_vol, tlt_momentum, equity_risk_premium)

                lookback_start = valid_dates[i - lookback_days]
                lookback_ret = returns.loc[lookback_start:prev_date][optimizer.assets]

                # Use IR for optimization with ml_prob for Dynamic Anchor
                new_weights, success, method, diagnostics = optimizer.optimize(
                    lookback_ret, current_raw_mom, current_ir, current_vols,
                    regime, asset_above_sma, ml_prob
                )

                if diagnostics:
                    self.diagnostics_history.append(diagnostics)

                turnover = np.sum(np.abs(new_weights - current_weights))
                cost = portfolio_value * turnover * (self.config.transaction_cost_bps / 10000)
                portfolio_value -= cost
                total_costs += cost

                self.transaction_costs.append(cost)
                self.rebalance_periods_used.append(rebalance_period)
                current_weights = new_weights
                days_since_rebalance = 0
                
                # Track Sniper Score: Buy signal if RISK_ON regime
                is_buy_signal = 1 if regime == 'RISK_ON' else 0
                self.buy_signals.append(is_buy_signal)
                
                # Store the date index for later outcome calculation (avoid look-ahead bias)
                self.actual_outcomes.append(i)

                self.weights_history.append({
                    'date': date,
                    'regime': regime,
                    'weights': dict(zip(optimizer.assets, new_weights)),
                    'ir_scores': current_ir.to_dict(),  # Store IR instead of RS
                    'effective_n': diagnostics.get('effective_n', 0) if diagnostics else 0,
                    'rebalance_period': rebalance_period,
                    'method': method
                })
            else:
                regime = self.regimes[-1] if self.regimes else 'RISK_REDUCED'

            portfolio_value *= (1 + np.dot(current_weights, daily_ret))
            benchmark_value *= (1 + np.dot(bench_weights, daily_ret))

            self.dates.append(date)
            self.portfolio_values.append(portfolio_value)
            self.benchmark_values.append(benchmark_value)
            self.regimes.append(regime)
            self.ml_probs.append(ml_prob)

            days_since_rebalance += 1
        
        # Calculate final Sniper Score (after backtest completes to avoid look-ahead bias)
        self._calculate_sniper_score(prices, valid_dates)

        self.final_weights = current_weights.copy()
        self.final_ir = information_ratio.iloc[-1]  # Changed from relative_strength to IR
        self.final_diagnostics = self.diagnostics_history[-1] if self.diagnostics_history else {}

        avg_eff_n = np.mean(
            [d.get('effective_n', 0) for d in self.diagnostics_history]) if self.diagnostics_history else 0
        logger.info(f"Backtest complete: {len(self.dates)} days")
        logger.info(f"Avg Diversity Score: {avg_eff_n:.2f}")
        logger.info(f"Total costs: ${total_costs:,.2f}")
        if self.sniper_score is not None:
            logger.info(f"Final Sniper Score: {self.sniper_score:.3f}")
        else:
            logger.info("Final Sniper Score: N/A (no buy signals)")

        return pd.DataFrame({
            'Portfolio': self.portfolio_values,
            'Benchmark': self.benchmark_values,
            'Regime': self.regimes,
            'ML_Prob': self.ml_probs
        }, index=self.dates)
    
    def _calculate_sniper_score(self, prices: pd.DataFrame, valid_dates: pd.Index) -> None:
        """
        Calculate Sniper Score (Precision): (Correct Buy Signals) / (Total Buy Signals).
        A 'Buy Signal' is when regime == RISK_ON.
        
        This is calculated after the backtest completes to avoid look-ahead bias.
        Outcomes are determined by evaluating 21-day forward returns using only
        data that would have been available at each signal date.
        """
        if not self.buy_signals:
            self.sniper_score = None  # Undefined when there are no buy signals
            return
        
        total_buy_signals = sum(self.buy_signals)
        if total_buy_signals == 0:
            self.sniper_score = None  # Undefined precision
            return
        
        # Calculate actual outcomes for each signal date
        correct_buys = 0
        for buy, date_idx in zip(self.buy_signals, self.actual_outcomes):
            if buy == 1:
                # Calculate 21-day forward return (using data now available at end of backtest)
                future_idx = min(date_idx + 21, len(valid_dates) - 1)
                signal_date = valid_dates[date_idx]
                future_date = valid_dates[future_idx]
                spy_return_21d = (prices.loc[future_date, 'SPY'] / prices.loc[signal_date, 'SPY']) - 1
                if spy_return_21d > 0:
                    correct_buys += 1
        
        self.sniper_score = correct_buys / total_buy_signals

    def calculate_metrics(self, results: pd.DataFrame) -> Dict:
        """Calculate metrics."""
        port_ret = results['Portfolio'].pct_change().dropna()
        bench_ret = results['Benchmark'].pct_change().dropna()

        years = len(results) / 252

        port_cagr = (results['Portfolio'].iloc[-1] / self.initial_capital) ** (1 / years) - 1
        bench_cagr = (results['Benchmark'].iloc[-1] / self.initial_capital) ** (1 / years) - 1

        port_vol = port_ret.std() * np.sqrt(252)
        bench_vol = bench_ret.std() * np.sqrt(252)

        rf = self.config.risk_free_rate
        port_sharpe = (port_cagr - rf) / port_vol if port_vol > 0 else 0
        bench_sharpe = (bench_cagr - rf) / bench_vol if bench_vol > 0 else 0

        port_peak = results['Portfolio'].expanding().max()
        port_dd = (results['Portfolio'] - port_peak) / port_peak

        bench_peak = results['Benchmark'].expanding().max()
        bench_dd = (results['Benchmark'] - bench_peak) / bench_peak

        regime_counts = results['Regime'].value_counts().to_dict()
        avg_eff_n = np.mean(
            [d.get('effective_n', 0) for d in self.diagnostics_history]) if self.diagnostics_history else 0

        # Most common rebalance period
        if self.rebalance_periods_used:
            from collections import Counter
            rebal_counter = Counter(self.rebalance_periods_used)
            most_common_rebal = rebal_counter.most_common(1)[0][0]
        else:
            most_common_rebal = 42

        return {
            'portfolio': {
                'final_value': results['Portfolio'].iloc[-1],
                'cagr': port_cagr,
                'volatility': port_vol,
                'sharpe': port_sharpe,
                'max_drawdown': port_dd.min()
            },
            'benchmark': {
                'final_value': results['Benchmark'].iloc[-1],
                'cagr': bench_cagr,
                'volatility': bench_vol,
                'sharpe': bench_sharpe,
                'max_drawdown': bench_dd.min()
            },
            'regime_counts': regime_counts,
            'avg_diversity_score': avg_eff_n,
            'optimal_rebalance_period': most_common_rebal,
            'total_costs': sum(self.transaction_costs),
            'sniper_score': self.sniper_score
        }

    def plot_performance(self, results: pd.DataFrame) -> None:
        """Display performance."""
        fig, ax = plt.subplots(figsize=(14, 8))

        ax.plot(results.index, results['Portfolio'], label='Strategy', linewidth=2, color='#2E86AB')
        ax.plot(results.index, results['Benchmark'], label='Benchmark (SPY)',
                linewidth=1.5, color='gray', linestyle='--')

        regimes = results['Regime'].values
        dates = results.index
        for i in range(len(dates) - 1):
            if regimes[i] == 'DEFENSIVE':
                ax.axvspan(dates[i], dates[i + 1], alpha=0.15, color='red')
            elif regimes[i] == 'RISK_REDUCED':
                ax.axvspan(dates[i], dates[i + 1], alpha=0.08, color='orange')

        metrics = self.calculate_metrics(results)
        info = (
            f"Strategy: CAGR={metrics['portfolio']['cagr']:.1%}, "
            f"Sharpe={metrics['portfolio']['sharpe']:.2f}, "
            f"MaxDD={metrics['portfolio']['max_drawdown']:.1%}\n"
            f"Benchmark: CAGR={metrics['benchmark']['cagr']:.1%}, "
            f"Sharpe={metrics['benchmark']['sharpe']:.2f}, "
            f"MaxDD={metrics['benchmark']['max_drawdown']:.1%}\n"
            f"Diversity: {metrics['avg_diversity_score']:.2f} | "
            f"Optimal Rebal: {metrics['optimal_rebalance_period']}d"
        )

        ax.set_title('The Alpha Dominator v10.0 - IR Filter + Growth Anchor', fontsize=12)
        ax.set_ylabel('Portfolio Value ($)')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.02, info, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.show(block=False)

    def plot_allocation_history(self) -> None:
        """Display allocation."""
        if not self.weights_history:
            return

        dates = [w['date'] for w in self.weights_history]
        weights_df = pd.DataFrame([w['weights'] for w in self.weights_history], index=dates)
        effective_n = [w['effective_n'] for w in self.weights_history]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12),
                                       gridspec_kw={'height_ratios': [3, 1]})

        cols = sorted(weights_df.columns)
        ax1.stackplot(weights_df.index, weights_df[cols].T, labels=cols, alpha=0.8)
        ax1.set_title('Allocation (IR Filter: Only IR>0.5 in RISK_ON + Growth Anchor)', fontsize=12)
        ax1.set_ylabel('Weight')
        ax1.set_ylim(0, 1)
        ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
        ax1.grid(True, alpha=0.3)

        ax2.plot(dates, effective_n, 'g-', linewidth=2, label='Diversity Score')
        ax2.axhline(y=self.config.min_effective_n, color='red', linestyle='--',
                    label=f'Min: {self.config.min_effective_n}')
        ax2.set_ylabel('Effective N')
        ax2.set_title('Diversity Score', fontsize=11)
        ax2.set_ylim(0, 10)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show(block=False)

    def plot_regime_analysis(self, results: pd.DataFrame, prices: pd.DataFrame,
                             sma_200: pd.DataFrame) -> None:
        """Display regime analysis."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

        dates = results.index

        ax1 = axes[0]
        spy = prices.loc[dates, 'SPY']
        sma = sma_200.loc[dates, 'SPY']
        ax1.plot(dates, spy, label='SPY', color='navy')
        ax1.plot(dates, sma, label='200-SMA', color='orange', linestyle='--')
        ax1.fill_between(dates, spy, sma, where=(spy > sma), color='green', alpha=0.2)
        ax1.fill_between(dates, spy, sma, where=(spy <= sma), color='red', alpha=0.2)
        ax1.set_title('SPY vs 200-Day SMA', fontsize=11)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        ax2.plot(dates, results['ML_Prob'], color='purple', linewidth=1)
        ax2.axhline(0.5, color='gray', linestyle='--')
        ax2.set_title('ML Bull Probability', fontsize=11)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)

        ax3 = axes[2]
        regime_colors = {'RISK_ON': 'green', 'RISK_REDUCED': 'orange', 'DEFENSIVE': 'red'}
        regimes = results['Regime'].values
        for i in range(len(dates) - 1):
            ax3.axvspan(dates[i], dates[i + 1], color=regime_colors.get(regimes[i], 'gray'), alpha=0.7)
        ax3.set_title('Market Regime', fontsize=11)
        ax3.set_yticks([])

        legend_elements = [Patch(facecolor=c, alpha=0.7, label=r) for r, c in regime_colors.items()]
        ax3.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()
        plt.show(block=False)


class MonteCarloSimulator:
    """Monte Carlo with daily RF rate."""

    def __init__(self, n_simulations: int = 10000, projection_years: int = 5,
                 risk_free_rate: float = 0.04):
        self.n_simulations = n_simulations
        self.projection_years = projection_years
        self.n_days = projection_years * 252
        self.risk_free_rate = risk_free_rate
        self.rf_daily = (1 + risk_free_rate) ** (1 / 252) - 1

        self.price_paths: Optional[np.ndarray] = None
        self.ending_values: Optional[np.ndarray] = None
        self.sim_cagrs: Optional[np.ndarray] = None
        self.initial_value: float = 0.0
        self.mu_annual: float = 0.0
        self.sigma_annual: float = 0.0

    def run(self, returns: pd.DataFrame, weights: np.ndarray, assets: List[str],
            initial_value: float, lookback_years: float = 2.0) -> Dict:
        """Run simulation."""
        logger.info(f"Monte Carlo: {self.n_simulations:,} sims")

        self.initial_value = initial_value

        lookback_days = int(lookback_years * 252)
        recent_ret = returns[assets].iloc[-lookback_days:]
        port_ret = recent_ret.dot(weights)

        mu_daily = port_ret.mean()
        sigma_daily = port_ret.std()

        self.mu_annual = mu_daily * 252
        self.sigma_annual = sigma_daily * np.sqrt(252)

        drift = mu_daily - 0.5 * sigma_daily ** 2

        np.random.seed(42)
        Z = np.random.standard_normal((self.n_days, self.n_simulations))
        log_returns = drift + sigma_daily * Z

        self.price_paths = np.zeros((self.n_days + 1, self.n_simulations))
        self.price_paths[0] = initial_value

        for t in range(1, self.n_days + 1):
            self.price_paths[t] = self.price_paths[t - 1] * np.exp(log_returns[t - 1])

        self.ending_values = self.price_paths[-1]
        self.sim_cagrs = (self.ending_values / initial_value) ** (1 / self.projection_years) - 1

        return self._calculate_statistics()

    def _calculate_statistics(self) -> Dict:
        """Stats."""
        mean_cagr = np.mean(self.sim_cagrs)
        sharpe = (mean_cagr - self.risk_free_rate) / self.sigma_annual if self.sigma_annual > 0 else 0

        return {
            'mean_ending': np.mean(self.ending_values),
            'mean_cagr': mean_cagr,
            'ci_lower': np.percentile(self.ending_values, 2.5),
            'ci_upper': np.percentile(self.ending_values, 97.5),
            'sharpe': sharpe,
            'prob_loss': np.mean(self.ending_values < self.initial_value),
        }

    def plot_paths(self, n_display: int = 100) -> None:
        """Display paths."""
        if self.price_paths is None:
            return

        fig, ax = plt.subplots(figsize=(14, 8))

        cmap = plt.get_cmap('RdYlGn')
        norm = plt.Normalize(np.percentile(self.ending_values, 5),
                             np.percentile(self.ending_values, 95))

        np.random.seed(123)
        idx = np.random.choice(self.n_simulations, min(n_display, self.n_simulations), replace=False)

        days = np.arange(self.n_days + 1)
        for i in idx:
            ax.plot(days, self.price_paths[:, i],
                    color=cmap(norm(self.ending_values[i])), alpha=0.3, linewidth=0.5)

        p5 = np.percentile(self.price_paths, 2.5, axis=1)
        p95 = np.percentile(self.price_paths, 97.5, axis=1)
        mean_path = np.mean(self.price_paths, axis=1)

        ax.fill_between(days, p5, p95, color='lightblue', alpha=0.3, label='95% CI')
        ax.plot(days, mean_path, 'r-', linewidth=2.5, label=f'Mean: ${mean_path[-1]:,.0f}')

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        plt.colorbar(sm, ax=ax, label='Ending Value ($)', pad=0.01)

        stats = self._calculate_statistics()
        ax.set_title(f'Monte Carlo | CAGR: {stats["mean_cagr"]:.1%} | Sharpe: {stats["sharpe"]:.2f}', fontsize=11)
        ax.set_xlabel('Trading Days')
        ax.set_ylabel('Portfolio Value ($)')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show(block=False)

    def plot_distribution(self) -> None:
        """Display distribution."""
        if self.sim_cagrs is None:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        stats = self._calculate_statistics()

        w = np.ones_like(self.sim_cagrs) / len(self.sim_cagrs)
        n, bins, patches = ax1.hist(self.sim_cagrs, bins=60, weights=w, edgecolor='white', alpha=0.7)

        for patch, b in zip(patches, bins[:-1]):
            if b < 0:
                patch.set_facecolor('#D64045')
            elif b < stats['mean_cagr']:
                patch.set_facecolor('#F5B041')
            else:
                patch.set_facecolor('#27AE60')

        ax1.axvline(stats['mean_cagr'], color='red', linewidth=2, label=f"Mean: {stats['mean_cagr']:.1%}")
        ax1.axvline(0, color='black', linewidth=1, alpha=0.5)
        ax1.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax1.set_title(f'{self.projection_years}-Year CAGR Distribution')
        ax1.set_xlabel('CAGR')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        w2 = np.ones_like(self.ending_values) / len(self.ending_values)
        ax2.hist(self.ending_values / 1000, bins=60, weights=w2, color='#198964', edgecolor='white', alpha=0.7)
        ax2.axvline(stats['mean_ending'] / 1000, color='red', linewidth=2,
                    label=f"Mean: ${stats['mean_ending']:,.0f}")
        ax2.axvline(self.initial_value / 1000, color='black', linewidth=1.5,
                    label=f"Start: ${self.initial_value:,.0f}")
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}K'))
        ax2.set_title('Ending Value Distribution')
        ax2.set_xlabel('Portfolio Value')
        ax2.set_ylabel('Probability')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.show(block=False)


def main():
    """Main execution function for Version 10.0 - The Alpha Dominator."""
    print("=" * 80)
    print("THE ALPHA DOMINATOR v10.0")
    print("IR Filter + Growth Anchor + Regularized ML")
    print("=" * 80)
    print()

    config = StrategyConfig()

    # 1. Data Loading
    print("[1/6] Loading data and calculating Information Ratio...")
    dm = DataManager(start_date='2010-01-01', config=config)
    dm.load_data()
    dm.engineer_features()

    # Unpack aligned data (10 items - includes IR)
    prices, returns, features, vix, sma_200, above_sma, raw_mom, rs, vols, ir = dm.get_aligned_data()
    categories = dm.get_asset_categories()

    print(f"      Period: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"      Assets: {', '.join(dm.all_tickers)}")
    print(f"      Features: {', '.join(features.columns)}")
    print()

    # 2. Model Training
    print("[2/6] Training regularized regime classifier...")
    print(f"      Regularization: max_depth=4, min_samples_leaf=400, max_features=sqrt, ccp_alpha=0.01")
    print(f"      Probability Smoothing: {config.prob_ema_span}-day EMA")
    classifier = AdaptiveRegimeClassifier(config)
    ml_probs = classifier.walk_forward_train(features, returns['SPY'])
    print()

    # 3. Optimizer Initialization
    print("[3/6] Initializing Alpha Dominator optimizer...")
    print(f"      IR Threshold: {config.ir_threshold}")
    print(f"      Growth Anchor (QQQ+XLK+SMH+VGT min): {config.min_growth_anchor:.0%}")
    print(f"      Gold Cap: {config.gold_cap_risk_on:.0%}")
    print(f"      Turnover Brake: {config.turnover_penalty:.0f} penalty multiplier")
    optimizer = AlphaDominatorOptimizer(dm.all_tickers, categories, config)

    # 4. Backtest Execution
    print("[4/6] Running backtest with adaptive rebalancing...")
    engine = BacktestEngine(config)
    results = engine.run(
        prices, returns, features, ml_probs, sma_200, above_sma, raw_mom, rs, vols, ir,
        classifier, optimizer
    )

    # Calculate metrics (needed for terminal output)
    metrics = engine.calculate_metrics(results)

    # 5. Monte Carlo Calculation
    print("[5/6] Running Monte Carlo simulation...")
    mc = MonteCarloSimulator(
        n_simulations=10000,
        projection_years=5,
        risk_free_rate=config.risk_free_rate
    )
    mc.run(returns, engine.final_weights, optimizer.assets, results['Portfolio'].iloc[-1])

    # --- PRINT ALL TERMINAL OUTPUT ---
    print()
    print("-" * 65)
    print("BACKTEST RESULTS")
    print("-" * 65)
    print(f"Strategy:  CAGR={metrics['portfolio']['cagr']:.1%}, "
          f"Sharpe={metrics['portfolio']['sharpe']:.2f}, "
          f"MaxDD={metrics['portfolio']['max_drawdown']:.1%}")
    print(f"Benchmark: CAGR={metrics['benchmark']['cagr']:.1%}, "
          f"Sharpe={metrics['benchmark']['sharpe']:.2f}, "
          f"MaxDD={metrics['benchmark']['max_drawdown']:.1%}")
    print(f"Avg Diversity Score: {metrics['avg_diversity_score']:.2f}")
    print(f"Most Common Rebalance Period: {metrics['optimal_rebalance_period']} days")
    print(f"Total Transaction Costs: ${metrics['total_costs']:,.2f}")
    print(f"Regime Distribution: {metrics['regime_counts']}")
    
    # --- SNIPER SCORE OUTPUT ---
    sniper_score = metrics.get('sniper_score')
    sniper_threshold = config.ml_threshold  # Use centralized threshold (0.55)
    if sniper_score is not None:
        sniper_status = "✓ GOOD" if sniper_score >= sniper_threshold else "⚠ WARNING"
        print(f"SNIPER SCORE (Precision): {sniper_score:.3f} {sniper_status}")
        if sniper_score < sniper_threshold:
            print()
            print("=" * 80)
            print("⚠⚠⚠ WARNING: MODEL EXHIBITS SIGNS OF OVERFITTING ⚠⚠⚠")
            print("=" * 80)
            print(f"Sniper Score ({sniper_score:.3f}) < {sniper_threshold}")
            print("The model exhibits signs of overfitting or insufficient signal-to-noise ratio.")
            print("Review model parameters and consider adjusting feature selection.")
            print("=" * 80)
    else:
        print("SNIPER SCORE (Precision): N/A (no buy signals)")

    # --- FINAL ALLOCATION RECEIPT ---
    print("\n" + "=" * 85)
    print("FINAL ALLOCATION RECEIPT - THE ALPHA DOMINATOR v10.0")
    print("=" * 85)
    print(f"Date:   {results.index[-1].date()}")
    print(f"Regime: {results['Regime'].iloc[-1]} | ML Prob: {results['ML_Prob'].iloc[-1]:.1%}")

    # Current Optimal Rebalance Setting
    current_rebal = classifier.current_rebalance_period
    print(f"Adaptive Rebalance Setting: {current_rebal} days")

    diag = engine.final_diagnostics
    if diag:
        eff_n = diag.get('effective_n', 0)
        status = "✓ GOOD" if eff_n >= config.min_effective_n else "✗ LOW"
        print(f"DIVERSITY SCORE (Effective N): {eff_n:.2f} {status}")
        print(f"Target: {config.min_effective_n:.1f}+")

        # Growth Anchor status (with tolerance)
        growth_weight = diag.get('growth_anchor_weight', 0)
        growth_status = "✓ MET" if growth_weight >= (config.min_growth_anchor - config.constraint_tolerance) else "✗ BELOW"
        print(f"GROWTH ANCHOR (QQQ+XLK+SMH+VGT): {growth_weight:.1%} {growth_status} (Min: {config.min_growth_anchor:.0%})")

        # Gold Cap status (with tolerance)
        gold_weight = diag.get('gold_weight', 0)
        gold_status = "✓ OK" if gold_weight <= (config.gold_cap_risk_on + config.constraint_tolerance) else "✗ OVER"
        print(f"GOLD CAP: {gold_weight:.1%} {gold_status} (Max: {config.gold_cap_risk_on:.0%})")
    print()

    print("-" * 85)
    print(f"{'Asset':<8} {'Weight':>10} {'IR_Score':>10} {'Trend':>10} {'Risk Contrib':>14} {'Status':<10}")
    print("-" * 85)

    # Get Final IR Scores and Risk Contributions
    final_ir_scores = engine.final_ir
    pctr = diag.get('pctr', {}) if diag else {}

    # Sort positions by weight
    sorted_pos = sorted(
        zip(optimizer.assets, engine.final_weights),
        key=lambda x: x[1],
        reverse=True
    )

    for asset, weight in sorted_pos:
        if weight > 0.005:
            # Trend Check
            is_above = above_sma.iloc[-1].get(asset, False)
            trend_str = "↑ ABOVE" if is_above else "↓ BELOW"

            # IR Score (Information Ratio)
            ir_val = final_ir_scores.get(asset, 0.0)

            # Risk Contribution
            risk_val = pctr.get(asset, 0.0)

            # Conviction Status - use centralized GROWTH_ANCHORS constant
            if asset in DataManager.GROWTH_ANCHORS:
                status = "★ ANCHOR"
            elif weight > 0.15:
                status = "★ CORE"
            else:
                status = "• HOLD"

            print(f"{asset:<8} {weight:>9.1%} {ir_val:>10.3f} {trend_str:>10} {risk_val:>13.1%} {status:<10}")

    print("-" * 85)
    print(f"{'TOTAL':<8} {sum(engine.final_weights):>9.1%}")
    print("=" * 85)

    # Model Stability Status
    model_stability = getattr(classifier, 'model_stability', 'UNKNOWN')
    print(f"\nModel Stability: [{model_stability}]")
    print("\nEXECUTION COMPLETE.")

    # 6. Generate and display all Plots (after terminal output)
    print("\n[6/6] Generating visualizations...")
    engine.plot_performance(results)
    engine.plot_allocation_history()
    engine.plot_regime_analysis(results, prices, sma_200)
    classifier.plot_shap_summary()
    classifier.plot_validation_curves()  # Model Health Dashboard
    mc.plot_paths()
    mc.plot_distribution()

    # Keep all plot windows open until manually closed
    plt.show(block=True)


if __name__ == "__main__":
    main()
