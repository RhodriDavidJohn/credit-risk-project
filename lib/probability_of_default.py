import numpy as np
import pandas as pd
from scipy.stats import chi2, binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import (roc_auc_score,
                             roc_curve,
                             average_precision_score,
                             precision_recall_curve,
                             auc,
                             brier_score_loss,
                             make_scorer)


def fit_logistic_regression(X_fit, y_fit, params=None, random_state=42, calibrate=False):
    lr = LogisticRegression(**(params or {}), random_state=random_state)
    lr.fit(X_fit, y_fit)
    return lr


def hosmer_lemeshow_test(y_true, y_pred, rating_class = None, n_groups = 10):
    y_true = pd.DataFrame({"y_true": y_true})
    y_pred = pd.DataFrame({"y_pred": y_pred})
    # Create decile groups based on the predicted probabilities
    if rating_class is None:
        y_true["decile"] = pd.qcut(y_pred["y_pred"], q=n_groups, labels=False) + 1
        y_pred["decile"] = pd.qcut(y_pred["y_pred"], q=n_groups, labels=False) + 1
    else:
        y_true["decile"] = rating_class
        y_pred["decile"] = rating_class
    
    # Compute the observed and expected number of events for each decile group
    obsevents_1 = y_true.groupby("decile").agg({"y_true": "sum"})["y_true"]
    obsevents_0 = (y_true.groupby("decile").agg({"y_true": "count"}) - obsevents_1)["y_true"]
    expevents_1 = y_pred.groupby("decile").agg({"y_pred": "sum"})["y_pred"]
    expevents_0 = (y_pred.groupby("decile").agg({"y_pred": "count"}) - expevents_1)["y_pred"]
    hosmer = (((obsevents_1 - expevents_1) ** 2 / expevents_1).sum()
              + ((obsevents_0 - expevents_0) ** 2 / expevents_0).sum())

    # Degrees of freedom: Number of groups - 2
    degrees_of_freedom = n_groups - 2

    # Calculate the p-value for the HL test
    p_value = 1 - chi2.cdf(hosmer, degrees_of_freedom)

    return hosmer, degrees_of_freedom, p_value


def binomial_calibration_test(y_true, y_prob, n_bins=10, strategy='uniform'):
    """Performs a Binomial Goodness-of-Fit test per bin for model calibration."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    
    # Define bin edges using uniform intervals or quantiles
    if strategy == 'uniform':
        edges = np.linspace(0, 1, n_bins + 1)
    elif strategy == 'quantile':
        edges = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
    else:
        raise ValueError("strategy must be 'uniform' or 'quantile'")
        
    # Assign probabilities to bins and clamp edge cases
    bin_ids = np.clip(np.digitize(y_prob, edges) - 1, 0, n_bins - 1)
    
    # Calculate metrics per bin and run exact binomial tests
    results = []
    for i in range(n_bins):
        mask = (bin_ids == i)
        n = np.sum(mask)
        if n == 0:
            continue  # Skip empty bins
        
        k = np.sum(y_true[mask])         # Observed positive outcomes
        p_pred = np.mean(y_prob[mask])   # Expected probability (model mean)
        
        # Avoid mathematical bounds boundary issues at absolute 0 or 1
        p_clipped = np.clip(p_pred, 1e-6, 1 - 1e-6)
        p_val = binomtest(k, n, p=p_clipped, alternative='two-sided').pvalue
        
        results.append({
            'bin': i, 'total_n': n, 'observed_pos': k, 
            'expected_prob': p_pred, 'observed_prob': k / n, 'p_value': p_val
        })
        
    return pd.DataFrame(results)



class ProbabilityOfDefault():

    def __init__(self, model, X_train, X_test, y_train, y_test):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def fit(self, cv_fold=5):
        self.calibrated_model = CalibratedClassifierCV(
            estimator=self.model,
            method='isotonic', # sigmoid if sample size is too small
            cv=cv_fold
        )
        self.calibrated_model.fit(self.X_train, self.y_train)

    def cross_validation_metrics(self, cv_folds=5):
        gini_scorer = make_scorer(
            lambda y_t, y_p: 2 * roc_auc_score(y_t, y_p) - 1, 
            response_method='predict_proba'
        )
        
        def pr_auc_calc(y_true, y_probs):
            precision, recall, _ = precision_recall_curve(y_true, y_probs)
            return auc(recall, precision)
            
        pr_auc_scorer = make_scorer(pr_auc_calc, response_method='predict_proba')
        
        scoring_dict = {
            'gini': gini_scorer,
            'brier_loss': 'neg_brier_score',
            'roc_auc': 'roc_auc',
            'pr_auc': pr_auc_scorer,
            'average_precision': 'average_precision'
        }

        cv_results = cross_validate(
            self.model, self.X_train, self.y_train, 
            cv=cv_folds, scoring=scoring_dict, return_train_score=False
        )
        
        return pd.DataFrame({
            'Fold': np.arange(1, cv_folds + 1),
            'Gini': cv_results['test_gini'],
            'ROC_AUC': cv_results['test_roc_auc'],
            'PR_AUC': cv_results['test_pr_auc'],
            'Precision': cv_results['test_average_precision'],
            'Brier_Score': -cv_results['test_brier_loss']
        }).set_index('Fold')

    def run_cv_calibration_tests(self, cv_folds=5, n_groups=10):
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        oof_probs = np.zeros(len(self.y_train))
        
        for train_idx, val_idx in cv.split(self.X_train, self.y_train):
            X_tr_fold, y_tr_fold = self.X_train.iloc[train_idx], self.y_train.iloc[train_idx]
            X_val_fold = self.X_train.iloc[val_idx]
            
            calibrated_model = CalibratedClassifierCV(
                estimator=self.model, 
                method='isotonic', 
                cv=3
            )
            
            calibrated_model.fit(X_tr_fold, y_tr_fold)
            oof_probs[val_idx] = calibrated_model.predict_proba(X_val_fold)[:, 1]
        
        hl_stat, hl_df, hl_pvalue = hosmer_lemeshow_test(self.y_train, oof_probs, n_groups=n_groups)
        binomial_df = binomial_calibration_test(self.y_train, oof_probs, n_bins=n_groups, strategy='quantile')
        
        return {
            "hosmer_lemeshow": {"statistic": hl_stat, "df": hl_df, "p_value": hl_pvalue},
            "binomial_test_by_bin": binomial_df
        }

    def predict_pd(self):
        return self.calibrated_model.predict_proba(self.X_test)[:, 1]

    def predict_default(self):
        return self.calibrated_model.predict(self.X_test)

    def roc_auc(self):
        self.fit()
        return roc_auc_score(self.y_test, self.predict_pd())

    def pr_auc(self):
        self.fit()
        precision, recall, _ = precision_recall_curve(self.y_test, self.predict_pd())
        return auc(recall, precision)

    def average_precision(self):
        self.fit()
        return average_precision_score(self.y_test, self.predict_pd())

    def gini_coefficient(self):
        self.fit()
        return 2 * self.roc_auc() - 1

    def ks_statistic(self, return_threshold=False):
        self.fit()
        fpr, tpr, threshold = roc_curve(self.y_test, self.predict_pd())
        ks_stat = max(tpr - fpr)
        if not return_threshold:
            return ks_stat
        else:
            best_threshold = threshold[np.where((tpr - fpr)==ks_stat)]
            return ks_stat, best_threshold

    def brier_score(self):
        self.fit()
        return brier_score_loss(self.y_test, self.predict_pd())

    def hosmer_lemeshow_test(self, rating_class = None, n_groups = 10):
        self.fit()
        y_true = pd.DataFrame({"y_true": self.y_test})
        y_pred = pd.DataFrame({"y_pred": self.predict_pd()})
        # Create decile groups based on the predicted probabilities
        if rating_class is None:
            y_true["decile"] = pd.qcut(y_pred["y_pred"], q=n_groups, labels=False) + 1
            y_pred["decile"] = pd.qcut(y_pred["y_pred"], q=n_groups, labels=False) + 1
        else:
            y_true["decile"] = rating_class
            y_pred["decile"] = rating_class
        
        # Compute the observed and expected number of events for each decile group
        obsevents_1 = y_true.groupby("decile").agg({"y_true": "sum"})["y_true"]
        obsevents_0 = (y_true.groupby("decile").agg({"y_true": "count"}) - obsevents_1)["y_true"]
        expevents_1 = y_pred.groupby("decile").agg({"y_pred": "sum"})["y_pred"]
        expevents_0 = (y_pred.groupby("decile").agg({"y_pred": "count"}) - expevents_1)["y_pred"]
        hosmer = (((obsevents_1 - expevents_1) ** 2 / expevents_1).sum()
                    + ((obsevents_0 - expevents_0) ** 2 / expevents_0).sum())
    
        # Degrees of freedom: Number of groups - 2
        degrees_of_freedom = n_groups - 2
    
        # Calculate the p-value for the HL test
        p_value = 1 - chi2.cdf(hosmer, degrees_of_freedom)
    
        return hosmer, degrees_of_freedom, p_value

    def binomial_goodness_of_fit_test(self, n_bins=10, strategy='uniform'):
        self.fit()
        y_true = np.asarray(self.y_test)
        y_prob = np.asarray(self.predict_pd())
        
        # Define bin edges using uniform intervals or quantiles
        if strategy == 'uniform':
            edges = np.linspace(0, 1, n_bins + 1)
        elif strategy == 'quantile':
            edges = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
        else:
            raise ValueError("strategy must be 'uniform' or 'quantile'")
            
        # Assign probabilities to bins and clamp edge cases
        bin_ids = np.clip(np.digitize(y_prob, edges) - 1, 0, n_bins - 1)
        
        # Calculate metrics per bin and run exact binomial tests
        results = []
        for i in range(n_bins):
            mask = (bin_ids == i)
            n = np.sum(mask)
            if n == 0:
                continue  # Skip empty bins
            
            k = np.sum(y_true[mask])         # Observed positive outcomes
            p_pred = np.mean(y_prob[mask])   # Expected probability (model mean)
            
            # Avoid mathematical bounds boundary issues at absolute 0 or 1
            p_clipped = np.clip(p_pred, 1e-6, 1 - 1e-6)
            p_val = binomtest(k, n, p=p_clipped, alternative='two-sided').pvalue
            
            results.append({
                'bin': i, 'total_n': n, 'observed_pos': k, 
                'expected_prob': p_pred, 'observed_prob': k / n, 'p_value': p_val
            })
            
        return pd.DataFrame(results)

    def generate_ranking_stability_report(self):
        self.fit()
        self.model.fit(self.X_train, self.y_train)
            
        report_data = []
        
        datasets = {
            'Train': (self.X_train, self.y_train),
            'Test': (self.X_test, self.y_test)
        }
        
        for split_name, (X_data, y_data) in datasets.items():
            # ---- Base Model Metrics ----
            base_probs = self.model.predict_proba(X_data)[:, 1]
            
            base_auc = roc_auc_score(y_data, base_probs)
            base_gini = 2 * base_auc - 1
            
            # Calculate Base KS
            fpr, tpr, thresholds = roc_curve(y_data, base_probs)
            ks_idx = np.argmax(tpr - fpr)
            base_ks = tpr[ks_idx] - fpr[ks_idx]
            base_ks_threshold = thresholds[ks_idx]
            
            # ---- Calibrated Model Metrics ----
            calib_probs = self.calibrated_model.predict_proba(X_data)[:, 1]
            
            calib_auc = roc_auc_score(y_data, calib_probs)
            calib_gini = 2 * calib_auc - 1
            
            # Calculate Calibrated KS
            fpr_c, tpr_c, thresholds_c = roc_curve(y_data, calib_probs)
            ks_idx_c = np.argmax(tpr_c - fpr_c)
            calib_ks = tpr_c[ks_idx_c] - fpr_c[ks_idx_c]
            calib_ks_threshold = thresholds_c[ks_idx_c]
            
            # Append Results
            report_data.append({
                'Dataset': split_name,
                'Model Version': 'Base Model',
                'Gini': base_gini,
                'KS Statistic': base_ks,
                'KS Threshold (PD)': base_ks_threshold
            })
            report_data.append({
                'Dataset': split_name,
                'Model Version': 'Isotonic Calibrated',
                'Gini': calib_gini,
                'KS Statistic': calib_ks,
                'KS Threshold (PD)': calib_ks_threshold
            })
            
        # Structure into a multi-indexed DataFrame for easy scannability
        df_report = pd.DataFrame(report_data)
        df_report.set_index(['Dataset', 'Model Version'], inplace=True)
        return df_report

    def convert_pd_to_credit_score(self, pd_array, target_score=600, target_odds=50, pdo=20):
        """
        Transforms raw calibrated probabilities into credit scoring points.
        Higher scores mean lower risk (lower probability of default).
        """
        pd_array = np.clip(pd_array, 1e-6, 1 - 1e-6) # Guard against 0 or 1 boundaries
        factor = pdo / np.log(2)
        offset = target_score - (factor * np.log(target_odds))
        
        # Calculate Log-Odds of defaulting
        log_odds = np.log(pd_array / (1 - pd_array))
        
        # Subtract from offset so that higher scores represent LOWER risk
        scores = offset - (factor * log_odds)
        return np.round(scores).astype(int)

    def generate_psi_report(self, n_bins=10):
        """
        Computes the Population Stability Index (PSI) on the calibrated 
        probabilities between the Train (Expected) and Test (Actual) populations.
        """
        self.fit()
            
        train_probs = self.calibrated_model.predict_proba(self.X_train)[:, 1]
        test_probs = self.calibrated_model.predict_proba(self.X_test)[:, 1]
        
        # 1. Define bucket edges based on training quantiles
        edges = np.percentile(train_probs, np.linspace(0, 100, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0 # Secure structural bounds
        
        # 2. Count distributions per bucket
        train_counts, _ = np.histogram(train_probs, bins=edges)
        test_counts, _ = np.histogram(test_probs, bins=edges)
        
        # 3. Convert to percentages
        train_pct = train_counts / len(train_probs)
        test_pct = test_counts / len(test_probs)
        
        # Handle zero-count protections mathematically (Laplace smoothing)
        train_pct = np.where(train_pct == 0, 1e-4, train_pct)
        test_pct = np.where(test_pct == 0, 1e-4, test_pct)
        
        # 4. Calculate localized PSI per bucket step
        psi_values = (test_pct - train_pct) * np.log(test_pct / train_pct)
        total_psi = np.sum(psi_values)
        
        # Build Summary Frame
        psi_df = pd.DataFrame({
            'Lower Bound PD': edges[:-1],
            'Upper Bound PD': edges[1:],
            'Train (Expected %)': train_pct * 100,
            'Test (Actual %)': test_pct * 100,
            'PSI Contribution': psi_values
        })
        
        # Map scoring tiers to bounds for context
        psi_df['Min Credit Score'] = self.convert_pd_to_credit_score(psi_df['Upper Bound PD'])
        psi_df['Max Credit Score'] = self.convert_pd_to_credit_score(psi_df['Lower Bound PD'])
        
        print(f"\n=== System Population Stability Index Assessment ===")
        print(f"Total Portfolio PSI: {total_psi:.4f}")
        if total_psi < 0.10:
            print("Status: STABLE POPULATION (No significant shift detected)")
        elif total_psi < 0.25:
            print("Status: WARNING (Moderate shift detected, track portfolio attributes)")
        else:
            print("Status: ACTION REQUIRED (Severe model drift detected, model needs recalculation)")
            
        return psi_df

