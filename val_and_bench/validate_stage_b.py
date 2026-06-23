import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import r2_score
from scipy.spatial.distance import jensenshannon

# Ensure we can import satlinksim
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from satlinksim.satellite_link_sim import simulate_station
from satlinksim.ground_stations import GROUND_STATIONS
from satlinksim.domain.link.itu_models import itu_rain_coefficients, itu_rain_height, effective_path_length, gaseous_absorption_db
from satlinksim.domain.link.budget import fspl_db, noise_power_dbw

# Re-use extract_features_and_targets from evaluate_stage_b
from evaluate_stage_b import extract_features_and_targets

def run_validation_suite():
    print("======================================================================")
    print("STAGE B ROBUSTNESS AND GENERALIZATION VALIDATION SUITE")
    print("======================================================================")
    
    # Retrieve Ground Stations
    gs_delhi = [s for s in GROUND_STATIONS if s["name"] == "Delhi"][0].copy()
    gs_saopaulo = [s for s in GROUND_STATIONS if s["name"] == "Sao Paulo"][0].copy()
    gs_tokyo = [s for s in GROUND_STATIONS if s["name"] == "Tokyo"][0].copy()
    gs_berlin = [s for s in GROUND_STATIONS if s["name"] == "Berlin"][0].copy()
    
    gs_delhi["norad_id"] = 26766
    gs_saopaulo["norad_id"] = 33153
    gs_tokyo["norad_id"] = 26900
    gs_berlin["norad_id"] = 27380
    
    stations = [gs_delhi, gs_saopaulo, gs_tokyo, gs_berlin]
    months = [7, 1, 9, 11]
    
    n_steps = 7200
    freq_hz = 14e9
    bandwidth_hz = 36e6
    polarization = "vertical"
    
    # Pre-generate standard training data (14 GHz, Seed 100)
    print("\n[Data Preparation] Generating base training data (14 GHz, Seed 100)...")
    X_train_list, y_train_list = [], []
    train_meta_list = []
    for i, gs in enumerate(stations):
        for force_rain in [True, False]:
            start_time = datetime(2026, months[i], 15, 12, 0, 0, tzinfo=timezone.utc)
            res = simulate_station(
                gs, n_steps=n_steps, seed=100, freq_hz=freq_hz,
                bandwidth_hz=bandwidth_hz, polarization=polarization,
                force_rain=force_rain, start_time=start_time
            )
            X_df, y_df, meta = extract_features_and_targets(res, gs, freq_hz, bandwidth_hz, polarization, start_time)
            X_train_list.append(X_df)
            y_train_list.append(y_df)
            train_meta_list.append(pd.DataFrame(meta))
            
    X_train = pd.concat(X_train_list, ignore_index=True)
    y_train = pd.concat(y_train_list, ignore_index=True)["true_rain_rate"].values
    train_meta = pd.concat(train_meta_list, ignore_index=True)
    
    # Pre-generate standard testing data (14 GHz, Seed 200)
    print("[Data Preparation] Generating base testing data (14 GHz, Seed 200)...")
    X_test_list, y_test_list = [], []
    test_meta_list = []
    for i, gs in enumerate(stations):
        for force_rain in [True, False]:
            start_time = datetime(2026, months[i], 15, 12, 0, 0, tzinfo=timezone.utc)
            res = simulate_station(
                gs, n_steps=n_steps, seed=200, freq_hz=freq_hz,
                bandwidth_hz=bandwidth_hz, polarization=polarization,
                force_rain=force_rain, start_time=start_time
            )
            X_df, y_df, meta = extract_features_and_targets(res, gs, freq_hz, bandwidth_hz, polarization, start_time)
            X_test_list.append(X_df)
            y_test_list.append(y_df)
            test_meta_list.append(pd.DataFrame(meta))
            
    X_test = pd.concat(X_test_list, ignore_index=True)
    y_test = pd.concat(y_test_list, ignore_index=True)["true_rain_rate"].values
    test_meta = pd.concat(test_meta_list, ignore_index=True)
    
    # Helper to train and evaluate cascade model
    def evaluate_cascade(X_tr, y_tr, X_te, y_te):
        y_tr_c = (y_tr > 0.1).astype(int)
        y_te_c = (y_te > 0.1).astype(int)
        
        clf = XGBClassifier(objective="binary:logistic", n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42, n_jobs=-1)
        clf.fit(X_tr, y_tr_c)
        
        reg = XGBRegressor(objective="reg:squarederror", n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42, n_jobs=-1)
        reg.fit(X_tr, y_tr)
        
        pred_c = clf.predict(X_te)
        pred_raw_r = reg.predict(X_te)
        pred_r = np.where(pred_c == 1, np.maximum(pred_raw_r, 0.0), 0.0)
        
        rmse = np.sqrt(np.mean((y_te - pred_r)**2))
        mae = np.mean(np.abs(y_te - pred_r))
        corr = np.corrcoef(y_te, pred_r)[0, 1] if np.std(pred_r) > 0 else 0.0
        r2 = r2_score(y_te, pred_r)
        
        # Classification F1 at 0.1 mm/h
        tp = np.sum((y_te > 0.1) & (pred_r > 0.1))
        fp = np.sum((y_te <= 0.1) & (pred_r > 0.1))
        fn = np.sum((y_te > 0.1) & (pred_r <= 0.1))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        
        return rmse, mae, corr, r2, prec, rec, f1, clf, reg, pred_r

    artifact_dir = "/home/satyansh/.gemini/antigravity-cli/brain/b30d89ad-2cb9-4e00-a92c-0ae53cdb775b"
    os.makedirs(artifact_dir, exist_ok=True)
    
    validation_report_sections = []
    
    # ------------------------------------------------------------------------
    # 1. BASELINE TRAINING & FEATURE IMPORTANCE
    # ------------------------------------------------------------------------
    print("\n--- 1. Feature Importance Analysis ---")
    rmse, mae, corr, r2, prec, rec, f1, base_clf, base_reg, base_preds = evaluate_cascade(X_train, y_train, X_test, y_test)
    
    importances = base_reg.feature_importances_
    feat_imp_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
    
    print(feat_imp_df.head(10).to_string(index=False))
    
    validation_report_sections.append(f"""### 1. Feature Importance Analysis

Below are the top 10 most important features for the XGBoost Regressor model:

| Feature | Importance |
|---|---|
{chr(10).join([f"| {row['Feature']} | {row['Importance']:.4f} |" for _, row in feat_imp_df.head(10).iterrows()])}
""")

    # ------------------------------------------------------------------------
    # 2. FEATURE LEAKAGE & ABLATION STUDY
    # ------------------------------------------------------------------------
    print("\n--- 2. Feature Leakage & Ablation Study ---")
    ablation_results = []
    
    # Define feature groups
    feature_groups = {
        "All Features": list(X_train.columns),
        "No Rolling Stats": [c for c in X_train.columns if not c.startswith("rolling_")],
        "No Excess Attn & L_eff": [c for c in X_train.columns if c != "excess_attn" and c != "L_eff" and not c.startswith("rolling_") and not c.startswith("lag_")],
        "No Climatology": [c for c in X_train.columns if not c.startswith("gs_") and not c.startswith("itu_")]
    }
    
    for name, cols in feature_groups.items():
        r_rmse, r_mae, r_corr, r_r2, r_prec, r_rec, r_f1, _, _, _ = evaluate_cascade(
            X_train[cols], y_train, X_test[cols], y_test
        )
        print(f"  {name:25s} | RMSE: {r_rmse:.4f} | R²: {r_r2:.4f} | Corr: {r_corr:.4f} | F1: {r_f1:.4f}")
        ablation_results.append((name, r_rmse, r_mae, r_corr, r_r2, r_f1))
        
    validation_report_sections.append(f"""### 2. Feature Leakage & Ablation Study

Evaluating the model after stripping groups of features to ensure it is not overly reliant on raw/direct simulator attenuation mapping:

| Ablation Group | RMSE (mm/h) | MAE (mm/h) | Correlation | R² Score | F1 (0.1 mm/h) |
|---|---|---|---|---|---|
{chr(10).join([f"| {name} | {rmse:.4f} | {mae:.4f} | {corr:.4f} | {r2:.4f} | {f1:.4f} |" for name, rmse, mae, corr, r2, f1 in ablation_results])}
""")

    # ------------------------------------------------------------------------
    # 3. LEAVE-ONE-STATION-OUT (LOSO) VALIDATION
    # ------------------------------------------------------------------------
    print("\n--- 3. Leave-One-Station-Out (LOSO) Generalization ---")
    loso_results = []
    
    for i, test_station in enumerate(stations):
        name = test_station["name"]
        
        # Train on other 3 stations (both train and test splits to get enough samples)
        train_idx = (train_meta["station"] != name)
        test_idx = (test_meta["station"] != name)
        
        X_tr_loso = pd.concat([X_train[train_idx], X_test[test_idx]], ignore_index=True)
        y_tr_loso = np.concatenate([y_train[train_idx.values], y_test[test_idx.values]])
        
        # Test ONLY on the excluded station (test split only)
        val_idx = (test_meta["station"] == name) & (test_meta["force_rain"] == False)
        X_te_loso = X_test[val_idx]
        y_te_loso = y_test[val_idx.values]
        
        l_rmse, l_mae, l_corr, l_r2, l_prec, l_rec, l_f1, _, _, _ = evaluate_cascade(
            X_tr_loso, y_tr_loso, X_te_loso, y_te_loso
        )
        print(f"  Excluded: {name:12s} | RMSE: {l_rmse:.4f} | R²: {l_r2:.4f} | Corr: {l_corr:.4f} | F1: {l_f1:.4f}")
        loso_results.append((name, l_rmse, l_mae, l_corr, l_r2, l_f1))
        
    validation_report_sections.append(f"""### 3. Leave-One-Station-Out (LOSO) Generalization

Evaluating how well the narrowcaster generalizes to a completely unseen geographic location (cross-climate validation):

| Excluded Ground Station | RMSE (mm/h) | MAE (mm/h) | Correlation | R² Score | F1 Score (0.1 mm/h) |
|---|---|---|---|---|---|
{chr(10).join([f"| {name} | {rmse:.4f} | {mae:.4f} | {corr:.4f} | {r2:.4f} | {f1:.4f} |" for name, rmse, mae, corr, r2, f1 in loso_results])}
""")

    # ------------------------------------------------------------------------
    # 4. FREQUENCY GENERALIZATION
    # ------------------------------------------------------------------------
    print("\n--- 4. Cross-Frequency Generalization ---")
    # Base model is trained on 14 GHz
    cross_freq_results = []
    
    for test_freq in [12e9, 20e9, 30e9]:
        X_tf_list, y_tf_list = [], []
        test_meta_f = []
        for i, gs in enumerate(stations):
            for force_rain in [True, False]:
                start_time = datetime(2026, months[i], 15, 12, 0, 0, tzinfo=timezone.utc)
                res = simulate_station(
                    gs, n_steps=n_steps, seed=300, freq_hz=test_freq,
                    bandwidth_hz=bandwidth_hz, polarization=polarization,
                    force_rain=force_rain, start_time=start_time
                )
                X_df, y_df, meta = extract_features_and_targets(res, gs, test_freq, bandwidth_hz, polarization, start_time)
                X_tf_list.append(X_df)
                y_tf_list.append(y_df)
                test_meta_f.append(pd.DataFrame(meta))
                
        X_tf = pd.concat(X_tf_list, ignore_index=True)
        y_tf = pd.concat(y_tf_list, ignore_index=True)["true_rain_rate"].values
        
        # Run inference using the pre-trained 14 GHz model
        pred_c = base_clf.predict(X_tf)
        pred_raw_r = base_reg.predict(X_tf)
        pred_r = np.where(pred_c == 1, np.maximum(pred_raw_r, 0.0), 0.0)
        
        f_rmse = np.sqrt(np.mean((y_tf - pred_r)**2))
        f_mae = np.mean(np.abs(y_tf - pred_r))
        f_corr = np.corrcoef(y_tf, pred_r)[0, 1] if np.std(pred_r) > 0 else 0.0
        f_r2 = r2_score(y_tf, pred_r)
        
        tp = np.sum((y_tf > 0.1) & (pred_r > 0.1))
        fp = np.sum((y_tf <= 0.1) & (pred_r > 0.1))
        fn = np.sum((y_tf > 0.1) & (pred_r <= 0.1))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f_f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        
        print(f"  Test Freq: {test_freq/1e9:.0f} GHz | RMSE: {f_rmse:.4f} | R²: {f_r2:.4f} | Corr: {f_corr:.4f} | F1: {f_f1:.4f}")
        cross_freq_results.append((f"{test_freq/1e9:.0f} GHz", f_rmse, f_mae, f_corr, f_r2, f_f1))
        
    validation_report_sections.append(f"""### 4. Cross-Frequency Generalization

Testing the model pre-trained at 14 GHz on unseen high-frequency channels (12 GHz, 20 GHz, 30 GHz) without retraining:

| Test Channel Frequency | RMSE (mm/h) | MAE (mm/h) | Correlation | R² Score | F1 Score (0.1 mm/h) |
|---|---|---|---|---|---|
{chr(10).join([f"| {freq} | {rmse:.4f} | {mae:.4f} | {corr:.4f} | {r2:.4f} | {f1:.4f} |" for freq, rmse, mae, corr, r2, f1 in cross_freq_results])}
""")

    # ------------------------------------------------------------------------
    # 5. DISTRIBUTION MATCHING & JENSEN-SHANNON DIVERGENCE
    # ------------------------------------------------------------------------
    print("\n--- 5. Distribution Matching ---")
    
    # Calculate JS Divergence for Stochastic Rain Sao Paulo
    idx_sp = (test_meta["station"] == "Sao Paulo") & (test_meta["force_rain"] == False)
    y_true_sp = y_test[idx_sp]
    y_pred_sp = base_preds[idx_sp]
    
    # Compute histogram probability distributions
    bins = np.linspace(0.0, 10.0, 100)
    p_true, _ = np.histogram(y_true_sp, bins=bins, density=True)
    p_pred, _ = np.histogram(y_pred_sp, bins=bins, density=True)
    
    # Smooth to avoid zero divisions
    p_true = p_true + 1e-12
    p_pred = p_pred + 1e-12
    p_true /= np.sum(p_true)
    p_pred /= np.sum(p_pred)
    
    js_div = jensenshannon(p_true, p_pred)
    print(f"  Sao Paulo Stochastic Rain JS Divergence: {js_div:.4f}")
    
    validation_report_sections.append(f"""### 5. Distribution Matching

To verify that the model produces physically realistic distributions rather than simply smoothing outliers:

* **Jensen-Shannon (JS) Divergence** ($P(R) \\parallel P(\\widehat{{R}})$) for Sao Paulo Stochastic Rain: **{js_div:.5f}** *(where 0.0 is a perfect match)*.
""")

    # ------------------------------------------------------------------------
    # 6. SIMULATOR PARAMETER MODIFICATION TESTING
    # ------------------------------------------------------------------------
    print("\n--- 6. Simulator Parameter Modification (Robustness under noise shifts) ---")
    
    # Generate test dataset with modified coherence time (tau_c = 600s) and 2x Scintillation
    X_mod_list, y_mod_list = [], []
    for i, gs in enumerate(stations):
        for force_rain in [True, False]:
            start_time = datetime(2026, months[i], 15, 12, 0, 0, tzinfo=timezone.utc)
            # Patch parameters in simulation
            # We run with modified seed and we can adjust the scale or scintillation power during feature extraction
            # To simulate 2.0x scintillation, we double the standard deviation of scintillation noise
            from satlinksim.domain.rain.engine import CorrelatedRainProcess
            res = simulate_station(
                gs, n_steps=n_steps, seed=400, freq_hz=freq_hz,
                bandwidth_hz=bandwidth_hz, polarization=polarization,
                force_rain=force_rain, start_time=start_time,
                rain_model_factory=lambda g: CorrelatedRainProcess([g], dt_s=1.0, tau_c=600.0, force_rain=force_rain, rain_rate_scale=1.5)
            )
            # Extracted feature standard scintillation is doubled
            # Let's extract features
            X_df, y_df, _ = extract_features_and_targets(res, gs, freq_hz, bandwidth_hz, polarization, start_time)
            
            # Artificially inject additional scintillation noise into excess attenuation
            # to verify robustness under heavy noise
            from satlinksim.domain.link.itu_models import scintillation_sigma_db
            ss = scintillation_sigma_db(freq_hz/1e9, np.array(res.elevation_series), gs["antenna_diam_m"], gs["humidity_pct"])
            additional_noise = np.random.normal(0, ss) # Add another 1x scintillation to double variance
            X_df["excess_attn"] += additional_noise
            
            X_mod_list.append(X_df)
            y_mod_list.append(y_df)
            
    X_mod = pd.concat(X_mod_list, ignore_index=True)
    y_mod = pd.concat(y_mod_list, ignore_index=True)["true_rain_rate"].values
    
    m_rmse, m_mae, m_corr, m_r2, m_prec, m_rec, m_f1, _, _, _ = evaluate_cascade(
        X_train, y_train, X_mod, y_mod
    )
    print(f"  Modified Dataset | RMSE: {m_rmse:.4f} | R²: {m_r2:.4f} | Corr: {m_corr:.4f} | F1: {m_f1:.4f}")
    
    validation_report_sections.append(f"""### 6. Simulator Parameter Modification (Noise Shifts)

Testing robustness when evaluated against a simulator run with modified dynamics (Rain coherence time tau_c = 600s, rain scale 1.5x, and injected 2.0x nominal scintillation power):

* **RMSE**: {m_rmse:.4f} mm/h
* **Correlation ($R$)**: {m_corr:.4f}
* **R² Score**: {m_r2:.4f}
* **F1 Score**: {m_f1:.4f}
""")

    # Write report
    report_path = "/home/satyansh/leo_meo/docs/inverse_rain_rate.md"
    with open(report_path, "a") as f:
        f.write("\n## Stage B Generalization and Robustness Validation\n\n")
        f.write("Following the Stage B Validation checklist, this section documents the generalization, leakage, and cross-domain validation testing:\n\n")
        f.write("\n".join(validation_report_sections))
        
    print(f"\nCompleted Stage B Robustness Validation. Report updated in {report_path}")

if __name__ == "__main__":
    run_validation_suite()
