import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from scipy.signal import butter, filtfilt

# Ensure we can import satlinksim
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from satlinksim.satellite_link_sim import simulate_station
from satlinksim.ground_stations import GROUND_STATIONS
from satlinksim.domain.link.itu_models import itu_rain_coefficients, itu_rain_height, effective_path_length, gaseous_absorption_db
from satlinksim.domain.link.budget import fspl_db, noise_power_dbw

def lowpass_butterworth(data, cutoff_freq=0.005, fs=1.0, order=2):
    """
    Apply a zero-phase low-pass Butterworth filter to smooth out scintillation.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    if len(data) > 3 * max(len(b), len(a)):
        return filtfilt(b, a, data)
    return data

def run_evaluation():
    # Retrieve Ground Stations
    gs_delhi = [s for s in GROUND_STATIONS if s["name"] == "Delhi"][0].copy()
    gs_saopaulo = [s for s in GROUND_STATIONS if s["name"] == "Sao Paulo"][0].copy()
    
    # Assign NORAD IDs to ensure orbital propagation
    gs_delhi["norad_id"] = 26766
    gs_saopaulo["norad_id"] = 33153
    
    stations_to_test = [gs_delhi, gs_saopaulo]
    
    n_steps = 7200  # 2 hours of 1Hz data
    seed = 42
    freq_hz = 14e9  # 14 GHz
    freq_ghz = freq_hz / 1e9
    polarization = "vertical"
    bandwidth_hz = 36e6
    
    itu_k, itu_alpha = itu_rain_coefficients(freq_ghz, polarization)
    
    # Directory to save plot artifacts
    artifact_dir = "/home/satyansh/.gemini/antigravity-cli/brain/b30d89ad-2cb9-4e00-a92c-0ae53cdb775b"
    os.makedirs(artifact_dir, exist_ok=True)
    
    thresholds_to_test = [0.1, 0.5, 1.0, 2.0, 5.0]
    results_by_station = {}
    
    # We will focus the detailed threshold analysis on stochastic_rain
    for force_rain in [True, False]:
        rain_label = "active_rain" if force_rain else "stochastic_rain"
        print(f"\n--- Running Evaluation: {rain_label} ---")
        
        for gs in stations_to_test:
            print(f"\nGround Station: {gs['name']}")
            
            # 1. Run simulation
            res = simulate_station(
                gs,
                n_steps=n_steps,
                seed=seed,
                freq_hz=freq_hz,
                bandwidth_hz=bandwidth_hz,
                polarization=polarization,
                force_rain=force_rain
            )
            
            # Extract lists/arrays
            snr_series = np.array(res.snr_series)
            true_rain = np.array(res.rain_series)
            el_series = np.array(res.elevation_series)
            slant_series = np.array(res.slant_range_series)
            
            # Calculate physical constants
            eirp = gs["eirp_dbw"]
            g_rx = gs["g_rx_dbi"]
            noise_floor = noise_power_dbw(gs["system_temp_k"], bandwidth_hz)
            rain_h = itu_rain_height(gs["latitude"])
            
            # 2. Subtract known physical components
            pl = fspl_db(freq_hz, slant_series)
            gas_loss = gaseous_absorption_db(freq_ghz, el_series, gs["wv_g_m3"])
            
            # Excess attenuation: RA + Scint
            total_gain = eirp + g_rx - noise_floor
            excess_attn = total_gain - snr_series - pl - gas_loss
            
            # 3. Apply Low-pass filtering to remove Scintillation
            filtered_attn = lowpass_butterworth(excess_attn, cutoff_freq=0.005, fs=1.0, order=2)
            filtered_attn = np.maximum(filtered_attn, 0.0)
            
            # 4. Invert kR^α formula
            ep = effective_path_length(el_series, rain_h, gs["altitude_km"], itu_k)
            ep_safe = np.maximum(ep, 1e-6)
            pred_rain = (filtered_attn / (itu_k * ep_safe)) ** (1.0 / itu_alpha)
            
            # 5. Threshold Analysis
            threshold_metrics = []
            for thresh in thresholds_to_test:
                true_rain_binary = true_rain > thresh
                pred_rain_binary = pred_rain > thresh
                
                tp = np.sum(true_rain_binary & pred_rain_binary)
                fp = np.sum((~true_rain_binary) & pred_rain_binary)
                fn = np.sum(true_rain_binary & (~pred_rain_binary))
                tn = np.sum((~true_rain_binary) & (~pred_rain_binary))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                threshold_metrics.append({
                    "threshold": thresh,
                    "tp": int(tp),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tn": int(tn),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1
                })
                
                print(f"  Threshold: {thresh} mm/h")
                print(f"    TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
                print(f"    Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
                
            # For plotting PR curves we scan thresholds more finely
            fine_thresholds = np.linspace(0.01, 10.0, 200)
            precisions = []
            recalls = []
            f1s = []
            for thresh in fine_thresholds:
                true_b = true_rain > thresh
                pred_b = pred_rain > thresh
                
                tp = np.sum(true_b & pred_b)
                fp = np.sum((~true_b) & pred_b)
                fn = np.sum(true_b & (~pred_b))
                
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                
                precisions.append(prec)
                recalls.append(rec)
                f1s.append(f1)
                
            key = f"{gs['name']}_{rain_label}"
            results_by_station[key] = {
                "threshold_metrics": threshold_metrics,
                "fine_thresholds": fine_thresholds,
                "precisions": precisions,
                "recalls": recalls,
                "f1s": f1s,
                "true_rain": true_rain,
                "pred_rain": pred_rain,
                "snr_series": snr_series
            }
            
            # Plot 1: Inversion Visual
            plt.figure(figsize=(12, 6))
            time_axis = np.arange(n_steps) / 3600.0
            
            plt.subplot(2, 1, 1)
            plt.plot(time_axis, snr_series, label="Observed SNR (dB)", color="blue", alpha=0.5)
            plt.title(f"Stage A Analytical Inversion: {gs['name']} ({rain_label})")
            plt.ylabel("SNR (dB)")
            plt.grid(True)
            plt.legend()
            
            plt.subplot(2, 1, 2)
            plt.plot(time_axis, true_rain, label="True Rain Rate (mm/h)", color="green", linewidth=2)
            plt.plot(time_axis, pred_rain, label="Estimated Rain Rate (mm/h)", color="red", linestyle="--", alpha=0.8)
            plt.xlabel("Time (hours)")
            plt.ylabel("Rain Rate (mm/h)")
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            plot_path = os.path.join(artifact_dir, f"stage_a_{gs['name'].lower().replace(' ', '')}_{rain_label}.png")
            plt.savefig(plot_path)
            plt.close()
            
            # Plot 2: Histogram Comparison
            plt.figure(figsize=(8, 5))
            bins = np.linspace(0, max(max(true_rain), max(pred_rain), 5.0), 50)
            plt.hist(true_rain, bins=bins, alpha=0.6, label="True Rain Rate", color="green", edgecolor="black")
            plt.hist(pred_rain, bins=bins, alpha=0.6, label="Estimated Rain Rate", color="red", edgecolor="black")
            plt.title(f"Rain Rate Histogram Comparison: {gs['name']} ({rain_label})")
            plt.xlabel("Rain Rate (mm/h)")
            plt.ylabel("Frequency (Counts)")
            plt.yscale("log", nonpositive='clip')
            plt.grid(True, which="both", ls="--", alpha=0.5)
            plt.legend()
            plt.tight_layout()
            hist_path = os.path.join(artifact_dir, f"stage_a_{gs['name'].lower().replace(' ', '')}_{rain_label}_hist.png")
            plt.savefig(hist_path)
            plt.close()
            
            # Plot 3: PR Curve & Metrics vs Threshold (Only for stochastic_rain)
            if not force_rain:
                plt.figure(figsize=(12, 5))
                
                # Subplot 3a: Precision-Recall Curve
                plt.subplot(1, 2, 1)
                plt.plot(recalls, precisions, color="blue", linewidth=2, label="PR Curve")
                # Highlight the tested thresholds
                colors = ["red", "orange", "green", "purple", "brown"]
                for i, tm in enumerate(threshold_metrics):
                    plt.scatter(tm["recall"], tm["precision"], color=colors[i], s=80, zorder=5, 
                                label=f"Thresh {tm['threshold']} mm/h (F1={tm['f1']:.2f})")
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.title(f"Precision-Recall Curve: {gs['name']}")
                plt.grid(True, ls="--", alpha=0.5)
                plt.legend()
                
                # Subplot 3b: Metrics vs Threshold
                plt.subplot(1, 2, 2)
                plt.plot(fine_thresholds, precisions, label="Precision", color="blue", alpha=0.8)
                plt.plot(fine_thresholds, recalls, label="Recall", color="green", alpha=0.8)
                plt.plot(fine_thresholds, f1s, label="F1-Score", color="red", linewidth=2)
                plt.xlabel("Threshold (mm/h)")
                plt.ylabel("Value")
                plt.title(f"Classification Metrics vs Threshold")
                plt.grid(True, ls="--", alpha=0.5)
                plt.legend()
                
                plt.tight_layout()
                pr_path = os.path.join(artifact_dir, f"stage_a_{gs['name'].lower().replace(' ', '')}_pr_analysis.png")
                plt.savefig(pr_path)
                plt.close()
                print(f"  Saved PR analysis plot to {pr_path}")

    # Write inverse rain rate docs markdown file
    docs_path = "/home/satyansh/leo_meo/docs/inverse_rain_rate.md"
    os.makedirs(os.path.dirname(docs_path), exist_ok=True)
    
    with open(docs_path, "w") as f:
        f.write("# Inverse Rain Rate Modeling Results\n\n")
        f.write("This document tracks the results of our multi-stage rain rate narrowcasting implementation.\n\n")
        
        f.write("## Stage A: Pure Analytical Inversion (No ML)\n\n")
        f.write("The analytical inversion pipeline is formulated as:\n")
        f.write("1. **Calculate Total Gain**: $G_{\\text{total}} = \\text{EIRP} + G_{rx} - N_{\\text{floor}}$\n")
        f.write("2. **Calculate Excess Attenuation**: $\\text{Attn}_{\\text{excess}} = G_{\\text{total}} - \\text{SNR} - \\text{FSPL} - \\text{GL}$\n")
        f.write("3. **Filter Scintillation**: Apply zero-phase low-pass Butterworth filter (cutoff = 0.005 Hz) to get $\\widehat{\\text{RA}}$\n")
        f.write("4. **Invert ITU-R P.618 Model**: $\\widehat{R} = \\left( \\frac{\\max(0, \\widehat{\\text{RA}})}{k \\cdot L_{\\text{eff}}} \\right)^{1/\\alpha}$\n\n")
        
        f.write("### Rain Threshold Sensitivity Analysis\n")
        f.write("Tropospheric scintillation noise mimics low-rate rain, introducing massive False Positives when using a low rain/clear threshold (e.g. $0.1\\text{ mm/h}$). We analyze the sensitivity of the classification performance across different detection thresholds for stochastic rain scenarios:\n\n")
        
        for gs in stations_to_test:
            key = f"{gs['name']}_stochastic_rain"
            metrics = results_by_station[key]["threshold_metrics"]
            f.write(f"#### {gs['name']} (Stochastic Rain)\n\n")
            f.write("| Threshold (mm/h) | TP | FP | FN | TN | Precision | Recall | F1 Score |\n")
            f.write("|---|---|---|---|---|---|---|---|\n")
            for m in metrics:
                f.write(f"| {m['threshold']} | {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} | {m['precision']*100:.1f}% | {m['recall']*100:.1f}% | {m['f1']:.4f} |\n")
            f.write("\n")
            
        f.write("### Visual Validation & PR Analysis\n\n")
        for gs in stations_to_test:
            gs_low = gs["name"].lower().replace(" ", "")
            f.write(f"#### {gs['name']}\n")
            f.write(f"* **Stochastic Rain Time-Series**: ![Time Series](file:///home/satyansh/.gemini/antigravity-cli/brain/b30d89ad-2cb9-4e00-a92c-0ae53cdb775b/stage_a_{gs_low}_stochastic_rain.png)\n")
            f.write(f"* **Histogram Distribution Comparison**: ![Histogram](file:///home/satyansh/.gemini/antigravity-cli/brain/b30d89ad-2cb9-4e00-a92c-0ae53cdb775b/stage_a_{gs_low}_stochastic_rain_hist.png)\n")
            f.write(f"* **Precision-Recall Analysis**: ![PR Analysis](file:///home/satyansh/.gemini/antigravity-cli/brain/b30d89ad-2cb9-4e00-a92c-0ae53cdb775b/stage_a_{gs_low}_pr_analysis.png)\n\n")
            
    print(f"\nCompleted Stage A. Report written to {docs_path}")

if __name__ == "__main__":
    run_evaluation()
