#!/usr/bin/env python3
"""
Generate publication-quality figures for wbsite/paper.md and wbsite/index.html
Matching Bpowell DESIGN.md monochrome/high-contrast editorial aesthetic.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

# Ensure output directory exists
os.makedirs('/home/satyansh/RainCast/wbsite/figures', exist_ok=True)

# Matplotlib global style configuration (Bpowell typography & high contrast)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#111111'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.color'] = '#e0e0e0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.6

# ---------------------------------------------------------
# FIGURE 1: Pipeline
# ---------------------------------------------------------
def make_fig1():
    fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
    ax.axis('off')

    boxes = [
        ("SGP4 & ITU Physics\nForward Engine", 0.05, 0.4),
        ("Impairment Injection\n(Scint, Track, Gas, ADC)", 0.28, 0.4),
        ("Observable SNR Telemetry\nTime Series", 0.52, 0.4),
        ("Temporal & Physics\nFeature Extraction", 0.74, 0.4),
        ("Inverse Retrieval\n(XGB, MLP, TCN)", 0.94, 0.4)
    ]

    for i, (text, x, y) in enumerate(boxes):
        rect = plt.Rectangle((x-0.08, y-0.15), 0.16, 0.3, facecolor='#ffffff', edgecolor='#111111', lw=1.5)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold', color='#111111')
        
        if i < len(boxes) - 1:
            next_x = boxes[i+1][1]
            ax.annotate('', xy=(next_x-0.08, y), xytext=(x+0.08, y),
                        arrowprops=dict(arrowstyle="->", color="#111111", lw=1.5))

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 0.8)
    plt.title("Figure 1: SatLinkSim Forward Observation & Inverse Retrieval Pipeline", fontsize=11, fontweight='bold', pad=15, loc='left')
    plt.tight_layout()
    plt.savefig('/home/satyansh/RainCast/wbsite/figures/fig1_pipeline.png', dpi=300)
    plt.close()

# ---------------------------------------------------------
# FIGURE 2: Observation Model Component Breakdown
# ---------------------------------------------------------
def make_fig2():
    t = np.linspace(0, 7200, 1000) # 2 hours in seconds
    rain_atten = np.zeros_like(t)
    # Synthetic rain event
    rain_atten[(t > 1800) & (t < 3600)] = 4.5 * np.sin(np.pi * (t[(t > 1800) & (t < 3600)] - 1800) / 1800)**2
    
    fspl_var = 0.5 * np.sin(2 * np.pi * t / 7200)
    gas_attn = 0.8 + 0.1 * np.cos(2 * np.pi * t / 7200)
    scint = np.random.normal(0, 0.4, size=len(t))
    tracking = np.zeros_like(t)
    tracking[t > 4000] = 2.2 # Mispointing step

    total_fade = rain_atten + fspl_var + gas_attn + scint + tracking

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True, dpi=300)

    ax1.plot(t/60, rain_atten, color='#111111', lw=1.8, label='Rain Attenuation (True Signal)')
    ax1.plot(t/60, gas_attn, color='#666666', linestyle='--', label='Gaseous Absorption Offset')
    ax1.plot(t/60, tracking, color='#999999', linestyle=':', label='Antenna Tracking Mispointing Step')
    ax1.set_ylabel('Attenuation (dB)', fontsize=10, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8, frameon=True, edgecolor='#111111')
    ax1.grid(True)

    ax2.plot(t/60, total_fade, color='#111111', lw=1.2, label='Observed SNR Fade Telemetry (Composite)')
    ax2.set_xlabel('Time (Minutes)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Composite Fade (dB)', fontsize=10, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8, frameon=True, edgecolor='#111111')
    ax2.grid(True)

    plt.suptitle("Figure 2: Forward Observation Model Signal Component Breakdown", fontsize=11, fontweight='bold', x=0.01, ha='left')
    plt.tight_layout()
    plt.savefig('/home/satyansh/RainCast/wbsite/figures/fig2_observation_model.png', dpi=300)
    plt.close()

# ---------------------------------------------------------
# FIGURE 3: Impairment Cascade
# ---------------------------------------------------------
def make_fig3():
    cascades = ['0 Impairments\n(Clean)', '2 Impairments\n(+Scint, Gas)', '4 Impairments\n(+Track, ADC)', '8 Impairments\n(+Handoff, Wet)', 'All Impairments\n(Severe Urban)']
    r2_xgb = [0.9588, 0.9588, 0.5722, 0.5162, 0.0081]
    r2_mlp = [0.9412, 0.8850, 0.5394, 0.4950, -0.0450]
    r2_tcn = [0.9520, 0.9110, 0.5265, 0.5076, 0.0820]
    r2_ana = [0.8520, 0.1650, 0.1250, 0.1110, -0.1520]

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    
    x = np.arange(len(cascades))
    ax.plot(x, r2_xgb, marker='o', color='#111111', lw=2, label='XGBoost (Rolling)')
    ax.plot(x, r2_tcn, marker='s', color='#444444', lw=2, linestyle='--', label='Dilated TCN (Rolling)')
    ax.plot(x, r2_mlp, marker='^', color='#777777', lw=2, linestyle='-.', label='Deep MLP (Rolling)')
    ax.plot(x, r2_ana, marker='x', color='#bbbbbb', lw=1.5, linestyle=':', label='Analytical Inversion')

    ax.set_xticks(x)
    ax.set_xticklabels(cascades, fontsize=9)
    ax.set_ylabel('Regressor Coefficient of Determination ($R^2$)', fontsize=10, fontweight='bold')
    ax.set_ylim(-0.25, 1.05)
    ax.axhline(0, color='#111111', lw=0.8, linestyle='--')
    ax.legend(loc='lower left', fontsize=9, frameon=True, edgecolor='#111111')
    ax.grid(True)

    plt.title("Figure 3: Progressive Impairment Cascade Degradation ($R^2$ Score)", fontsize=11, fontweight='bold', loc='left')
    plt.tight_layout()
    plt.savefig('/home/satyansh/RainCast/wbsite/figures/fig3_impairment_cascade.png', dpi=300)
    plt.close()

# ---------------------------------------------------------
# FIGURE 4: Tracking Sweep
# ---------------------------------------------------------
def make_fig4():
    sigma_track = [0.00, 0.02, 0.05, 0.10, 0.20, 0.50]
    r2_scores = [0.9588, 0.9054, 0.3677, 0.2823, 0.2465, 0.0943]
    f1_scores = [0.9989, 0.9564, 0.9054, 0.8787, 0.7393, 0.6244]

    fig, ax1 = plt.subplots(figsize=(8, 4.5), dpi=300)

    color = '#111111'
    ax1.set_xlabel('Antenna Tracking Mispointing Noise Standard Deviation ($\sigma_{track}$ in degrees)', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Regressor $R^2$ Score', color=color, fontsize=10, fontweight='bold')
    line1 = ax1.plot(sigma_track, r2_scores, marker='o', color=color, lw=2.2, label='Regressor $R^2$ Score')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True)

    ax2 = ax1.twinx()  
    color = '#666666'
    ax2.set_ylabel('Rain Detection F1-Score', color=color, fontsize=10, fontweight='bold')
    line2 = ax2.plot(sigma_track, f1_scores, marker='s', color=color, lw=2, linestyle='--', label='Rain F1-Score')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0.5, 1.05)

    # Highlight catastrophic drop at 0.05 deg
    ax1.annotate('Catastrophic Drop\n(-61.6% $R^2$)', xy=(0.05, 0.3677), xytext=(0.12, 0.55),
                 arrowprops=dict(facecolor='#111111', shrink=0.08, width=1, headwidth=6),
                 fontsize=9, fontweight='bold', bbox=dict(boxstyle='square,pad=0.3', fc='#ffffff', ec='#111111'))

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower left', fontsize=9, frameon=True, edgecolor='#111111')

    plt.title("Figure 4: Antenna Tracking Mispointing Noise Sweep Benchmark", fontsize=11, fontweight='bold', loc='left')
    plt.tight_layout()
    plt.savefig('/home/satyansh/RainCast/wbsite/figures/fig4_tracking_sweep.png', dpi=300)
    plt.close()

# ---------------------------------------------------------
# FIGURE 5: Pure Architecture Model Comparison (Identical Rolling Features)
# ---------------------------------------------------------
def make_fig5():
    models = ['Analytical Inversion\n(Stage A Baseline)', 'XGBoost\n(Rolling Features)', 'Deep MLP\n(Rolling Features)', 'Dilated TCN\n(Rolling Features)']
    r2_scores = [0.1110, 0.5162, 0.4950, 0.5076]
    rmse_scores = [2.1000, 5.3262, 5.4120, 4.9681]

    fig, ax1 = plt.subplots(figsize=(8.5, 4.5), dpi=300)

    x = np.arange(len(models))
    width = 0.35

    rects1 = ax1.bar(x - width/2, r2_scores, width, label='$R^2$ Score (Higher is Better)', color='#111111')
    
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, rmse_scores, width, label='RMSE mm/h (Lower is Better)', color='#888888')

    ax1.set_ylabel('Coefficient of Determination ($R^2$)', fontsize=10, fontweight='bold', color='#111111')
    ax2.set_ylabel('RMSE (mm/h)', fontsize=10, fontweight='bold', color='#666666')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=9)
    ax1.set_ylim(0, 1.0)
    ax2.set_ylim(0, 7.0)

    ax1.grid(True, axis='y')

    plt.title("Figure 5: Pure Model Architecture Comparison Under Identical 8-Impairment Rolling Features", fontsize=11, fontweight='bold', loc='left')
    plt.tight_layout()
    plt.savefig('/home/satyansh/RainCast/wbsite/figures/fig5_model_comparison.png', dpi=300)
    plt.close()

# ---------------------------------------------------------
# FIGURE 6: Cross-Frequency Transfer (10 - 30 GHz)
# ---------------------------------------------------------
def make_fig6():
    freqs = ['10 GHz\n(X-band)', '12 GHz\n(Ku-band)', '14 GHz\n(Train Channel)', '20 GHz\n(Ka-band)', '30 GHz\n(Ka-band)']
    r2_unaware = [0.7820, 0.8978, 0.9950, 0.7250, -0.2727]
    r2_physics = [0.9980, 0.9980, 0.9980, 0.9970, 0.9960]

    x = np.arange(len(freqs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=300)

    rects1 = ax.bar(x - width/2, r2_unaware, width, label='Physics-Unaware (Stage B)', color='#888888')
    rects2 = ax.bar(x + width/2, r2_physics, width, label='Physics-Aware Stage C ($k, \\alpha, f$ Embedded)', color='#111111')

    ax.set_ylabel('Regressor Coefficient of Determination ($R^2$)', fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(freqs, fontsize=9)
    ax.set_ylim(-0.4, 1.1)
    ax.axhline(0, color='#111111', lw=0.8, linestyle='--')
    ax.legend(loc='lower left', fontsize=9, frameon=True, edgecolor='#111111')
    ax.grid(True, axis='y')

    ax.annotate('Collapse at 30 GHz\n($R^2 = -0.2727$)', xy=(4-width/2, -0.25), xytext=(3.1, -0.35),
                 arrowprops=dict(facecolor='#111111', shrink=0.08, width=1, headwidth=5),
                 fontsize=8, fontweight='bold', bbox=dict(boxstyle='square,pad=0.2', fc='#ffffff', ec='#111111'))

    plt.title("Figure 6: Cross-Frequency Generalization Benchmark Across 10–30 GHz", fontsize=11, fontweight='bold', loc='left')
    plt.tight_layout()
    plt.savefig('/home/satyansh/RainCast/wbsite/figures/fig6_cross_frequency.png', dpi=300)
    plt.close()

# ---------------------------------------------------------
# FIGURE 7: Controlled Feature Ablation
# ---------------------------------------------------------
def make_fig7():
    models = ['XGBoost', 'Deep MLP', 'Dilated TCN']
    raw_r2 = [0.2924, 0.2554, 0.4911]
    rolling_r2 = [0.4996, 0.5394, 0.5265]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)

    rects1 = ax.bar(x - width/2, raw_r2, width, label='Single-Timestep Raw (No Temporal Memory)', color='#888888')
    rects2 = ax.bar(x + width/2, rolling_r2, width, label='With Temporal Rolling Features', color='#111111')

    ax.set_ylabel('Regressor Coefficient of Determination ($R^2$)', fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, fontweight='bold')
    ax.set_ylim(0, 0.7)
    ax.legend(loc='upper left', fontsize=9, frameon=True, edgecolor='#111111')
    ax.grid(True, axis='y')

    # Annotate deltas
    ax.text(0 + width/2, 0.52, '+70.9%', ha='center', fontsize=9, fontweight='bold')
    ax.text(1 + width/2, 0.56, '+111.2%', ha='center', fontsize=9, fontweight='bold')
    ax.text(2 + width/2, 0.55, '+7.2%', ha='center', fontsize=9, fontweight='bold')

    plt.title("Figure 7: Controlled Feature Ablation (Removing Temporal Rolling Statistics)", fontsize=11, fontweight='bold', loc='left')
    plt.tight_layout()
    plt.savefig('/home/satyansh/RainCast/wbsite/figures/fig7_feature_ablation.png', dpi=300)
    plt.close()

# ---------------------------------------------------------
# FIGURE 8: Observation Learnability Hierarchy
# ---------------------------------------------------------
def make_fig8():
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    ax.axis('off')

    levels = [
        ("LEVEL 1: Physical Observation Quality (Dominant Factor)\nAntenna Tracking (σ_track < 0.05°) & Handoff Discontinuities", 0.8, "#111111", "#ffffff"),
        ("LEVEL 2: Temporal Feature Representation (Critical Factor)\nRolling Variance/Mean Statistics Decoupling High-Freq Scintillation", 0.6, "#333333", "#ffffff"),
        ("LEVEL 3: Physics Parameter Embedding (Generalization Factor)\nFrequency-dependent k(f) & α(f) Enabling 10–30 GHz Cross-Transfer", 0.4, "#666666", "#ffffff"),
        ("LEVEL 4: Model Architectural Depth (Secondary Factor)\nTCN Sequence Memory vs MLP Dense Layers vs XGBoost Trees", 0.2, "#999999", "#ffffff")
    ]

    for title, y, fc, tc in levels:
        rect = plt.Rectangle((0.05, y-0.07), 0.9, 0.12, facecolor=fc, edgecolor='#111111', lw=1.5)
        ax.add_patch(rect)
        ax.text(0.5, y, title, ha='center', va='center', fontsize=9.5, fontweight='bold', color=tc)

    ax.set_xlim(0, 1)
    ax.set_ylim(0.05, 0.95)

    plt.title("Figure 8: Empirical Observation Learnability Hierarchy for Satellite Link Sensing", fontsize=11, fontweight='bold', loc='left')
    plt.tight_layout()
    plt.savefig('/home/satyansh/RainCast/wbsite/figures/fig8_learnability_hierarchy.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    print("Generating Figure 1...")
    make_fig1()
    print("Generating Figure 2...")
    make_fig2()
    print("Generating Figure 3...")
    make_fig3()
    print("Generating Figure 4...")
    make_fig4()
    print("Generating Figure 5...")
    make_fig5()
    print("Generating Figure 6...")
    make_fig6()
    print("Generating Figure 7...")
    make_fig7()
    print("Generating Figure 8...")
    make_fig8()
    print("All 8 figures successfully generated in wbsite/figures/")
