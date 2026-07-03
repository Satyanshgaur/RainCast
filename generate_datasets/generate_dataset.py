#!/usr/bin/env python3
import os
import sys
import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# Set matplotlib to run headlessly
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add src to python path so we can import satlinksim
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(root_dir, "src"))

from satlinksim.satellite_link_sim import SimulationEngine
from satlinksim.domain.models import Constellation, Satellite
from satlinksim.domain.link.budget import fspl_db
from satlinksim.domain.link.itu_models import gaseous_absorption_db
from satlinksim.ground_stations import GROUND_STATIONS

def parse_tle_epoch(line1):
    try:
        yr = int(line1[18:20])
        day = float(line1[20:32])
        year = 2000 + yr if yr < 57 else 1900 + yr
        dt = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day - 1)
        return dt
    except Exception:
        return None

def get_constellation_name(name):
    n = name.lower()
    if "starlink" in n:
        return "Starlink"
    elif "oneweb" in n:
        return "OneWeb"
    elif "iridium" in n:
        return "Iridium"
    elif "globalstar" in n:
        return "Globalstar"
    else:
        return "GEO/Other"

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate simulated satellite link performance datasets.")
    parser.add_argument("--size", type=str, choices=["small", "medium"], default="small",
                        help="Size of the dataset to generate (small ~100k rows, medium ~1M rows).")
    args = parser.parse_args()
    
    size = args.size
    print(f"Initializing dataset generation (size: {size}) with refreshed TLEs and stratified constellations...")
    
    # Database path
    db_path = os.path.join(root_dir, "src", "satlinksim", "satellites.db")
    if not os.path.exists(db_path):
        db_path = os.path.join(root_dir, "satellites.db")
    
    if not os.path.exists(db_path):
        print(f"Error: satellites.db not found at {db_path}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Loading satellites from database: {db_path}")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name, norad_id, tle_line1, tle_line2 FROM satellites")
    sat_rows = cur.fetchall()
    conn.close()
    
    all_satellites = []
    for row in sat_rows:
        sat = Satellite(name=row[0], norad_id=row[1], tle_line1=row[2], tle_line2=row[3])
        all_satellites.append(sat)
                
    print(f"Loaded {len(all_satellites)} total satellites from database.")
    
    # Stratified constellation grouping
    starlink_sats = [s for s in all_satellites if "starlink" in s.name.lower()]
    oneweb_sats = [s for s in all_satellites if "oneweb" in s.name.lower()]
    iridium_sats = [s for s in all_satellites if "iridium" in s.name.lower()]
    globalstar_sats = [s for s in all_satellites if "globalstar" in s.name.lower()]
    other_sats = [s for s in all_satellites if not any(x in s.name.lower() for x in ["starlink", "oneweb", "iridium", "globalstar"])]
    
    print(f"Database breakdown: Starlink={len(starlink_sats)}, OneWeb={len(oneweb_sats)}, Iridium={len(iridium_sats)}, Globalstar={len(globalstar_sats)}, GEO/Other={len(other_sats)}")
    
    # Sampling: Starlink (3000), OneWeb (500), Iridium (all/80), Globalstar (all/28), GEO/Other (400)
    import random
    random.seed(42)
    
    sampled_starlink = random.sample(starlink_sats, min(len(starlink_sats), 3000))
    sampled_oneweb = random.sample(oneweb_sats, min(len(oneweb_sats), 500))
    sampled_iridium = random.sample(iridium_sats, min(len(iridium_sats), 80))
    sampled_globalstar = random.sample(globalstar_sats, min(len(globalstar_sats), 28))
    sampled_other = random.sample(other_sats, min(len(other_sats), 400))
    
    satellites = sampled_starlink + sampled_oneweb + sampled_iridium + sampled_globalstar + sampled_other
    print(f"Sampled {len(satellites)} satellites for simulation:")
    print(f"  -> Starlink: {len(sampled_starlink)}")
    print(f"  -> OneWeb: {len(sampled_oneweb)}")
    print(f"  -> Iridium: {len(sampled_iridium)}")
    print(f"  -> Globalstar: {len(sampled_globalstar)}")
    print(f"  -> GEO/Other: {len(sampled_other)}")
    
    tle_dates = []
    for sat in satellites:
        if sat.tle_line1:
            dt = parse_tle_epoch(sat.tle_line1)
            if dt:
                tle_dates.append(dt)
                
    # Determine TLE epoch range
    if tle_dates:
        min_tle_date = min(tle_dates)
        max_tle_date = max(tle_dates)
        tle_info_str = f"{min_tle_date.strftime('%Y-%m-%d')} to {max_tle_date.strftime('%Y-%m-%d')}"
        print(f"Selected satellite TLE dates range from: {tle_info_str}")
    else:
        min_tle_date = None
        max_tle_date = None
        tle_info_str = "Unknown TLE epoch range"
        
    constellation = Constellation(name="DatasetConstellation", satellites=satellites)
    
    # Target stations (Delhi, Sao Paulo, Berlin)
    station_names = ["Delhi", "Sao Paulo", "Berlin"]
    stations = [gs for gs in GROUND_STATIONS if gs["name"] in station_names]
    
    # Map name display for Sao Paulo to "São Paulo"
    display_names = {
        "Delhi": "Delhi",
        "Berlin": "Berlin",
        "Sao Paulo": "São Paulo"
    }
    
    # Carrier frequencies in Hz (spanning Ku and Ka bands)
    frequencies = [12.0e9, 14.0e9, 20.0e9, 30.0e9]
    
    # Setup step parameters
    base_steps = 8334
    num_epochs = 10 if size == "medium" else 1
    n_steps = base_steps * num_epochs
    dt_s = 60 # 1 minute steps
    
    # Simulating on July 3, 2026 (matching updated TLE epoch)
    start_time = datetime(2026, 7, 3, 12, 0, 0, tzinfo=timezone.utc)
    
    print(f"Starting simulation runs starting on {start_time.isoformat()} ({num_epochs} epoch(s) of {base_steps} steps)...")
    engine = SimulationEngine()
    
    all_dfs = []
    plot_data = {} # To store raw run data for plotting
    
    for gs in stations:
        gs_name = gs["name"]
        gs_display = display_names.get(gs_name, gs_name)
        plot_data[gs_display] = {}
        print(f"Simulating ground station: {gs_display}")
        
        for freq in frequencies:
            freq_ghz = freq / 1e9
            band = "Ku" if freq_ghz < 18 else "Ka"
            print(f"  Frequency: {freq_ghz:.1f} GHz ({band} band)")
            
            run_dfs = []
            for epoch in range(num_epochs):
                epoch_start = start_time + epoch * timedelta(days=6)
                if size == "medium":
                    print(f"    Epoch {epoch+1}/{num_epochs} starting at {epoch_start.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Fixed reproducible seed per epoch run
                seed = 42 + len(all_dfs) * 10 + epoch
                results = engine.simulate_all_batched(
                    ground_stations=[gs],
                    n_steps=base_steps,
                    dt_s=dt_s,
                    start_time=epoch_start,
                    freq_hz=freq,
                    constellation=constellation,
                    seed=seed,
                    force_rain=False
                )
                
                res = results[0]
                
                # Generate timestamps
                times = [epoch_start + timedelta(seconds=i*dt_s) for i in range(base_steps)]
                
                # Extract returned series
                elevations = np.array(res.elevation_series)
                slant_ranges = np.array(res.slant_range_series)
                rain_rates = np.array(res.rain_series)
                rain_db = np.array(res.rain_db_series)
                scintillation = np.array(res.scint_series)
                snr = np.array(res.snr_series)
                pkt_loss = np.array(res.pkt_loss_series)
                sat_names = res.sat_name_series
                
                # Classify active constellation for each timestep
                active_consts = [get_constellation_name(name) for name in sat_names]
                
                # Recalculate missing variables to match simulator logic
                fspl = fspl_db(freq, slant_ranges)
                gaseous = gaseous_absorption_db(freq_ghz, elevations, gs["wv_g_m3"])
                total_path_loss = fspl + gaseous + rain_db + scintillation
                
                # Store plot data from epoch 0 (to make plots clean and represent the first 5.8 days)
                if epoch == 0:
                    plot_data[gs_display][freq_ghz] = {
                        "times": times,
                        "snr": snr,
                        "rain_rate": rain_rates,
                        "rain_db": rain_db,
                        "total_path_loss": total_path_loss,
                        "elevation": elevations,
                        "slant_range": slant_ranges,
                        "active_sat": sat_names,
                        "active_const": active_consts
                    }
                
                # Create DataFrame for this epoch
                df = pd.DataFrame({
                    "timestamp": times,
                    "ground_station_identity": [gs_display] * base_steps,
                    "active_satellite_identity": sat_names,
                    "active_satellite_constellation": active_consts,
                    "carrier_frequency": [freq] * base_steps,
                    "elevation_angle": elevations,
                    "slant_range": slant_ranges,
                    "FSPL": fspl,
                    "gaseous_attenuation": gaseous,
                    "rain_attenuation": rain_db,
                    "scintillation_loss": scintillation,
                    "total_path_loss": total_path_loss,
                    "received_snr": snr,
                    "packet_loss_probability": pkt_loss,
                    "whether_system_in_rain_event": rain_rates > 0.0,
                    "instantaneous_rain_rate": rain_rates
                })
                run_dfs.append(df)
                
                # Clear references and force garbage collection
                res = None
                results = None
                import gc
                gc.collect()
            
            # Combine all epochs for this run
            combined_run_df = pd.concat(run_dfs, ignore_index=True)
            all_dfs.append(combined_run_df)
            
    print("Combining simulation runs into single dataset...")
    full_df = pd.concat(all_dfs, ignore_index=True)
    
    # Save directory
    output_dir = os.path.join(root_dir, "datasets", "satellite_time_series")
    os.makedirs(output_dir, exist_ok=True)
    
    parquet_name = f"satellite_time_series_{size}.parquet"
    parquet_path = os.path.join(output_dir, parquet_name)
    print(f"Saving dataset to {parquet_path}...")
    full_df.to_parquet(parquet_path, index=False)
    
    # Generate metadata JSON
    metadata_path = os.path.join(output_dir, "metadata.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}
    else:
        metadata = {}
        
    metadata["dataset_name"] = "Satellite Link Performance Time Series Dataset"
    metadata["description"] = (
        "A simulated time-series dataset of satellite link performance covering multiple "
        "geographic locations (Delhi, São Paulo, Berlin) and multiple carrier frequencies in the "
        "Ku and Ka bands. Developed for studying link budget variation, attenuation modeling, and "
        "predictive packet loss analysis."
    )
    metadata["creation_time"] = datetime.now(timezone.utc).isoformat()
    metadata["satellite_tle_metadata"] = {
        "source_database": db_path,
        "tle_reference_date": "2026-07-03",
        "tle_epoch_range": tle_info_str,
        "min_tle_epoch": min_tle_date.isoformat() if min_tle_date else None,
        "max_tle_epoch": max_tle_date.isoformat() if max_tle_date else None,
        "total_satellites_loaded": len(all_satellites),
        "sampled_satellites_simulated": len(satellites),
        "sampled_constellations_breakdown": {
            "Starlink": len(sampled_starlink),
            "OneWeb": len(sampled_oneweb),
            "Iridium": len(sampled_iridium),
            "Globalstar": len(sampled_globalstar),
            "GEO/Other": len(sampled_other)
        }
    }
    metadata["simulation_parameters"] = {
        "start_time": start_time.isoformat(),
        "dt_s": dt_s,
        "frequencies_hz": frequencies,
        "ground_stations": [
            {
                "name": gs_display,
                "latitude": gs["latitude"],
                "longitude": gs["longitude"],
                "altitude_km": gs["altitude_km"],
                "antenna_diam_m": gs["antenna_diam_m"],
                "climate_type": "heavy monsoon" if gs_name == "Delhi" else ("tropical" if gs_name == "Sao Paulo" else "temperate")
            }
            for gs in stations
            for gs_name, gs_display in [(gs["name"], display_names[gs["name"]])]
        ]
    }
    
    if "datasets" not in metadata:
        metadata["datasets"] = {}
        
    end_time_val = start_time + timedelta(seconds=n_steps * dt_s)
    days_val = (n_steps * dt_s) / (24 * 3600)
    
    metadata["datasets"][size] = {
        "filename": f"satellite_time_series_{size}.parquet",
        "total_rows": len(full_df),
        "n_steps_per_run": n_steps,
        "coverage_duration": f"{days_val:.2f} days",
        "end_time": end_time_val.isoformat()
    }
    
    metadata["columns"] = {
        "timestamp": "UTC timestamp of the simulation step",
        "ground_station_identity": "Name of the ground station (Delhi, São Paulo, Berlin)",
        "active_satellite_identity": "Name of the active satellite currently tracked",
        "active_satellite_constellation": "Constellation group of the active satellite (Starlink, OneWeb, Iridium, Globalstar, GEO/Other)",
        "carrier_frequency": "Carrier frequency of the satellite link in Hz",
        "elevation_angle": "Elevation angle of the active satellite in degrees",
        "slant_range": "Slant range to the active satellite in km",
        "FSPL": "Free Space Path Loss in dB",
        "gaseous_attenuation": "Gaseous absorption attenuation in dB (from ITU-R P.676)",
        "rain_attenuation": "Rain attenuation in dB (from ITU-R P.618 and P.837)",
        "scintillation_loss": "Scintillation loss in dB (from ITU-R P.618)",
        "total_path_loss": "Total path loss in dB (FSPL + gaseous + rain + scintillation)",
        "received_snr": "Received signal-to-noise ratio in dB",
        "packet_loss_probability": "Calculated packet loss probability based on SNR",
        "whether_system_in_rain_event": "Boolean indicating if the instantaneous rain rate is greater than 0",
        "instantaneous_rain_rate": "Instantaneous rain rate in mm/h generated via a correlated rain process"
    }
    
    print(f"Saving merged metadata to {metadata_path}...")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    medium_meta_path = os.path.join(output_dir, "metadata_medium.json")
    if os.path.exists(medium_meta_path):
        try:
            os.remove(medium_meta_path)
            print(f"Removed redundant {medium_meta_path}")
        except Exception:
            pass
        
    # Generate stations.csv
    stations_data = []
    for gs in stations:
        gs_name = gs["name"]
        gs_display = display_names[gs["name"]]
        stations_data.append({
            "name": gs_display,
            "latitude": gs["latitude"],
            "longitude": gs["longitude"],
            "altitude_km": gs["altitude_km"],
            "eirp_dbw": gs["eirp_dbw"],
            "g_rx_dbi": gs["g_rx_dbi"],
            "system_temp_k": gs["system_temp_k"],
            "antenna_diam_m": gs["antenna_diam_m"],
            "wv_g_m3": gs["wv_g_m3"],
            "humidity_pct": gs["humidity_pct"],
            "climate_type": "heavy monsoon" if gs_name == "Delhi" else ("tropical" if gs_name == "Sao Paulo" else "temperate")
        })
    stations_df = pd.DataFrame(stations_data)
    stations_path = os.path.join(output_dir, "stations.csv")
    print(f"Saving stations to {stations_path}...")
    stations_df.to_csv(stations_path, index=False)
    
    # Generate satellites.csv
    satellites_data = []
    for sat in satellites:
        satellites_data.append({
            "name": sat.name,
            "norad_id": sat.norad_id,
            "constellation": get_constellation_name(sat.name),
            "tle_line1": sat.tle_line1,
            "tle_line2": sat.tle_line2
        })
    satellites_df = pd.DataFrame(satellites_data)
    satellites_path = os.path.join(output_dir, "satellites.csv")
    print(f"Saving satellites to {satellites_path}...")
    satellites_df.to_csv(satellites_path, index=False)

    # Generate column_dictionary.csv
    column_dict_data = [
        {"Column": "timestamp", "Type": "datetime", "Units": "UTC", "Description": "Simulation timestamp (1-minute resolution)"},
        {"Column": "ground_station_identity", "Type": "string", "Units": "N/A", "Description": "Name of the ground station location (Delhi, São Paulo, Berlin)"},
        {"Column": "active_satellite_identity", "Type": "string", "Units": "N/A", "Description": "Name of the active satellite currently tracked"},
        {"Column": "active_satellite_constellation", "Type": "string", "Units": "N/A", "Description": "Constellation group of the active satellite (Starlink, OneWeb, Iridium, Globalstar, GEO/Other)"},
        {"Column": "carrier_frequency", "Type": "float", "Units": "Hz", "Description": "Carrier frequency of the satellite link"},
        {"Column": "elevation_angle", "Type": "float", "Units": "degrees", "Description": "Ground-to-satellite elevation angle"},
        {"Column": "slant_range", "Type": "float", "Units": "km", "Description": "Slant range distance from ground station to satellite"},
        {"Column": "FSPL", "Type": "float", "Units": "dB", "Description": "Free Space Path Loss"},
        {"Column": "gaseous_attenuation", "Type": "float", "Units": "dB", "Description": "Atmospheric gas absorption loss (ITU-R P.676)"},
        {"Column": "rain_attenuation", "Type": "float", "Units": "dB", "Description": "Rain attenuation (ITU-R P.618 and P.837)"},
        {"Column": "scintillation_loss", "Type": "float", "Units": "dB", "Description": "Tropospheric scintillation loss (ITU-R P.618)"},
        {"Column": "total_path_loss", "Type": "float", "Units": "dB", "Description": "Total path loss (FSPL + gaseous + rain + scintillation)"},
        {"Column": "received_snr", "Type": "float", "Units": "dB", "Description": "Received signal-to-noise ratio"},
        {"Column": "packet_loss_probability", "Type": "float", "Units": "N/A", "Description": "Calculated probability of packet loss based on link SNR"},
        {"Column": "whether_system_in_rain_event", "Type": "boolean", "Units": "N/A", "Description": "Flag indicating if instantaneous rain rate > 0"},
        {"Column": "instantaneous_rain_rate", "Type": "float", "Units": "mm/h", "Description": "Instantaneous rain rate generated via a correlated rain process"}
    ]
    col_df = pd.DataFrame(column_dict_data)
    col_dict_path = os.path.join(output_dir, "column_dictionary.csv")
    print(f"Saving column dictionary to {col_dict_path}...")
    col_df.to_csv(col_dict_path, index=False)
    
    # Generate README.md
    readme_content = f"""# Satellite Link Performance Time Series Dataset

This directory contains a simulated time series dataset representing satellite link performance under various geographic, climatic, and RF frequency configurations. The dataset is generated directly from the `satlinksim` physics-based simulation engine.

## Files
- `satellite_time_series_small.parquet`: The main time series dataset containing {len(full_df):,} timesteps.
- `metadata.json`: Dataset configuration, descriptions of columns, TLE details, and global parameters.
- `column_dictionary.csv`: A map of columns, types, units, and descriptions.
- `stations.csv`: Ground stations configured in the simulation (Delhi, São Paulo, and Berlin), detailing physical hardware properties and local weather settings.
- `satellites.csv`: Satellites included in the constellation simulation, along with their NORAD IDs, constellations, and TLE orbit parameters.
- `sample_plots/`: Visualizations of the simulated data:
  - `snr_timeseries.png`: Received SNR over time for the three stations.
  - `rain_event.png`: Close-up of a rain event showing rain rate and rain attenuation.
  - `attenuation_vs_frequency.png`: Comparison of rain attenuation across frequencies in Ku and Ka bands.
  - `station_map.png`: Ground station coordinates and climates plotted on a 2D map.

## Dataset Structure
Each row in `satellite_time_series_small.parquet` represents a single timestep (1-minute resolution) for a specific ground station and carrier frequency combination.

## Simulation Setup
- **Locations**:
  - **Delhi**: Heavy monsoon climate (ITU rain zone K).
  - **São Paulo**: Tropical convective rain climate (ITU rain zone N/P).
  - **Berlin**: Temperate maritime climate (ITU rain zone E).
- **Frequencies**: Spans Ku-band (12 GHz, 14 GHz) and Ka-band (20 GHz, 30 GHz).
- **Epoch**: July 3, 2026 (matching the refreshed satellite database TLE epoch: {tle_info_str}).
- **Resolution**: 1-minute intervals (`dt_s = 60`).
- **Duration**: {n_steps} steps (approx. 5.8 days) per run, leading to a total dataset size of {len(full_df):,} rows.
- **Constellations Simulated**: Starlink, OneWeb, Iridium, Globalstar, and GEO/Others (4,008 total satellites in the pool).
"""
    
    readme_path = os.path.join(output_dir, "README.md")
    print(f"Saving README to {readme_path}...")
    with open(readme_path, "w") as f:
        f.write(readme_content)
        
    # Generate sample plots
    plots_dir = os.path.join(output_dir, "sample_plots")
    os.makedirs(plots_dir, exist_ok=True)
    print("Generating sample plots...")
    
    # Plot 1: snr_timeseries.png
    plt.figure(figsize=(10, 5))
    colors = {"Delhi": "#e74c3c", "Berlin": "#3498db", "São Paulo": "#2ecc71"}
    
    for station_name in ["Delhi", "Berlin", "São Paulo"]:
        data = plot_data[station_name][20.0]
        t = data["times"][:1440]
        snr = data["snr"][:1440]
        plt.plot(t, snr, label=station_name, color=colors[station_name], alpha=0.85)
        
    plt.title("Received Link SNR at 20 GHz (Ka Band) Over 24 Hours", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Time (UTC)", fontsize=11)
    plt.ylabel("Received SNR (dB)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "snr_timeseries.png"), dpi=150)
    plt.close()
    
    # Plot 2: rain_event.png
    sp_data = plot_data["São Paulo"][30.0]
    rain_rates = sp_data["rain_rate"]
    rain_db = sp_data["rain_db"]
    times = sp_data["times"]
    
    rain_indices = np.where(rain_rates > 0.1)[0]
    if len(rain_indices) > 0:
        peak_idx = rain_indices[np.argmax(rain_rates[rain_indices])]
        start_idx = max(0, peak_idx - 120)
        end_idx = min(n_steps, peak_idx + 120)
    else:
        start_idx = 0
        end_idx = 240
        
    t_window = times[start_idx:end_idx]
    rain_window = rain_rates[start_idx:end_idx]
    atten_window = rain_db[start_idx:end_idx]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(t_window, rain_window, color="#2ecc71", label="Rain Rate")
    ax1.set_ylabel("Rain Rate (mm/h)", color="#27ae60", fontsize=11)
    ax1.tick_params(axis='y', labelcolor="#27ae60")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.set_title("Close-up of a Simulated Rain Event and Corresponding Attenuation (São Paulo, 30 GHz)", fontsize=13, fontweight="bold", pad=12)
    
    ax2.plot(t_window, atten_window, color="#e67e22", label="Rain Attenuation")
    ax2.set_ylabel("Rain Attenuation (dB)", color="#d35400", fontsize=11)
    ax2.tick_params(axis='y', labelcolor="#d35400")
    ax2.set_xlabel("Time (UTC)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "rain_event.png"), dpi=150)
    plt.close()
    
    # Plot 3: attenuation_vs_frequency.png
    max_attens = []
    freq_labels = ["12 GHz\n(Ku)", "14 GHz\n(Ku)", "20 GHz\n(Ka)", "30 GHz\n(Ka)"]
    
    for f in [12.0, 14.0, 20.0, 30.0]:
        data = plot_data["Delhi"][f]
        max_attens.append(np.max(data["rain_db"]))
        
    plt.figure(figsize=(8, 5))
    bar_colors = ["#5dade2", "#2980b9", "#f5b041", "#e67e22"]
    plt.bar(freq_labels, max_attens, color=bar_colors, edgecolor="#2c3e50", width=0.6, alpha=0.9)
    plt.title("Frequency Scaling: Peak Rain Attenuation (Delhi)", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Peak Rain Attenuation (dB)", fontsize=11)
    plt.xlabel("Carrier Frequency & Band", fontsize=11)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    
    for idx, val in enumerate(max_attens):
        plt.text(idx, val + 0.1, f"{val:.1f} dB", ha="center", va="bottom", fontsize=10, fontweight="bold")
        
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "attenuation_vs_frequency.png"), dpi=150)
    plt.close()
    
    # Plot 4: station_map.png
    lats = [gs["latitude"] for gs in stations]
    lons = [gs["longitude"] for gs in stations]
    names = [display_names[gs["name"]] for gs in stations]
    climates = ["Heavy Monsoon", "Temperate", "Tropical"]
    
    plt.figure(figsize=(9, 5.5))
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.axhline(0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
    plt.axvline(0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
    
    plt.fill_between([-180, 180], -23.5, 23.5, color="#f9ebd2", alpha=0.2, label="Tropics Band (high rain rates)")
    
    for lat, lon, name, climate in zip(lats, lons, names, climates):
        plt.scatter(lon, lat, s=150, color=colors[name], edgecolors="black", zorder=5, label=f"{name} ({climate})")
        offset = 5
        if name == "São Paulo":
            plt.annotate(f"{name}\n({lat}°, {lon}°)\n{climate}", (lon, lat), textcoords="offset points", 
                         xytext=(offset, -25), ha='left', va='center', fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", edgecolor="gray", alpha=0.8))
        else:
            plt.annotate(f"{name}\n({lat}°, {lon}°)\n{climate}", (lon, lat), textcoords="offset points", 
                         xytext=(offset, 15), ha='left', va='center', fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", edgecolor="gray", alpha=0.8))
            
    plt.title("Geographic Locations of Simulated Ground Stations", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Longitude (Degrees)", fontsize=11)
    plt.ylabel("Latitude (Degrees)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "station_map.png"), dpi=150)
    plt.close()
    
    print("Dataset generation and plot creation completed successfully!")

if __name__ == "__main__":
    main()
