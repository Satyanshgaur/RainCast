
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, "src"))

from datetime import datetime, timezone, timedelta

from satlinksim.propogate import Propagator, Satellite

# --- Observation Data from SatNOGS ---
# Observation ID: 14217654
# Satellite: CONNECTA IOT-1 (60472)
OBSERVATION = {
    "id": 14217654,
    "norad_id": 60472,
    "start": "2026-06-02T16:06:08Z",
    "end": "2026-06-02T16:11:00Z",
    "gs_lat": 33.797628,
    "gs_lon": -79.159123,
    "gs_alt": 12 / 1000.0, # km
    "tle1": "1 60472U 24149E   26153.18620229  .00004418  00000-0  18856-3 0  9994",
    "tle2": "2 60472  97.3762 229.7318 0001676  25.2916 334.8403 15.23416626 99517",
    "max_el_expected": 79.0
}

def main():
    print(f"Validating SGP4 Propagation against SatNOGS Observation {OBSERVATION['id']}...")
    
    # Setup Propagator
    # We use a dummy DB path or just pass the Satellite object
    prop = Propagator(db_path=":memory:") 
    sat = Satellite(
        norad_id=OBSERVATION["norad_id"],
        name="CONNECTA IOT-1",
        tle_line1=OBSERVATION["tle1"],
        tle_line2=OBSERVATION["tle2"]
    )
    
    # Times during the observation
    start_dt = datetime.fromisoformat(OBSERVATION["start"].replace('Z', '+00:00'))
    end_dt = datetime.fromisoformat(OBSERVATION["end"].replace('Z', '+00:00'))
    
    dts = []
    curr = start_dt
    while curr <= end_dt:
        dts.append(curr)
        curr += timedelta(seconds=10)
    
    # 1. Current Implementation (Broken)
    print("Calculating geometry using current Propagator...")
    geo = prop.get_geometry_batch(sat, dts, OBSERVATION["gs_lat"], OBSERVATION["gs_lon"], OBSERVATION["gs_alt"])
    
    if geo is None:
        print("Error: Propagation failed.")
        return

    max_el_calc = np.max(geo.elevation_deg)
    print(f"Max Elevation (Current Code): {max_el_calc:.2f}°")
    print(f"Max Elevation (SatNOGS Expected): {OBSERVATION['max_el_expected']}°")
    
    # 2. Check for Frame Mismatch
    # If the code is broken, the elevation might even be negative (satellite 'below' ground)
    # even though it was observed.
    
    is_visible = np.any(geo.elevation_deg > 0)
    print(f"Is satellite visible during observation window? {is_visible}")
    
    # Plotting
    plt.figure(figsize=(10, 5))
    times_rel = [(dt - start_dt).total_seconds() / 60.0 for dt in dts]
    
    plt.plot(times_rel, geo.elevation_deg, 'r-', label='Current Implementation (Broken)')
    plt.axhline(0, color='black', linestyle='--')
    plt.axhline(OBSERVATION['max_el_expected'], color='green', linestyle=':', label='SatNOGS Reported Peak')
    
    plt.xlabel('Minutes since start')
    plt.ylabel('Elevation (degrees)')
    plt.title(f'SGP4 Validation: SatNOGS Obs {OBSERVATION["id"]} (NORAD {OBSERVATION["norad_id"]})')
    plt.legend()
    plt.grid(True)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    out_plot = os.path.join(plots_dir, "val_sgp4_satnogs.png")
    plt.savefig(out_plot)
    print(f"Plot saved to {out_plot}")

    if not is_visible or abs(max_el_calc - OBSERVATION['max_el_expected']) > 10:
        print("\n[!] VALIDATION FAILED")
        print("The predicted elevation deviates significantly from reality.")
        print("Root cause: SGP4 returns TEME coordinates, but the code treats them as ECEF.")
    else:
        print("\n[+] VALIDATION PASSED (Unexpectedly?)")

if __name__ == "__main__":
    main()
