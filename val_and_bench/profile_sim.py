import cProfile
import pstats
import sys
import os
import io
from datetime import datetime, timezone

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from satellite_link_sim import simulate_all_batched, GROUND_STATIONS

def run_profile():
    # Large n_steps to get meaningful profiling data
    n_steps = 10000
    
    # Warm-up run to trigger Numba JIT compilation
    print("Warming up Numba JIT kernel...")
    simulate_all_batched(GROUND_STATIONS, n_steps=100)
    
    print(f"Running profile for {len(GROUND_STATIONS)} stations over {n_steps} steps...")
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run the simulation
    simulate_all_batched(GROUND_STATIONS, n_steps=n_steps)
    
    profiler.disable()
    
    # Save to profile.out
    profiler.dump_stats("profile.out")
    
    # Capture pstats output
    s = io.StringIO()
    ps = pstats.Stats("profile.out", stream=s).sort_stats("cumtime")
    ps.print_stats()
    
    # Categorize results
    # We'll map function names to categories
    category_map = {
        "Rain Process": ["step", "rain_attenuation_db", "itu_rain_height", "_simulate_rain_kernel"],
        "SGP4 & Prop": ["get_geometry_batch", "sgp4_array", "get_gmst", "rotate_teme_to_ecef", "jday", "_sgp4", "geodetic_to_ecef", "ecef_to_enu_matrix"],
        "Handoff Logic": ["select", "HandoffManager"],
        "Link Budget": ["fspl_db", "gaseous_absorption_db", "scintillation_sigma_db", "effective_path_length", "noise_power_dbw", "packet_loss_from_snr"],
        "NumPy Overhead": ["argmax", "where", "normal", "rand", "implement_array_function", "_wrapfunc", "std", "mean", "asanyarray", "fromnumeric", "numpy"],
        "Data & Results": ["StationResult", "append", "tolist", "dataclasses", "timedelta", "datetime", "copy", "<listcomp>", "<genexpr>"],
        "Sim Control": ["simulate_all_batched", "run_profile"]
    }
    
    # Get all stats
    # ps.stats is a dict: (file, line, name) -> (cc, nc, tt, ct, callers)
    # tt = total time (in this function)
    # ct = cumulative time (in this function and all subcalls)
    
    category_totals = {cat: 0.0 for cat in category_map}
    total_measured_time = 0.0
    
    for (file_path, line, func_name), (cc, nc, tt, ct, callers) in ps.stats.items():
        total_measured_time += tt
        matched = False
        for cat, keywords in category_map.items():
            if any(k in func_name for k in keywords) or any(k in file_path for k in keywords):
                category_totals[cat] += tt
                matched = True
                break
    
    # Create the report
    with open("profile_report.md", "w") as f:
        f.write("# Simulation Profile Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Steps: {n_steps}\n")
        f.write(f"Stations: {len(GROUND_STATIONS)}\n\n")
        
        f.write("## Component Runtime Breakdown\n\n")
        f.write("| Component    | Runtime | Time (s) |\n")
        f.write("| ------------ | ------- | -------- |\n")
        
        # Sort by runtime percentage
        sorted_cats = sorted(category_totals.items(), key=lambda x: x[1], reverse=True)
        
        for cat, time_s in sorted_cats:
            percentage = (time_s / total_measured_time) * 100 if total_measured_time > 0 else 0
            f.write(f"| {cat:<14} | {percentage:6.1f}% | {time_s:8.4f}s |\n")
            
        # Add "Other"
        categorized_time = sum(category_totals.values())
        other_time = total_measured_time - categorized_time
        other_percentage = (other_time / total_measured_time) * 100 if total_measured_time > 0 else 0
        if other_time > 0.0001:
            f.write(f"| {'Misc Other':<14} | {other_percentage:6.1f}% | {other_time:8.4f}s |\n\n")
        else:
            f.write("\n")
        
        f.write("## Top 50 Detailed Stats\n\n")
        f.write("```\n")
        # Reuse pstats to print top 50
        ps.sort_stats("cumtime").print_stats(50)
        f.write(s.getvalue().split("   ncalls  tottime  percall  cumtime  percall filename:lineno(function)")[-1])
        f.write("```\n")

    # Generate a plot
    try:
        import matplotlib.pyplot as plt
        
        labels = [c for c, _ in sorted_cats] + ["Other"]
        times = [t for _, t in sorted_cats] + [other_time]
        percentages = [(t / total_measured_time) * 100 for t in times]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, percentages, color=['#3498db', '#e74c3c', '#2ecc71', '#f1c40f', '#95a5a6'])
        plt.ylabel('Runtime Percentage (%)')
        plt.title(f'Simulation Component Breakdown ({n_steps} steps)')
        plt.ylim(0, 100)
        
        # Add labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                     f'{height:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig("profile_plots/component_breakdown.png")
        print("Plot saved to profile_plots/component_breakdown.png")
    except ImportError:
        print("Matplotlib not found, skipping plot generation.")

    print("Profile completed. Report saved to profile_report.md")

if __name__ == "__main__":
    # Ensure we are in the val_and_bench directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_profile()
