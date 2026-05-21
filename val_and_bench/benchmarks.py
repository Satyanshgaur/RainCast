import time
import sys
import os
import psutil
import statistics
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timezone

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from satellite_link_sim import simulate_station, simulate_all
from ground_stations import GROUND_STATIONS
from propogate import Propagator

def benchmark_throughput():
    print("--- Simulation Throughput Benchmark ---")
    step_counts = [10, 100, 1000, 5000, 10000]
    throughputs = []
    
    gs = GROUND_STATIONS[0] # Use Delhi as representative
    
    for n_steps in step_counts:
        start_time = time.perf_counter()
        simulate_station(gs, n_steps=n_steps, dt_s=1.0)
        end_time = time.perf_counter()
        
        duration = end_time - start_time
        ts_per_sec = n_steps / duration
        throughputs.append(ts_per_sec)
        print(f"Steps: {n_steps:5d} | Time: {duration:8.4f}s | Throughput: {ts_per_sec:10.2f} timesteps/sec")
    
    return step_counts, throughputs

def benchmark_scaling():
    print("\n--- Runtime Scaling Benchmark ---")
    station_counts = [1, 2, 4, 8, 16, 32]
    runtimes = []
    n_steps = 100
    
    # Base stations
    base_stations = GROUND_STATIONS
    
    for count in station_counts:
        # Create a list of 'count' stations by cycling through base stations
        test_stations = [base_stations[i % len(base_stations)].copy() for i in range(count)]
        # Ensure unique names if needed, though simulate_all doesn't care
        for i, s in enumerate(test_stations):
            s['name'] = f"{s['name']}_{i}"
            
        start_time = time.perf_counter()
        # Mocking simulate_all for custom station list
        [simulate_station(gs, n_steps=n_steps) for gs in test_stations]
        end_time = time.perf_counter()
        
        duration = end_time - start_time
        runtimes.append(duration)
        print(f"Stations: {count:3d} | Total Time: {duration:8.4f}s | Avg/Station: {duration/count:8.4f}s")
        
    return station_counts, runtimes

def benchmark_memory():
    print("\n--- Memory Usage Benchmark ---")
    process = psutil.Process(os.getpid())
    
    # Baseline
    mem_baseline = process.memory_info().rss / 1024 / 1024 # MB
    
    n_steps = 500000 # Large simulation
    gs = GROUND_STATIONS[0]
    
    start_mem = process.memory_info().rss / 1024 / 1024
    # Keep result in a list to prevent immediate GC
    results = []
    results.append(simulate_station(gs, n_steps=n_steps))
    end_mem = process.memory_info().rss / 1024 / 1024
    
    print(f"Large Sim (n_steps={n_steps})")
    print(f"Baseline Memory: {mem_baseline:8.2f} MB")
    print(f"Pre-sim Memory:  {start_mem:8.2f} MB")
    print(f"Post-sim Memory: {end_mem:8.2f} MB")
    print(f"Delta Memory:    {end_mem - start_mem:8.2f} MB")
    
    # Explicitly clear and collect to show memory release
    del results
    import gc
    gc.collect()
    final_mem = process.memory_info().rss / 1024 / 1024
    print(f"Post-GC Memory:  {final_mem:8.2f} MB")
    
    return n_steps, (start_mem, end_mem)

def benchmark_profiling():
    print("\n--- Performance Profiling ---")
    gs = GROUND_STATIONS[0]
    n_steps = 1000
    process = psutil.Process(os.getpid())
    
    # Profile propagation latency specifically
    prop = Propagator()
    sat_id = gs.get("norad_id")
    curr_time = datetime.now(timezone.utc)
    lat, lon, alt = gs["latitude"], gs["longitude"], gs["altitude_km"]
    
    prop_times = []
    for _ in range(100):
        t0 = time.perf_counter()
        prop.get_geometry(sat_id, curr_time, lat, lon, alt)
        prop_times.append(time.perf_counter() - t0)
    
    avg_prop = statistics.mean(prop_times) * 1000 # ms
    print(f"Avg Propagation Latency (get_geometry): {avg_prop:.4f} ms")
    
    # Profile whole simulation loop breakdown (approximated)
    start_time = time.perf_counter()
    simulate_station(gs, n_steps=n_steps)
    total_duration = time.perf_counter() - start_time
    
    print(f"Total time for {n_steps} steps: {total_duration:.4f}s")
    print(f"Avg time per step: {(total_duration/n_steps)*1000:.4f} ms")
    
    # CPU Utilization
    print("Measuring process CPU utilization over 2 seconds of heavy simulation...")
    # Initial call to cpu_percent to start measurement
    process.cpu_percent()
    stop_at = time.time() + 2
    count = 0
    while time.time() < stop_at:
        simulate_station(gs, n_steps=10000)
        count += 1
    
    avg_cpu = process.cpu_percent()
    print(f"Process CPU Utilization: {avg_cpu:.1f}%")
    
    return avg_prop, avg_cpu

def benchmark_memory_scaling():
    print("\n--- Memory Scaling Benchmark ---")
    step_counts = [1000, 10000, 50000, 100000, 200000]
    memory_deltas = []
    gs = GROUND_STATIONS[0]
    process = psutil.Process(os.getpid())
    
    for n_steps in step_counts:
        import gc
        gc.collect()
        start_mem = process.memory_info().rss / 1024 / 1024
        res = simulate_station(gs, n_steps=n_steps)
        end_mem = process.memory_info().rss / 1024 / 1024
        memory_deltas.append(max(0, end_mem - start_mem))
        del res
        print(f"Steps: {n_steps:6d} | Memory Delta: {memory_deltas[-1]:8.2f} MB")
        
    return step_counts, memory_deltas

def generate_plots(throughput_data, scaling_data, memory_data):
    print("\n--- Generating Benchmark Plots ---")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Throughput Plot
    step_counts, throughputs = throughput_data
    plt.figure(figsize=(10, 6))
    plt.plot(step_counts, throughputs, 'o-', linewidth=2)
    plt.xscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.title("Simulation Throughput vs. Step Count")
    plt.xlabel("Number of Timesteps (Log Scale)")
    plt.ylabel("Throughput (timesteps/sec)")
    plt.savefig(os.path.join(base_dir, "bench_throughput.png"))
    
    # 2. Scaling Plot
    station_counts, runtimes = scaling_data
    plt.figure(figsize=(10, 6))
    plt.plot(station_counts, runtimes, 's-', color='orange', linewidth=2)
    plt.grid(True, alpha=0.5)
    plt.title("Runtime Scaling vs. Number of Stations")
    plt.xlabel("Number of Ground Stations")
    plt.ylabel("Total Execution Time (s)")
    plt.savefig(os.path.join(base_dir, "bench_scaling.png"))

    # 3. Memory Plot
    mem_steps, mem_deltas = memory_data
    plt.figure(figsize=(10, 6))
    plt.plot(mem_steps, mem_deltas, 'd-', color='green', linewidth=2)
    plt.grid(True, alpha=0.5)
    plt.title("Memory Usage vs. Step Count")
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Memory Delta (MB)")
    plt.savefig(os.path.join(base_dir, "bench_memory.png"))
    
    print(f"Plots saved to {base_dir}")

if __name__ == "__main__":
    t_data = benchmark_throughput()
    s_data = benchmark_scaling()
    benchmark_memory()
    m_data = benchmark_memory_scaling()
    benchmark_profiling()
    generate_plots(t_data, s_data, m_data)
    print("\nBenchmarks Completed Successfully.")
