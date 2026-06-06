# Benchmark Results

The benchmark suite evaluates throughput, parallel scalability,
and memory efficiency of the simulation engine under realistic
satellite communication workloads.

## Benchmark Environment

- CPU: Intel i5 13420H
- RAM: 16 GB DDR4
- OS: Ubuntu 24.04 
- Python: 3.12
- NumPy: 1.24.4

## Methodology

Unless otherwise stated:

- Benchmarks were executed 10 times and averaged.
- CPU frequency scaling remained enabled.
- Measurements were collected using `time.perf_counter()`.
- Memory measurements were obtained using `psutil`.
- Results represent wall-clock execution time.

## 1. Performance Validation

### 1.1 Simulation Throughput
The vectorized NumPy engine achieves approximately **275,000 timesteps/sec** for sequential execution (single satellite). This high throughput is achieved by minimizing Python-level loops and maximizing cache locality through contiguous memory access.

![Throughput Benchmark](../val_and_bench/bench_throughput.png)
*Figure 1: Simulation throughput as a function of the total number of timesteps. Performance stabilizes for windows exceeding 1,000 steps.*

### 1.2 Parallel Scaling
The system demonstrates strong speedup for Monte Carlo iterations by distributing independent rain realizations across multiple CPU cores.

| Workers | Speedup | Efficiency |
| ------- | ------- | ---------- |
| 1       | 1.0     | 100%       |
| 2       | 1.6     | 80%        |
| 4       | 2.4     | 59%        |
| 8       | 3.0     | 38%        |
| 12      | 3.4     | 28%        |


### 1.3 Runtime Breakdown
Profiling of the full constellation/handoff pipeline (14 GHz, 10,000 steps, 4 stations) reveals the following computational distribution:

| Component       | Runtime Share |
| --------------- | ------------- |
| NumPy Overhead  | 34.7%         |
| SGP4 / Geometry | 24.1%         |
| Data & Results  | 12.4%         |
| Handoff Logic   | 11.0%         |
| Sim Control     | 7.8%          |
| Link Budget     | 1.8%          |
| Rain Process    | 1.8%          |
| Misc Other      | 6.4%          |

*Note: After Numba JIT optimization, the Rain Process is no longer a significant bottleneck.*

### 1.4 Runtime Optimization History

A profiling-driven optimization effort was conducted to identify and eliminate simulation bottlenecks.

| Component | Before (Python) | After (Numba JIT) |
|------------|---------|---------|
| Rain Process | 50.6% | 0.7% |
| Runtime | 96.1 ms | 0.5 ms |
| Speedup | 1× | ~192× |

The rain synthesis engine was migrated from an interpreted Python implementation to a Numba JIT-compiled kernel. The optimization reduced rain generation from the dominant runtime component to a negligible fraction of total execution time while preserving numerical behavior.

For a detailed analysis, see the [Complete Profiling Report](../val_and_bench/profile_report.md).

### 1.5 Memory Scaling
The simulator is designed for a low memory footprint. A 500,000-step simulation (approx. 1 year of 1-minute data for one station) consumes **326 MB** of RAM.

![Memory Benchmark](../val_and_bench/bench_memory.png)
*Figure 2: Memory usage delta vs. simulation step count.*

---

## 2. System Architecture Evolution
The simulator has evolved from a simple scalar model to a highly complex, stateful constellation manager. Despite the increased complexity, the system maintains high performance through aggressive vectorization.

| Version | Throughput | Description |
| ------------- | ---------- | ----------- |
| **Scalar** | ~6k/s | Legacy baseline (pre-vectorization) |
| **Vectorized** | ~275k/s | Current NumPy-optimized core (single sat) |
| **Constellation**| ~74k/s | Vectorized + Dynamic Multi-Sat Geometry (1k+ satellites) |
| **Handoff** | ~73k/s | Full pipeline + Stateful Switch Policies |

### Performance Analysis
- **Vectorization Gain**: Transitioning from scalar loops to NumPy operations provided a **~45x performance boost**.
- **Constellation Overhead**: Introducing dynamic multi-satellite propagation (SGP4) for every station significantly increases CPU load per timestep. However, even with a database of **1,335 satellites**, the system still achieves **74,000 steps/sec**, which is equivalent to simulating a full year of 1-minute data for one station in less than 8 seconds.
- **Handoff Stability**: The Handoff Manager (Hysteresis, Dwell Time) introduces negligible overhead (~1.6%) while providing realistic connection stability and preventing "ping-pong" switching between satellites.


