# Benchmark Results

Performance is measured across three primary dimensions: throughput, scaling, and memory efficiency. The engine is optimized for high-frequency trading and real-time link monitoring applications.

## 1. Simulation Throughput
The vectorized NumPy engine achieves approximately **60,000 timesteps/sec** for sequential execution. This high throughput is achieved by minimizing Python-level loops and maximizing cache locality through contiguous memory access.

![Throughput Benchmark](../val_and_bench/bench_throughput.png)
*Figure 1: Simulation throughput as a function of the total number of timesteps. Performance stabilizes for windows exceeding 1,000 steps.*

## 2. Parallel Scaling Analysis
The system demonstrates strong speedup for Monte Carlo iterations by distributing independent rain realizations across multiple CPU cores.
- **Speedup**: ~2.5x speedup for typical workloads using multiprocessing with 12 workers.
- **Efficiency**: Peak efficiency of ~60% observed at low worker counts, with performance plateaus determined by IPC overhead for large result matrices.

![Parallel Scaling](../val_and_bench/bench_parallel_scaling.png)
*Figure 2: Execution time reduction and speedup factor for Monte Carlo iterations. Note the near-linear scaling for small worker pools.*

![Worker Efficiency](../val_and_bench/bench_worker_efficiency.png)
*Figure 3: Efficiency factor per worker. Maintaining high granularity in the simulation window ensures optimal utilization of multi-core hardware.*

## 3. Propagation Latency
Individual SGP4 propagation calls average **18 microseconds (µs)** per step. The use of the optimized SGP4 C++ kernel (via Python bindings) allows for rapid geometry updates even in large-constellation scenarios.

## 4. Memory Efficiency
The simulator is designed for a low memory footprint. A 500,000-step simulation (approx. 1 year of 1-minute data for one station) consumes only **122 MB** of RAM. All simulation data is stored in compact NumPy arrays, ensuring efficient memory management and rapid garbage collection.

![Memory Benchmark](../val_and_bench/bench_memory.png)
*Figure 4: Memory usage delta vs. simulation step count. The linear relationship confirms predictable resource consumption for long-duration studies.*
