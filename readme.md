![Version](https://img.shields.io/badge/version-v1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

# Multi-Satellite Constillation Link Quality Simulator
High-fidelity satellite communication simulator combining
orbital propagation, atmospheric attenuation modeling, and
stochastic rain fading to evaluate  link performance across dynamic constellations.

Physics-first satellite link simulator integrating:
- **Constellation Management**: Support for multi-satellite systems with dynamic handoff logic.
- **SGP4 orbital propagation** (via `sgp4`)
- **ITU-R P.618/P.676/P.837/P.838** atmospheric models
- **Temporally correlated rain fading** (Maseng-Bakken AR(1))
- **XGBoost** link quality prediction

### Highlights
- **275k timesteps/sec** (single-satellite vectorized mode)
- **74k timesteps/sec** (full constellation + handoff mode)
- **JIT-Accelerated Physics**: 192x faster rain dynamics via Numba JIT.
- **1,300+ Satellite Database**: Integrated live CelesTrak TLE updates (OneWeb, Iridium, GEO).
- **Dynamic Handoffs**: State-aware switching based on highest elevation or SNR with hysteresis.
- **Validation suite**: Extensive quantitative reports for physics and network integrity.
- **Interactive Streamlit dashboard**: Multi-sat selection and real-time handoff visualization.

---

### Architecture

```mermaid
graph TD
    A[TLE Catalog / satellites.db] --> B[SGP4 Propagation]

    B --> C[Geometry Engine]
    C --> D[ITU-R Models]
    D --> E[Link Budget Engine]

    E --> F[Candidate Link Matrix]

    F --> G[Handoff Manager]

    G --> H[Selected Link Timeline]

    H --> I[Feature Extraction]
    I --> J[XGBoost Scoring]
    J --> K[Streamlit Dashboard]

    subgraph JIT_ACCELERATED [Performance Layer]
        R[Maseng-Bakken Rain Process]
        R -- Numba JIT --> D
    end
```

---

### Quick Start

```bash
# 1. Install the package in editable mode
pip install -e .

# 2. Update satellite database with live TLEs
satlinksim-update

# 3. Run the interactive Streamlit dashboard
satlinksim-ui

# 4. Run tests
python3 -m pytest

# 5. Run validation and benchmarks
python3 val_and_bench/validation_correctness.py
python3 val_and_bench/benchmarks.py
```

---

### Docker Usage

You can also run the simulator using Docker:

**Using Docker Build:**
```bash
# Build the image
docker build -t satlinksim .

# Run the container
docker run -p 8501:8501 satlinksim
```

**Using Docker Compose:**
```bash
# Start the service
docker-compose up
```

The dashboard will be available at `http://localhost:8501`.

---

### Key Features
The simulator computes a full high-fidelity link budget at each time step, tracking everything from geometric path loss and gaseous absorption to rapid tropospheric scintillation. It models dynamic LEO/MEO constellations by utilizing live TLE data and SGP4 propagation.

A dedicated **Handoff Manager** handles stateful satellite switching using configurable policies (Highest SNR or Elevation) with built-in hysteresis and minimum dwell-time constraints to prevent rapid connection toggling.

The simulation engine combines NumPy vectorization, Numba JIT compilation, async orbital propagation, and multiprocessing-based Monte Carlo execution to support large-scale availability studies while maintaining interactive performance.

---

### Validation & Benchmarks
- **FSPL accuracy**: <1e-4 dB
- **Throughput**: 275k/sec (Single-Sat) | 74k/sec (Constellation)
- **SGP4 latency**: 75µs
- **Memory**: 326MB @ 500k steps
- **Monte Carlo Speedup**: ~2.5x (12 workers)

---

### Repository Structure
The project follows a standard `src`-layout for Python packages:

```text
├── docs/                   # Detailed physics and architecture docs
├── src/
│   └── satlinksim/         # Core Python package
│       ├── app.py          # Streamlit Dashboard UI
│       ├── satellite_link_sim.py  # Vectorized Physics Engine
│       ├── propogate.py     # SGP4 & Constellation Management
│       ├── update_tle.py    # Live TLE Update Tool
│       └── ground_stations.py # Station Database
├── tests/                  # Unit, physics, and regression tests
├── real_world_validation/   # Comparison against external datasets
└── val_and_bench/          # Performance and internal consistency tools
```

---

### Documentation Links
- [Physics Models](docs/physics_models.md)
- [Rain Model (Maseng-Bakken)](docs/rain_model.md)
- [System Architecture](docs/architecture.md)
- [Validation Methodology](docs/validation.md)
- [Benchmark Results](docs/benchmarks.md)
- [References](docs/references.md)
