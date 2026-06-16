# System Architecture

The simulator is designed for scalable simulation workloads through vectorization, concurrency, and parallel execution. It operates as a high-fidelity time-series engine, bridging the gap between orbital mechanics and link-layer performance.

## Architecture Diagram

```mermaid
graph TD
    A[TLE Catalog / satellites.db] --> B[SGP4 Propagation]

    B --> C[Geometry Engine]
    C --> D[ITU-R Models]
    D --> E[Link Budget Engine]

    subgraph Service Layer [FastAPI Layer]
        F[Simulation REST API]
        F --> G[Simulation Engine]
    end

    G --> H[Candidate Link Matrix]
    H --> I[Handoff Manager]
    I --> J[Selected Link Timeline]

    J --> K[Feature Extraction]
    K --> L[XGBoost Scoring]
    
    M[CLI / REST Client] --> F
    N[Streamlit / REST Client] --> F

    subgraph JIT_ACCELERATED [Performance Layer]
        R[Maseng-Bakken Rain Process]
        R -- Numba JIT --> D
    end
```

## Core Modules (Clean Architecture)
The codebase follows a Clean Architecture layout to strictly separate domain physics from application orchestration and external infrastructure:
- **`src/satlinksim/domain/`**: Contains pure physical models, link budget math, geometric transformations, and core data structures (`models.py`). This layer has no external dependencies.
- **`src/satlinksim/application/`**: Contains the `SimulationEngine`, orchestrating domain logic to fulfill batched, concurrent, and Monte Carlo simulation requests.
- **`src/satlinksim/infrastructure/api/`**: **(New)** The RESTful service layer.
    - `server.py`: FastAPI server exposing the simulation engine.
    - `schemas.py`: Pydantic V2 models for JSON serialization and validation.
    - `client.py`: High-level Python client for consuming the REST API.
- **`src/satlinksim/infrastructure/`**: Handles other external integrations, including the Streamlit dashboard (`ui/`), SGP4 TLE propagation (`tle/`), database persistence (`persistence/`), and ML model training/inference (`ml/`).

## Service Architecture (REST API)
The simulator has been transformed into a service-oriented platform, allowing for distributed execution and integration with external tools:

1. **FastAPI Server**: A high-performance REST server that exposes a `/simulate` endpoint. It accepts structured JSON requests (stations, constellations, handoff policies) and returns high-fidelity time-series results.
2. **Unified Entry Point**: The `satlinksim` CLI provides a single command to manage the entire lifecycle:
    - `satlinksim api`: Start the backend service.
    - `satlinksim ui`: Launch the interactive dashboard (supports both local and remote modes).
    - `satlinksim simulate`: Execute a headless simulation via the REST API.
3. **Data Integrity**: All API communication is strictly validated using Pydantic V2 schemas, ensuring that physical parameters (frequencies, coordinates) are correct before reaching the simulation core.
4. **Remote Execution Mode**: The Streamlit UI can act as a thin client to the API server, enabling simulations to be offloaded to dedicated hardware.

## High-Performance Engineering
The simulator utilizes a multi-tier optimization strategy to handle large-scale satellite constellations:
- **Numba JIT Compilation**: The Maseng-Bakken AR(1) rain synthesis engine is JIT-compiled into a machine-code kernel, achieving a **~192x speedup** over interpreted Python.
- **NumPy Vectorization**: Link budget and atmospheric calculations are processed as matrix operations, significantly reducing Python overhead.
- **Async Concurrency**: Asyncio coordinates concurrent propagation workflows for multi-station simulations.
- **Multiprocessing**: Monte Carlo iterations are distributed across CPU cores using `ProcessPoolExecutor`.

*Note: Performance figures are derived from the [Benchmark Suite](benchmarks.md).*

## The Timestep Simulation Loop
The core of the simulator is a stateful time-series engine. For each timestep in the simulation window:

1. **Propagate**: Update candidate satellite ECEF states using SGP4 orbital kernels.
2. **Handoff**: Evaluate switching logic (Hysteresis/Dwell) to select the optimal active satellite.
3. **Geometry**: Recompute range, elevation, and Doppler shift for the selected satellite.
4. **Atmospheric Models**: Evaluate frequency- and angle-dependent losses (FSPL, Gas, Scintillation).
5. **Rain Dynamics**: Advance the Maseng-Bakken correlated rain process (AR(1)) using a **JIT-accelerated kernel**.
6. **Link Budget**: Consolidated SNR calculation including hardware gains and noise floor.
7. **Aggregate**: Collect time-series metrics (SNR, packet loss) for final scoring.

## Simulation Workflow
1. **Initialize**: Load satellite TLEs and ground station parameters from SQLite.
2. **Parallel Execution**: Distribute independent simulation workloads across available CPU cores for large-scale availability studies.
3. **Matrix Processing**: Apply vectorized atmospheric models across the entire time series.
4. **ML Inference**: Extract features and score the station using the pre-trained XGBoost model.
5. **Dashboard Rendering**: Present interactive charts and comparative rankings in the UI.
