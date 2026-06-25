# SatLinkSim v1 Cloud-Native API Platform Documentation

This document explains the architecture, security, and reference endpoints for the SatLinkSim Cloud-Native API Platform (`/api/v1`). 

The v1 API transforms SatLinkSim from a simple command line wrapper into a robust, cloud-ready satellite simulation platform.

---

## 1. Core Architecture Principles

1. **Simulations as Persistent Resources**: Avoid duplicate computations by creating simulations once via `POST /api/v1/simulations` and querying sub-resources (`/results`, `/attenuation`, `/link-budget`, `/visibility`, `/availability`, `/handoffs`, `/orbit`) as needed.
2. **Stateless Calculators**: For interactive web widgets, stateless calculators (`/api/v1/calculators/...`) evaluate metrics on-the-fly without creating database/in-memory records.
3. **Asynchronous Batch Execution**: Long-running simulations, parameter sweeps, and Monte Carlo experiments are submitted as `/api/v1/jobs` and executed in the background.
4. **Unified Multi-Format Outputs**: Every tabular and timeseries endpoint supports selection of the output format (`json`, `csv`, or `parquet`) via the `format` query parameter.
5. **Pagination by Default**: Directory endpoints (`/satellites`, `/stations`, `/datasets`, `/simulations`) support structured cursor/page pagination to ensure high performance.

---

## 2. Authentication

All requests to the versioned `/api/v1/...` routes require **Bearer Token Authentication** in the HTTP request headers:

```http
Authorization: Bearer sk_test_token
```

* **API Key Format**: Any key starting with `sk_` (e.g. `sk_live_123456`, `sk_test_token`).
* **Errors**: Requests without an `Authorization` header, or with an invalid key format, return `401 Unauthorized` or `403 Forbidden`.
* **Legacy Root Routes**: Root-level legacy endpoints (like `/simulate`, `/link-budget`) do not require authentication to ensure absolute backward compatibility with the existing Streamlit UI and CLI.

---

## 3. Global Query Parameters

### 3.1 Format Selection
* **`format`** (string, default: `"json"`): Can be `json`, `csv`, or `parquet`.
  * `json`: Returns structured JSON payloads.
  * `csv`: Returns a text CSV file download attachment (`text/csv`).
  * `parquet`: Returns a binary Apache Parquet file download (`application/octet-stream`).

### 3.2 Pagination (Directories)
Applicable to `/satellites`, `/stations`, `/datasets`, and `/simulations`:
* **`page`** (integer, default: `1`): The page number to retrieve (1-indexed).
* **`limit`** (integer, default: `10`): Number of items per page.
* **`operator`** (string, optional): Filter by operator name.
* **`constellation`** (string, optional): Filter by constellation name (e.g., `Starlink`, `OneWeb`).

---

## 4. API Reference

### 4.1 Simulations Resource (`/api/v1/simulations`)

#### Create a Simulation
Starts a simulation run and returns its identifier.
* **Path**: `POST /api/v1/simulations`
* **Request Schema**: Accepts either `PublicSimulationRequest` (simplified) or `SimulationRequest` (full RF details).
* **Example JSON Request**:
  ```json
  {
    "satellites": ["GALAXY 16"],
    "ground_station": "Delhi",
    "frequency": 14e9,
    "duration": 600,
    "step": 60,
    "rain": true,
    "handoff": true
  }
  ```
* **Example JSON Response** (Status Code `201`):
  ```json
  {
    "id": "e2a225de-8c83-49fb-811c-99d8213bfa70",
    "status": "completed",
    "created_at": "2026-06-26T02:30:10Z",
    "request_type": "public"
  }
  ```

#### List Simulations
Returns a paginated list of created simulations.
* **Path**: `GET /api/v1/simulations`

#### Get Simulation Metadata
Retrieves configuration, creation epoch, and status.
* **Path**: `GET /api/v1/simulations/{id}`

#### Delete a Simulation
Cleans up the simulation resource from in-memory/persistent store.
* **Path**: `DELETE /api/v1/simulations/{id}`

#### Retrieve Outputs (`/results`)
Returns the complete timeseries array of simulated metrics.
* **Path**: `GET /api/v1/simulations/{id}/results`

#### Sub-Resource: Attenuation (`/attenuation`)
Extracts gaseous, rain, scintillation, and total attenuation timeseries.
* **Path**: `GET /api/v1/simulations/{id}/attenuation`
* **Response Output**:
  ```json
  {
    "gaseous_attenuation": [0.81, 0.82, 0.83],
    "rain_attenuation": [0.0, 1.25, 4.30],
    "scintillation_attenuation": [0.15, -0.02, 0.09],
    "total_attenuation": [0.96, 2.05, 5.22]
  }
  ```

#### Sub-Resource: Link Budget (`/link-budget`)
Extracts RF parameters like EIRP, path loss, received power, noise floor, and SNR.
* **Path**: `GET /api/v1/simulations/{id}/link-budget`

#### Sub-Resource: Visibility (`/visibility`)
Retrieves elevation, azimuth, and visibility booleans for all constellation satellites.
* **Path**: `GET /api/v1/simulations/{id}/visibility?min_elevation=10.0`

#### Sub-Resource: Availability (`/availability`)
Computes availability fraction, total outages, and detailed outage start/end timestamps.
* **Path**: `GET /api/v1/simulations/{id}/availability?snr_threshold=5.0`

#### Sub-Resource: Handoffs (`/handoffs`)
Retrieves the array of handoff events during the simulation.
* **Path**: `GET /api/v1/simulations/{id}/handoffs`

#### Sub-Resource: Orbit (`/orbit`)
Retrieves orbital coordinate tracking states (elevation, slant range, radial velocity, azimuth).
* **Path**: `GET /api/v1/simulations/{id}/orbit`

---

### 4.2 Batch Jobs Resource (`/api/v1/jobs`)

For asynchronous runs, parameter sweeps, and multi-station simulations:
* **Create Job**: `POST /api/v1/jobs` (takes `BatchSimulationRequest`, runs in background).
* **Check Status**: `GET /api/v1/jobs/{id}` (returns `"pending"`, `"running"`, `"completed"`, or `"failed"`).
* **Get Results**: `GET /api/v1/jobs/{id}/results` (returns batch dictionary or tabular timeseries).

---

### 4.3 Stateless Calculators (`/api/v1/calculators`)

Stateless APIs designed for real-time web utilities. They accept parameters, run calculations on-the-fly, and return formatted data directly without caching.
* **Link Budget Calculator**: `POST /api/v1/calculators/link-budget`
* **Atmospheric Attenuation Calculator**: `POST /api/v1/calculators/attenuation`
* **Availability Calculator**: `POST /api/v1/calculators/availability`

---

## 5. Rain Services (`/api/v1/rain`)

Exposes physics-guided Narrowcasting algorithms and Maseng-Bakken stochastic forecasting.
* **Stage A Predictor (Telemetry Inversion)**: `POST /api/v1/rain/predict`
  * Inverts elevation, SNR, slant range, and frequency to predict rain rate.
* **Ensemble Forecast**: `POST /api/v1/rain/forecast`
  * Projects forward ensemble realizations using Maseng-Bakken AR(1) process parameters.

---

## 6. Coverage (`/api/v1/coverage`)

* **Local Station Coverage**: `POST /api/v1/coverage`
  * Evaluates duration covered by a constellation for specific ground stations.
* **Global Grid Coverage Map**: `POST /api/v1/coverage/global`
  * Calculates coverage fraction across a global latitude-longitude coordinates grid (e.g. `grid_size=30` degrees) for premium dashboard visualizations.

---

## 7. Orbit Propagation (`/api/v1/orbit`)

* **Live Geodetic Coordinate Lookup**: `GET /api/v1/orbit/{satellite}`
  * Searches the SQLite TLE database for the satellite name or NORAD ID, propagates to the **current exact epoch**, and returns WGS84 Geodetic Latitude, Longitude, Altitude, ECEF velocity vectors, and TLE lines.
* **Batch Propagator**: `POST /api/v1/orbit/propagate` (propagates coordinates over a defined duration).

---

## 8. Directories (`/api/v1/stations`, `/api/v1/satellites`)

* **GET `/api/v1/stations`**: Returns ground station metadata database (paginated, supports `operator` filter).
* **GET `/api/v1/satellites`**: Returns tracked satellites database (paginated, supports `constellation` filter).

---

## 9. Scientific Validation (`/api/v1/validation`)

Exposes scientific correctness and regression assertions directly:
* **All Validations**: `GET /api/v1/validation`
* **Category Physics**: `GET /api/v1/validation/physics` (FSPL, Slant Range)
* **Category ITU**: `GET /api/v1/validation/itu` (ITU-R P.839 height validations)
* **Category NASA**: `GET /api/v1/validation/nasa` (NASA GPM rain statistics correlation validations)
* **Category Regression**: `GET /api/v1/validation/regression`

---

## 10. Benchmarks (`/api/v1/benchmarks`)

Exposes simulator performance metrics:
* **GET `/api/v1/benchmarks`**:
  * Returns `throughput_timesteps_per_second`, `memory_rss_mb`, `cpu_utilization_pct`, `numba_acceleration_ratio`, and `monte_carlo_speedup_factor`.

---

## 11. Health (`/api/v1/health`)

* **GET `/api/v1/health`** (Public Endpoint):
  * Returns: `{"status": "healthy", "timestamp": "...", "database": "connected"}`.
