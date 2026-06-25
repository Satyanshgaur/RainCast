# SatLinkSim v1 Cloud-Native API Platform Specification

Welcome to the official developer documentation for the SatLinkSim Cloud-Native API Platform. This specification defines version `v1` of our resource-oriented REST and WebSocket endpoints.

> [!NOTE]
> All versioned endpoints require Bearer Token Authentication. Root-level legacy routes (e.g. `/simulate`, `/visibility`) continue to operate without authentication to ensure full backward compatibility with the existing Streamlit UI and CLI tools.

---

## 1. Global Platform Configuration

### 1.1 Base URL
All API requests must be directed to the following versioned base URL:
```http
https://api.satlinksim.com/api/v1
```

### 1.2 Authentication & Scopes
Authenticate your requests by including your API key in the `Authorization` header as a Bearer token:
```http
Authorization: Bearer sk_live_51N2x...
```

The platform supports fine-grained authorization scopes:
| Scope Name | Description |
| :--- | :--- |
| `simulation.read` | Allows retrieving simulation metadata and raw output results. |
| `simulation.write` | Allows creating, triggering, and deleting simulation resources. |
| `datasets.read` | Allows listing and downloading simulation training datasets. |
| `datasets.write` | Allows uploading custom datasets. |
| `admin` | Full administrative access, including TLE updates and hardware validation. |

### 1.3 Rate Limits
Rate limits are enforced based on account tiers. Clients exceeding limits will receive a `429 Too Many Requests` response.
| Plan Tier | Hourly Request Limit | Max Concurrent Simulations | Max Simulation Duration | Max Satellites |
| :--- | :--- | :--- | :--- | :--- |
| **Free** | 100 requests/hr | 5 | 24 hours | 100 |
| **Developer** | 2,000 requests/hr | 20 | 168 hours | 500 |
| **Enterprise** | Unlimited | Custom | Unlimited | Unlimited |

---

## 2. Standard Error Schema

All error responses return a standardized JSON payload to facilitate client-side SDK generation and error handling.

### 2.1 Common HTTP Status Codes
* **`400 Bad Request`**: Malformed payload or invalid query parameters.
* **`401 Unauthorized`**: Missing or malformed Bearer Token.
* **`403 Forbidden`**: Valid token, but lacks required scope.
* **`404 Not Found`**: Target resource does not exist.
* **`422 Unprocessable Entity`**: Request payload failed semantic schema validation.
* **`429 Too Many Requests`**: Rate limit exceeded.
* **`500 Internal Server Error`**: Core simulator computation crash.

### 2.2 Error Payload Example
```json
{
  "error": "InvalidFrequency",
  "message": "Frequency must be between 1 GHz and 100 GHz."
}
```

---

## 3. API Reference

### 3.1 Simulations Resource (`/simulations`)

Simulation execution is asynchronous to accommodate long-running parameter sweeps, Monte Carlo runs, and high-fidelity orbits.

#### Create a Simulation
* **Endpoint**: `POST https://api.satlinksim.com/api/v1/simulations`
* **Headers**: 
  * `Authorization: Bearer sk_test_token`
  * `Idempotency-Key: <UUID>` (Optional - prevents duplicate runs on retries)
* **Example Request Payload**:
  ```json
  {
    "satellites": ["GALAXY 16"],
    "ground_station": "Delhi",
    "frequency": 14e9,
    "duration": 86400,
    "step": 60,
    "rain": true,
    "handoff": true
  }
  ```
* **Example Response Payload** (`202 Accepted`):
  ```json
  {
    "id": "e2a225de-8c83-49fb-811c-99d8213bfa70",
    "status": "pending",
    "created_at": "2026-06-26T02:30:10Z",
    "version": "1.0.0"
  }
  ```

#### Get Simulation Status & Metadata
* **Endpoint**: `GET https://api.satlinksim.com/api/v1/simulations/{id}`
* **Response Payload** (`200 OK`):
  ```json
  {
    "id": "e2a225de-8c83-49fb-811c-99d8213bfa70",
    "status": "completed",
    "created_at": "2026-06-26T02:30:10Z",
    "finished_at": "2026-06-26T02:30:12Z",
    "compute_time": 2.15,
    "duration": 86400,
    "timesteps": 1440,
    "num_satellites": 1,
    "version": "1.0.0",
    "request_type": "public",
    "request_data": {
      "satellites": ["GALAXY 16"],
      "ground_station": "Delhi",
      "frequency": 14e9,
      "duration": 86400,
      "step": 60,
      "rain": true,
      "handoff": true
    }
  }
  ```
  *(Status transitions: `pending` ➔ `running` ➔ `completed` or `failed`)*

#### Retrieve Output Timeseries Results
* **Endpoint**: `GET https://api.satlinksim.com/api/v1/simulations/{id}/results?format=json`
* **Response Payload**: Standard simulation output timeseries object (SNR, Availability, Handoffs, Rain Loss).

#### Download Simulation Data
Explicit endpoints for direct raw data extraction in CSV or Parquet format:
* **JSON/CSV/Parquet Auto-negotiation**: `GET https://api.satlinksim.com/api/v1/simulations/{id}/download?format=parquet`
* **CSV Explicit File**: `GET https://api.satlinksim.com/api/v1/simulations/{id}/download.csv`
* **Parquet Explicit File**: `GET https://api.satlinksim.com/api/v1/simulations/{id}/download.parquet`

#### Sub-Resource Query Paths
* **Attenuation**: `GET https://api.satlinksim.com/api/v1/simulations/{id}/attenuation`
* **Link Budget**: `GET https://api.satlinksim.com/api/v1/simulations/{id}/link-budget`
* **Visibility**: `GET https://api.satlinksim.com/api/v1/simulations/{id}/visibility`
* **Availability**: `GET https://api.satlinksim.com/api/v1/simulations/{id}/availability`
* **Handoffs**: `GET https://api.satlinksim.com/api/v1/simulations/{id}/handoffs`
* **Orbit Coordinates**: `GET https://api.satlinksim.com/api/v1/simulations/{id}/orbit`

---

### 3.2 Coverage Resource (`/coverage`)

* **Station Coverage**: `POST https://api.satlinksim.com/api/v1/coverage/station`
  * Evaluates coverage statistics and duration intervals for specific ground stations.
* **Global Grid Coverage Map**: `POST https://api.satlinksim.com/api/v1/coverage/global`
  * Generates a global grid coverage density report based on the specified `grid_size` parameters.

---

### 3.3 Orbit Propagation Resource (`/orbit`)

* **Live Coordinate Lookup**: `GET https://api.satlinksim.com/api/v1/orbit/{satellite}`
  * Propagates to current system epoch.
* **Epoch-Specific Coordinate Lookup**: `GET https://api.satlinksim.com/api/v1/orbit/{satellite}?epoch=2026-06-25T14:00:00Z`
  * Propagates to the designated historical or future epoch.
* **Batch Orbit Propagator**: `POST https://api.satlinksim.com/api/v1/orbit/propagate`
  * Propagates coordinates over a defined duration and step size.

---

### 3.4 Rain Services Resource (`/rain`)

Exposes physics-guided telemetry inversion and stochastic forecasting models.

* **Stage A Predictor (SNR Telemetry Inversion)**: `POST https://api.satlinksim.com/api/v1/rain/predict`
  * Exposes the active algorithm model type in the response:
    ```json
    {
      "predicted_rain_rate": [0.0, 1.25, 4.3],
      "model": "stage-a"
    }
    ```
* **Ensemble Forecast**: `POST https://api.satlinksim.com/api/v1/rain/forecast`

---

### 3.5 Benchmarks Resource (`/benchmarks`)

Retrieve high-fidelity platform benchmark results:
* **Endpoint**: `GET https://api.satlinksim.com/api/v1/benchmarks`
* **Response Payload**:
  ```json
  {
    "cpu": "Intel(R) Xeon(R) Gold 6154 CPU @ 3.00GHz",
    "gpu": "NVIDIA A100-SXM4-40GB (Optional)",
    "memory_rss_mb": 142.85,
    "throughput_timesteps_per_second": 8345.10,
    "sgp4_latency_ms": 0.042,
    "rain_model_latency_ms": 0.125,
    "jit_speedup_ratio": 42.15,
    "version": "1.0.0",
    "benchmark_machine": "gcp-compute-c2-standard-4"
  }
  ```

---

### 3.6 Scientific Validation Resource (`/validation`)

Exposes mathematical accuracy, physics checks, and ITU-R standard verification status.

* **List Categories**: `GET https://api.satlinksim.com/api/v1/validation`
* **Validation Sub-Resources**:
  * Physics Invariants: `GET https://api.satlinksim.com/api/v1/validation/physics`
  * ITU Model Verification: `GET https://api.satlinksim.com/api/v1/validation/itu`
  * NASA GPM Correlation: `GET https://api.satlinksim.com/api/v1/validation/nasa`
* **Validation Artifact Downloads**:
  * JSON Report: `GET https://api.satlinksim.com/api/v1/validation/report.json`
  * PDF Formal Report: `GET https://api.satlinksim.com/api/v1/validation/report.pdf`

---

### 3.7 Datasets Resource (`/datasets`)

Exposes training datasets for link quality machine learning models.

* **List Datasets**: `GET https://api.satlinksim.com/api/v1/datasets`
* **Dataset Detail**: `GET https://api.satlinksim.com/api/v1/datasets/{id}`
* **Download Dataset**: `GET https://api.satlinksim.com/api/v1/datasets/{id}/download`
* **Upload Custom Dataset**: `POST https://api.satlinksim.com/api/v1/datasets`

---

### 3.8 Directory Resources (`/stations`, `/satellites`)
* **List Stations**: `GET https://api.satlinksim.com/api/v1/stations`
* **Station Detail**: `GET https://api.satlinksim.com/api/v1/stations/{id}` (e.g. `/stations/delhi`)
* **List Satellites**: `GET https://api.satlinksim.com/api/v1/satellites`
* **Satellite Detail**: `GET https://api.satlinksim.com/api/v1/satellites/{id}` (e.g. `/satellites/29236`)

---

### 3.9 WebSocket Streaming API

For real-time visual globes and telemetry dashboards, the platform exposes sub-second WebSocket event streams:
* **Base URL**: `wss://api.satlinksim.com/api/v1`

| Stream Route | Event Contents |
| :--- | :--- |
| `wss://api.satlinksim.com/api/v1/live/handoffs` | Emits active handoff execution events. |
| `wss://api.satlinksim.com/api/v1/live/orbits` | Stream coordinates of all tracked satellites. |
| `wss://api.satlinksim.com/api/v1/live/rain` | Emits real-time atmospheric attenuation states. |
| `wss://api.satlinksim.com/api/v1/live/snr` | Telemetry carrier-to-noise ratio streams. |
| `wss://api.satlinksim.com/api/v1/live/satellites` | Active NORAD satellite catalog alerts. |

---

### 3.10 OpenAPI & Interactive Docs
Interactive OpenAPI specs are automatically generated and served:
* **Interactive Swagger UI**: `https://api.satlinksim.com/docs`
* **JSON Raw Specification**: `https://api.satlinksim.com/openapi.json`

---

### 3.11 Health & Status Endpoint

* **Endpoint**: `GET https://api.satlinksim.com/api/v1/health`
* **Response Payload**:
  ```json
  {
    "status": "healthy",
    "timestamp": "2026-06-26T02:50:00Z",
    "database": "connected",
    "database_status": "online",
    "tle_database": "updated",
    "api_version": "1.0.0",
    "uptime": "21 days",
    "build": "abc123"
  }
  ```

---

## 4. Platform Deprecation Policy

To guarantee API stability for client systems:
* **Version Support**: A major version (e.g. `/api/v1`) is officially supported for a minimum of 12 months after a subsequent major version (e.g. `/api/v2`) is launched.
* **Deprecation Notice**: Deprecated endpoints will return a `Warning` HTTP header indicating deprecation dates.
* **Incompatibilities**: Minor and patch versions maintain complete backward compatibility in accordance with Semantic Versioning.
