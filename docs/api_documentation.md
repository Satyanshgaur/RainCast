# SatLinkSim v1 Cloud-Native API Platform Specification

Welcome to the official developer documentation for the SatLinkSim Cloud-Native API Platform. This specification defines version `v1` of our resource-oriented REST and WebSocket endpoints.

### Relationship Between Legacy and v1 APIs
> [!NOTE]
> Root-level endpoints (`/simulate`, `/visibility`, etc.) remain available for backward compatibility with the CLI and Streamlit dashboard. New integrations should use the versioned `/api/v1` endpoints, which provide authentication, standardized responses, and long-term stability. All versioned endpoints require Bearer Token Authentication.

---

## 1. Global Platform Configuration

### 1.1 Base URL
All API requests must be directed to the following versioned base URL:
```http
https://api.satlinksim.com/api/v1
```

### 1.2 Timestamps Standard
All timestamps in request payloads and response bodies are standardized as ISO-8601 UTC.
Example: `2026-06-26T14:22:31Z`.

### 1.3 Authentication & Scopes
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
| `admin` | Full administrative access, including TLE updates and hardware validation. |

### 1.4 Rate Limits
Rate limits are enforced based on account tiers. Clients exceeding limits will receive a `429 Too Many Requests` response.
| Plan Tier | Hourly Request Limit | Max Concurrent Simulations | Max Simulation Duration | Max Satellites |
| :--- | :--- | :--- | :--- | :--- |
| **Free** | 100 requests/hr | 5 | 24 hours | 100 |
| **Developer** | 2,000 requests/hr | 20 | 168 hours | 500 |
| **Enterprise** | Unlimited | Custom | Unlimited | Unlimited |

### 1.5 Standard Response Envelope
All versioned JSON responses (under `/api/v1`) are wrapped in a standard response envelope. The payload maintains top-level backward compatibility, with the target response resource encapsulated in `data`, metadata in `meta` (including request IDs), and self link references in `links`.

Example Envelope:
```json
{
  "id": "e2a225de-8c83-49fb-811c-99d8213bfa70",
  "status": "completed",
  "data": {
    "id": "e2a225de-8c83-49fb-811c-99d8213bfa70",
    "status": "completed"
  },
  "meta": {
    "api_version": "1.0.0",
    "request_id": "8f2f53d9-a720-43ef-a9c6-121f0629a9ee"
  },
  "links": {
    "self": "/api/v1/simulations/e2a225de-8c83-49fb-811c-99d8213bfa70"
  }
}
```

### 1.6 Pagination Envelope
For resources returning lists of items (such as `/stations`, `/satellites`, and `/datasets`), results are wrapped in a pagination envelope. The root level preserves pagination statistics, next/previous link cursors, and the items list inside `data`.

> [!NOTE]
> **Cursor Pagination**: While page numbers are currently used for small datasets, the platform plans to transition to cursor-based pagination (e.g., `GET /simulations?cursor=eyJpZCI6...`) in future versions to efficiently scale for millions of simulation records.

Example Pagination Envelope:
```json
{
  "page": 1,
  "limit": 10,
  "total_items": 1423,
  "total_pages": 143,
  "data": [
    {
      "id": "link_training_data",
      "name": "Link Training Data"
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 10,
    "total": 1423,
    "next": "/api/v1/datasets?page=2&limit=10",
    "previous": null
  },
  "meta": {
    "api_version": "1.0.0",
    "request_id": "8f2f53d9-a720-43ef-a9c6-121f0629a9ee"
  },
  "links": {
    "self": "/api/v1/datasets"
  }
}
```

### 1.7 Simulation Lifecycle State Machine
Simulations are stateful resources with an explicit lifecycle to support asynchronous executions. 

Allowed states:
* `pending`: Simulation has been registered.
* `queued`: Simulation is in queue waiting to be executed.
* `running`: Simulation is currently executing.
* `paused`: Simulation execution is temporarily suspended.
* `completed`: Simulation executed successfully and results are ready.
* `failed`: Simulation execution failed.
* `cancelled`: Simulation execution was aborted.

Allowed transitions:
* `pending` ➔ `queued`, `cancelled`
* `queued` ➔ `running`, `cancelled`
* `running` ➔ `paused`, `completed`, `failed`, `cancelled`
* `paused` ➔ `running`, `cancelled`

### 1.8 Idempotency Support
For mutation requests (e.g., `POST /simulations`), users can include an optional `Idempotency-Key` header with a unique UUID.
* **Storage**: Keys and their original HTTP responses are cached for up to 24 hours.
* **Reuse**: Re-submitting a request with the same `Idempotency-Key` returns the cached status code and response payload immediately without re-calculating the simulation.
* **Usage Guideline**: Users should use idempotency keys on retry attempts after network timeouts to prevent duplicate resource creations and duplicated computation billing (similar to Stripe’s idempotency pattern).

### 1.9 HTTP API Headers
Every HTTP response contains versioning, tracking, and rate-limiting metadata:
* `X-Request-ID`: Unique request ID (e.g. `8f2f53d9...`) for auditing and debugging.
* `X-Compute-Time`: Server-side execution duration in seconds.
* `API-Version`: The version of the API server (`1.0.0`).
* `Deprecation`: Boolean indicating whether the requested version is deprecated (`false`).
* `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`: Standard rate-limiting headers.

### 1.10 Production Security Headers
For production deployments, the gateway/server enforces the following standard security headers:
* `Strict-Transport-Security`: `max-age=63072000; includeSubDomains; preload` (enforces HTTPS).
* `Content-Security-Policy`: `default-src 'self'; frame-ancestors 'none';` (prevents clickjacking and unauthorized script loading).
* `X-Content-Type-Options`: `nosniff` (disables MIME-type sniffing).
* `X-Frame-Options`: `DENY` (prohibits framing).

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
    "handoff": true,
    "name": "LEO Coverage Test",
    "tags": ["paper", "starlink", "ka-band"]
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

#### List Simulations
* **Endpoint**: `GET https://api.satlinksim.com/api/v1/simulations`
* **Query Parameters**:
  * `page` (Optional, integer, default: `1`): Page number for pagination.
  * `limit` (Optional, integer, default: `10`): Number of items per page.
  * `status` (Optional, string): Filter simulations by status (e.g. `completed`, `pending`, `failed`, `paused`, `running`).
  * `ground_station` (Optional, string): Filter simulations by ground station name (case-insensitive substring match).
  * `created_after` (Optional, string): Filter simulations created after this ISO-8601 timestamp.
  * `tags` (Optional, string): Filter simulations by comma-separated tags (returns simulations matching any of the specified tags).
* **Example Response Payload** (`200 OK`):
  ```json
  {
    "page": 1,
    "limit": 10,
    "total_items": 1,
    "total_pages": 1,
    "data": [
      {
        "id": "e2a225de-8c83-49fb-811c-99d8213bfa70",
        "status": "completed",
        "name": "LEO Coverage Test",
        "tags": ["paper", "starlink", "ka-band"],
        "created_at": "2026-06-26T02:30:10Z",
        "request_type": "public"
      }
    ],
    "pagination": {
      "page": 1,
      "limit": 10,
      "total": 1,
      "next": null,
      "previous": null
    },
    "meta": {
      "api_version": "1.0.0",
      "request_id": "8f2f53d9-a720-43ef-a9c6-121f0629a9ee"
    },
    "links": {
      "self": "/api/v1/simulations"
    }
  }
  ```

#### Get Simulation Status & Metadata
* **Endpoint**: `GET https://api.satlinksim.com/api/v1/simulations/{id}`
* **Response Payload** (`200 OK`):
  ```json
  {
    "id": "e2a225de-8c83-49fb-811c-99d8213bfa70",
    "status": "completed",
    "name": "LEO Coverage Test",
    "tags": ["paper", "starlink", "ka-band"],
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
      "handoff": true,
      "name": "LEO Coverage Test",
      "tags": ["paper", "starlink", "ka-band"]
    }
  }
  ```
  *(Status transitions: `pending` ➔ `running` ➔ `completed` or `failed`)*

#### Retrieve Original Request Configuration
* **Endpoint**: `GET https://api.satlinksim.com/api/v1/simulations/{id}/request`
* **Response Payload** (`200 OK`):
  ```json
  {
    "id": "e2a225de-8c83-49fb-811c-99d8213bfa70",
    "status": "completed",
    "data": {
      "satellites": ["GALAXY 16"],
      "ground_station": "Delhi",
      "frequency": 14000000000.0,
      "duration": 86400,
      "step": 60.0,
      "rain": true,
      "handoff": true,
      "name": "LEO Coverage Test",
      "tags": ["paper", "starlink", "ka-band"]
    },
    "meta": {
      "api_version": "1.0.0",
      "request_id": "8f2f53d9-a720-43ef-a9c6-121f0629a9ee"
    },
    "links": {
      "self": "/api/v1/simulations/e2a225de-8c83-49fb-811c-99d8213bfa70/request"
    }
  }
  ```


#### Pause a Simulation
* **Endpoint**: `POST https://api.satlinksim.com/api/v1/simulations/{id}/pause`
* **Response Payload** (`200 OK`):
  ```json
  {
    "id": "e2a225de-8c83-49fb-811c-99d8213bfa70",
    "status": "paused",
    "action": "pause"
  }
  ```

#### Resume a Simulation
* **Endpoint**: `POST https://api.satlinksim.com/api/v1/simulations/{id}/resume`
* **Response Payload** (`200 OK`):
  ```json
  {
    "id": "e2a225de-8c83-49fb-811c-99d8213bfa70",
    "status": "running",
    "action": "resume"
  }
  ```

#### Cancel a Simulation
* **Endpoint**: `POST https://api.satlinksim.com/api/v1/simulations/{id}/cancel`
* **Response Payload** (`200 OK`):
  ```json
  {
    "id": "e2a225de-8c83-49fb-811c-99d8213bfa70",
    "status": "cancelled",
    "action": "cancel"
  }
  ```

#### Delete a Simulation
* **Endpoint**: `DELETE https://api.satlinksim.com/api/v1/simulations/{id}`
* **Response Payload** (`200 OK`):
  ```json
  {
    "status": "success",
    "message": "Simulation e2a225de-8c83-49fb-811c-99d8213bfa70 deleted."
  }
  ```

#### Get Simulation Summary Metrics
* **Endpoint**: `GET https://api.satlinksim.com/api/v1/simulations/{id}/summary`
* **Response Payload** (`200 OK`):
  ```json
  {
    "availability": 98.2,
    "mean_snr": 14.5,
    "min_snr": 7.8,
    "max_snr": 21.4,
    "handoffs": 5,
    "outages": 2,
    "max_rain_loss": 12.3,
    "mean_rain_loss": 1.4,
    "runtime_seconds": 2.17,
    "simulation_duration": 86400,
    "samples": 1440,
    "satellites_used": 8
  }
  ```

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
* **Detailed Position Info**: `GET https://api.satlinksim.com/api/v1/orbit/{satellite}/position?epoch=2026-06-25T14:00:00Z`
  * Returns geodetic coordinates (latitude, longitude, altitude) and velocity in ECEF frame.
* **Ground Track Generator**: `GET https://api.satlinksim.com/api/v1/orbit/{satellite}/groundtrack?duration=5400&step=60`
  * Generates a series of coordinates showing the satellite's path over a defined window.
* **Pass Calculator**: `GET https://api.satlinksim.com/api/v1/orbit/{satellite}/passes?ground_station=Delhi&min_elevation=10.0`
  * Calculates rise, culmination, and set passes for the satellite over a designated station.
* **Next Pass Predictor**: `GET https://api.satlinksim.com/api/v1/orbit/{satellite}/next-pass?ground_station=Delhi&min_elevation=10.0`
  * Returns:
    ```json
    {
      "rise": "2026-06-26T03:30:00Z",
      "max_elevation": 72.5,
      "set": "2026-06-26T03:42:00Z"
    }
    ```
* **Batch Orbit Propagator**: `POST https://api.satlinksim.com/api/v1/orbit/propagate`
  * Propagates coordinates over a defined duration and step size.

---

### 3.4 Rain Services Resource (`/rain`)

Exposes physics-guided telemetry prediction and stochastic forecasting models.

* **Telemetry Predictor (SNR Telemetry Inversion)**: `POST https://api.satlinksim.com/api/v1/rain/predict` (or `/rain/invert` alias)
  * Returns:
    ```json
    {
      "model": "stage-c-frequency-aware",
      "rain_rate": [0.0, 1.25, 4.3],
      "confidence": [0.95, 0.925, 0.864]
    }
    ```
* **Ensemble Forecast**: `POST https://api.satlinksim.com/api/v1/rain/forecast`

---

### 3.5 Calculators Resource (`/calculators`)

Fast, standalone scientific calculators that bypass simulation lifecycle management.

* **Free Space Path Loss (FSPL)**: `POST https://api.satlinksim.com/api/v1/calculators/fspl`
  * Calculates attenuation due to path length:
    ```json
    {
      "frequency_hz": 14e9,
      "distance_km": 35786
    }
    ```
* **Slant Range**: `POST https://api.satlinksim.com/api/v1/calculators/slant-range`
  * Calculates distance between station and satellite:
    ```json
    {
      "altitude_km": 550.0,
      "elevation_deg": 45.0
    }
    ```
* **Thermal Noise Floor**: `POST https://api.satlinksim.com/api/v1/calculators/noise-floor`
  * Calculates thermal noise level:
    ```json
    {
      "system_temp_k": 290.0,
      "bandwidth_hz": 250e6
    }
    ```
* **Equivalent Isotropically Radiated Power (EIRP)**: `POST https://api.satlinksim.com/api/v1/calculators/eirp`
  * Calculates total radiated transmitter power:
    ```json
    {
      "tx_power_dbw": 10.0,
      "tx_gain_dbi": 38.5,
      "line_loss_db": 1.5
    }
    ```
* **ITU-R Rain Attenuation**: `POST https://api.satlinksim.com/api/v1/calculators/rain-attenuation`
  * Computes attenuation using ITU-R P.618-13 models:
    ```json
    {
      "rain_rate": 25.0,
      "elevation_deg": 35.0,
      "frequency_hz": 20e9,
      "polarization": "circular",
      "gs_latitude": 28.6
    }
    ```
* **Specific Attenuation**: `POST https://api.satlinksim.com/api/v1/calculators/specific-attenuation`
  * Computes specific attenuation $\gamma_R$ (dB/km) (ITU-R P.838):
    ```json
    {
      "rain_rate": 25.0,
      "frequency_hz": 20e9,
      "polarization": "circular"
    }
    ```
* **Effective Path Length**: `POST https://api.satlinksim.com/api/v1/calculators/effective-path`
  * Computes effective propagation path length $L_E$ (km) (ITU-R P.618):
    ```json
    {
      "elevation_deg": 35.0,
      "gs_latitude": 28.6,
      "frequency_hz": 20e9,
      "polarization": "circular"
    }
    ```
* **Total Rain Attenuation**: `POST https://api.satlinksim.com/api/v1/calculators/total-rain-attenuation`
  * Computes total rain attenuation $A = \gamma_R \cdot L_E$ (dB):
    ```json
    {
      "specific_attenuation_db_km": 1.25,
      "effective_path_length_km": 5.4
    }
    ```
* **Gaseous Absorption**: `POST https://api.satlinksim.com/api/v1/calculators/gaseous-attenuation`
  * Computes dry air and water vapor attenuation (ITU-R P.676):
    ```json
    {
      "frequency_hz": 20e9,
      "elevation_deg": 35.0,
      "water_vapor_g_m3": 7.5
    }
    ```
* **Scintillation**: `POST https://api.satlinksim.com/api/v1/calculators/scintillation`
  * Computes tropospheric scintillation intensity (ITU-R P.618):
    ```json
    {
      "frequency_hz": 20e9,
      "elevation_deg": 35.0,
      "gs_antenna_diam": 1.2
    }
    ```
* **Batch Link Budget**: `POST https://api.satlinksim.com/api/v1/calculators/link-budget`
* **Batch Attenuation**: `POST https://api.satlinksim.com/api/v1/calculators/attenuation`
* **Batch Availability**: `POST https://api.satlinksim.com/api/v1/calculators/availability`

---

### 3.6 Directory Resources (`/stations`, `/satellites`)
* **List Stations**: `GET https://api.satlinksim.com/api/v1/stations`
* **Station Detail**: `GET https://api.satlinksim.com/api/v1/stations/{id}` (e.g. `/stations/delhi`)
* **List Satellites**: `GET https://api.satlinksim.com/api/v1/satellites`
* **Satellite Detail**: `GET https://api.satlinksim.com/api/v1/satellites/{id}` (e.g. `/satellites/29236`)

---

### 3.7 Datasets Resource (`/datasets`)

Exposes training datasets for link quality machine learning models.

* **List Datasets**: `GET https://api.satlinksim.com/api/v1/datasets`
* **Dataset Detail**: `GET https://api.satlinksim.com/api/v1/datasets/{id}`
* **Download Dataset**: `GET https://api.satlinksim.com/api/v1/datasets/{id}/download`

> [!NOTE]
> `POST /datasets` has been removed initially to maintain strict compliance with storage policies and authorization controls.

---

### 3.8 Realtime & TLE APIs

Endpoints supporting real-time visualization globes, SaaS tools, and operator catalogs.

* **Realtime Globe Metrics**: `GET https://api.satlinksim.com/api/v1/realtime/globe` (or `/live/globe` alias)
  * Returns active simulations and average satellite elevations.
* **Constellation Status**: `GET https://api.satlinksim.com/api/v1/realtime/constellation?constellation=Starlink` (or `/live/constellation` alias)
  * Returns availability metrics and count of active nodes in named constellations.
* **Recent Handoff Events**: `GET https://api.satlinksim.com/api/v1/realtime/handoffs` (or `/live/handoffs` alias)
  * Returns the latest execution records for inter-satellite/station handoffs.
* **TLE Cache Status**: `GET https://api.satlinksim.com/api/v1/tle/status`
* **Trigger TLE Update**: `POST https://api.satlinksim.com/api/v1/tle/update` (Or `/tle` alias)
  * Accepts an optional body containing satellite groups to fetch:
    ```json
    {
      "groups": ["active", "starlink"]
    }
    ```
* **NORAD Operators Catalog**: `GET https://api.satlinksim.com/api/v1/tle/operators`

---

### 3.9 WebSocket Streaming API

For real-time visual globes and telemetry dashboards, the platform exposes a single sub-second WebSocket event stream:
* **Base URL**: `wss://api.satlinksim.com/api/v1/stream/events`

Supported client event subscriptions include:
* `orbit_update`: Live coordinate changes for propagated satellites.
* `handoff`: Instantaneous handoff execution triggers.
* `rain_event`: Dynamic changes in localized rain rates.
* `snr_update`: Real-time signal strength telemetry updates.
* `availability_change`: Instant status switches.
* `tle_update`: Alerts when database cache retrieves new ephemerides.

Standardized Event Payload Format:
```json
{
  "event": "orbit_update",
  "timestamp": "2026-06-26T03:20:00Z",
  "simulation_id": "e2a225de-8c83-49fb-811c-99d8213bfa70",
  "payload": {
    "satellite": "GALAXY 16",
    "latitude": 0.0,
    "longitude": -99.0,
    "altitude_km": 35786.0
  }
}
```

#### Connection Management & Reconnection Policy
* **Heartbeat Protocol**: The server sends a heartbeat `ping` frame every 30 seconds. Clients must respond with a corresponding `pong` frame within 5 seconds to prevent the server from terminating the connection.
* **Reconnection Strategy**: In case of unexpected disconnection, clients should implement exponential backoff reconnection with random jitter (e.g., initial delay of 1s, doubling up to a maximum of 60s, plus or minus 15% random jitter). This prevents the platform from experiencing a thundering herd problem during server/network recovery.
* **Delivery Semantics & Event Ordering**: Event delivery follows a **best-effort** model. Telemetry events originating from the same simulation are guaranteed to be ordered sequentially, but message delivery is not guaranteed during net splits. Clients should handle dropped telemetry packets gracefully by checking statuses via REST.

---

### 3.10 OpenAPI & Interactive Docs
Interactive OpenAPI specs are automatically generated and served:
* **Interactive Swagger UI**: `https://api.satlinksim.com/docs`
* **JSON Raw Specification**: `https://api.satlinksim.com/openapi.json`

> [!NOTE]
> Official SDKs generated from the OpenAPI specification will be published for Python, JavaScript/TypeScript, and Go.

---

### 3.11 Health & Status Endpoint

* **Endpoint**: `GET https://api.satlinksim.com/api/v1/health`
* **Response Payload**:
  ```json
  {
    "status": "healthy",
    "database": "connected",
    "tle_cache": "updated",
    "uptime": "21d",
    "version": "1.0.0",
    "python": "3.12",
    "numba": "enabled",
    "cpu": "healthy"
  }
  ```

---

### 3.12 System Information Endpoint

* **Endpoint**: `GET https://api.satlinksim.com/api/v1/system/info`
* **Response Payload**:
  ```json
  {
    "version": "1.0.0",
    "physics_models": [
      "ITU-R P.618-13",
      "ITU-R P.676-13",
      "ITU-R P.837-7",
      "ITU-R P.838-3",
      "SGP4"
    ],
    "ml_models": [
      "Stage-C Frequency-Aware XGBoost"
    ],
    "build": "abc123",
    "last_tle_update": "2026-06-26T03:20:00Z"
  }
  ```

---

## 4. Documentation & Benchmarks (Static Resources)

In accordance with keeping our API surfaces focused and operational, historical validation reports and static benchmark results are hosted as markdown resources:
* For physics models and verification, refer to: [validation.md](file:///home/satyansh/leo_meo/docs/validation.md)
* For hardware and execution benchmarks, refer to: [benchmarks.md](file:///home/satyansh/leo_meo/docs/benchmarks.md)

---

## 5. Structured Documentation Portal Layout

The documentation is published as a unified portal using the following structure:

* **Getting Started**
  * Installation
  * Quick Start
  * Tutorials
* **API Reference**
  * Authenticating requests
  * Simulation management
  * Directory & Operators
  * Telemetry stream
* **Physics Reference**
  * ITU-R RF Propagation
  * Orbital Mechanics (SGP4)
  * Atmospheric models
* **Validation & Metrics**
  * FSPL & Attenuation Reports
  * Speed & CPU Benchmarks
  * Machine Learning datasets
* **Additional Resources**
  * Examples & Client Notebooks
  * Frequently Asked Questions (FAQ)
  * Version changelog

---

## 6. Platform Deprecation Policy

To guarantee API stability for client systems, we adhere to a structured deprecation lifecycle:

```
Active & Supported ➔ Deprecated (Supported for 12 Months) ➔ Sunset / Removed
```

* **Lifecycle Definition**:
  1. **Active & Supported**: Current stable production version.
  2. **Deprecated (12 Months Support)**: When a new major API version (e.g. `/api/v2`) is introduced, the legacy version (e.g. `/api/v1`) enters deprecation status. It is officially supported with security patches and critical bug fixes for exactly **12 months** from the deprecation announcement date.
  3. **Sunset / Removed**: After the 12-month period expires, the version is fully removed and calls will return `410 Gone` or `404 Not Found`.

* **HTTP Deprecation Headers**:
  Endpoints belonging to a deprecated version will return the following headers in every response:
  * `Sunset`: Specifying the timestamp of version removal (RFC 7231 format).
    * *Example*: `Sunset: Sat, 26 Jun 2027 00:00:00 GMT`
  * `Warning`: Standard API warning header (code `299`).
    * *Example*: `Warning: 299 - "API version v1 is deprecated and will be sunset on 2027-06-26T00:00:00Z"`
  * `Link`: Hyperlink referencing version migration guides or deprecation announcements.
    * *Example*: `Link: <https://api.satlinksim.com/docs/deprecation/v1>; rel="sunset"`

* **Incompatibilities**: Minor and patch versions maintain complete backward compatibility in accordance with Semantic Versioning.

