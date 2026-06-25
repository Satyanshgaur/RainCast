# REST API Documentation

This document explains how to run, configure, and query the public REST APIs in `satlinksim`.

All endpoints are built using FastAPI and support three output formats: **JSON**, **CSV**, and **Parquet**, which can be specified using the `format` query parameter.

---

## Running the API Server

Before querying the APIs, start the FastAPI server. Ensure your virtual environment is active:

### 1. Development Mode (with hot-reload)
Use Uvicorn directly to start the server:
```bash
.venv/bin/uvicorn satlinksim.infrastructure.api.server:app --reload --host 0.0.0.0 --port 8000
```

### 2. Standard Service Mode
Use the CLI command:
```bash
.venv/bin/satlinksim api
# Or if your virtual environment is activated:
satlinksim api
```

### 3. Interactive OpenAPI Docs (Swagger)
Once the server is running, you can explore the schemas, inputs, and execute test requests in your browser:
* **Interactive Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
* **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## Global Format Parameter

Every endpoint listed below accepts an optional query parameter `format`:
* **`format`** (string, default: `"json"`): Output format. Choices are `json`, `csv`, or `parquet`.
  * **`json`**: Returns a JSON object/list.
  * **`csv`**: Returns a tabular text file (`text/csv`) download.
  * **`parquet`**: Returns a binary Apache Parquet file (`application/octet-stream`) download.

*Note: For `csv` and `parquet` formats on summary endpoints (e.g., `/availability`), scalar summary fields are returned as custom HTTP response headers (e.g., `X-Availability-Fraction`) while the timeseries data is returned in the file.*

---

## API Reference

### 1. POST `/simulate`
Executes an orbital and atmospheric link simulation over time, returning key metrics (SNR, rain attenuation, availability status, and handoff history).

* **Path:** `/simulate`
* **Method:** `POST`
* **Request Body (JSON):**
  * `satellites` (List of strings or integers, required): Satellite names or NORAD IDs to simulate (e.g., `["STARLINK-1008"]`).
  * `ground_station` (string or object, required): Predefined ground station name (e.g. `"Delhi"`, `"Tokyo"`, `"Berlin"`, `"Sao Paulo"`) OR a dictionary containing custom ground station parameters:
    * `name` (string, required): Ground station identifier.
    * `latitude` (float, required): Latitude in degrees.
    * `longitude` (float, required): Longitude in degrees.
    * `altitude_km` (float, required): Altitude above sea level in km.
    * `eirp_dbw` (float, required): Transmitter EIRP in dBW.
    * `g_rx_dbi` (float, required): Antenna receiver gain in dBi.
    * `system_temp_k` (float, required): System noise temperature in Kelvin.
    * `antenna_diam_m` (float, required): Antenna diameter in meters.
  * `frequency` (float, optional, default: `14e9`): Transmission frequency in Hz.
  * `duration` (integer, optional, default: `86400`): Duration of simulation in seconds.
  * `step` (float, optional, default: `60.0`): Simulation step size in seconds.
  * `rain` (boolean, optional, default: `true`): If `true`, stochastically generates rain attenuation. If `false`, rain attenuation is mathematically zeroed out.
  * `handoff` (boolean, optional, default: `true`): If `true`, performs dynamic highest-elevation satellite handoffs. If `false`, locks onto the first satellite in the list.

* **Example JSON Request:**
  ```json
  {
    "satellites": ["STARLINK-1008"],
    "ground_station": "Delhi",
    "frequency": 14e9,
    "duration": 86400,
    "step": 60,
    "rain": true,
    "handoff": true
  }
  ```

* **Example JSON Response:**
  ```json
  {
    "snr": [15.2, 14.8, 11.5, 15.1],
    "availability": [1, 1, 1, 1],
    "handoffs": [
      {
        "time_step": 120,
        "old_sat": "STARLINK-1008",
        "new_sat": "STARLINK-1009",
        "reason": "elevation",
        "metric_delta": 4.5
      }
    ],
    "rain_loss": [0.0, 0.0, 3.4, 0.0],
    "stations": ["Delhi"]
  }
  ```

* **Tabular Schema (CSV/Parquet):**
  Columns: `time_step`, `station`, `satellite`, `snr`, `availability`, `rain_loss`.

* **Example `curl` Command:**
  ```bash
  curl -X POST "http://localhost:8000/simulate?format=json" \
       -H "Content-Type: application/json" \
       -d '{
         "satellites": ["STARLINK-1008"],
         "ground_station": "Delhi",
         "frequency": 14e9,
         "duration": 86400,
         "step": 60
       }'
  ```

---

### 2. POST `/link-budget`
Returns a detailed timestep-by-timestep breakdown of the link budget components (EIRP, FSPL, gas absorption, rain loss, scintillation loss, received power, noise floor, and SNR).

* **Path:** `/link-budget`
* **Method:** `POST`
* **Request Body (JSON):**
  Includes all options from `/simulate`, plus RF parameters:
  * `polarization` (string, optional, default: `"vertical"`): Wave polarization (`"vertical"` or `"horizontal"`).
  * `bandwidth_hz` (float, optional, default: `36e6`): Carrier bandwidth in Hz (used to compute the thermal noise floor).

* **Example JSON Request:**
  ```json
  {
    "satellites": ["STARLINK-1008"],
    "ground_station": "Delhi",
    "frequency": 14e9,
    "duration": 3600,
    "step": 60,
    "rain": true,
    "handoff": true,
    "polarization": "vertical",
    "bandwidth_hz": 36e6
  }
  ```

* **Example JSON Response:**
  ```json
  {
    "time_step": [0, 1, 2],
    "eirp": [52.0, 52.0, 52.0],
    "path_loss": [205.1, 205.2, 205.3],
    "gas_loss": [0.80, 0.81, 0.82],
    "rain_loss": [0.0, 0.0, 0.0],
    "scint_loss": [0.12, -0.05, 0.08],
    "rx_power": [-108.92, -108.76, -108.96],
    "noise_floor": [-127.4, -127.4, -127.4],
    "snr": [18.48, 18.64, 18.44]
  }
  ```

* **Tabular Schema (CSV/Parquet):**
  Columns: `time_step`, `eirp`, `path_loss`, `gas_loss`, `rain_loss`, `scint_loss`, `rx_power`, `noise_floor`, `snr`.

* **Example `curl` Command:**
  ```bash
  curl -X POST "http://localhost:8000/link-budget?format=csv" \
       -H "Content-Type: application/json" \
       -d '{
         "satellites": ["STARLINK-1008"],
         "ground_station": "Delhi",
         "frequency": 14e9,
         "duration": 600,
         "step": 60
       }' --output link_budget.csv
  ```

---

### 3. POST `/attenuation`
Exposes the specific breakdown of atmospheric signal attenuation over time.

* **Path:** `/attenuation`
* **Method:** `POST`
* **Request Body (JSON):**
  Same as `/simulate`.

* **Example JSON Response:**
  ```json
  {
    "gaseous_attenuation": [0.81, 0.82, 0.83],
    "rain_attenuation": [0.0, 1.25, 4.30],
    "scintillation_attenuation": [0.15, -0.02, 0.09],
    "total_attenuation": [0.96, 2.05, 5.22]
  }
  ```

* **Tabular Schema (CSV/Parquet):**
  Columns: `time_step`, `gaseous_attenuation`, `rain_attenuation`, `scintillation_attenuation`, `total_attenuation`.

* **Example `curl` Command:**
  ```bash
  curl -X POST "http://localhost:8000/attenuation?format=json" \
       -H "Content-Type: application/json" \
       -d '{
         "satellites": ["STARLINK-1008"],
         "ground_station": "Delhi",
         "frequency": 14e9,
         "duration": 600,
         "step": 60
       }'
  ```

---

### 4. POST `/visibility`
Calculates when requested satellites are visible (above a minimum elevation angle) to the ground station.

* **Path:** `/visibility`
* **Method:** `POST`
* **Request Body (JSON):**
  * `satellites` (List, required): List of satellite names or NORAD IDs.
  * `ground_station` (string/object, required): Predefined profile or custom station specs.
  * `duration` (integer, optional, default: `86400`): Duration in seconds.
  * `step` (float, optional, default: `60.0`): Step size in seconds.
  * `min_elevation` (float, optional, default: `10.0`): Minimum elevation angle in degrees for visibility.

* **Example JSON Request:**
  ```json
  {
    "satellites": ["STARLINK-1008"],
    "ground_station": "Delhi",
    "duration": 3600,
    "step": 60,
    "min_elevation": 10.0
  }
  ```

* **Example JSON Response:**
  ```json
  {
    "time_step": [0, 1, 2],
    "satellites": {
      "STARLINK-1008": {
        "elevation": [12.4, 15.6, 9.8],
        "azimuth": [180.2, 185.4, 190.1],
        "visible": [1, 1, 0]
      }
    }
  }
  ```

* **Tabular Schema (CSV/Parquet):**
  Long-format table representing all target satellites over time:
  Columns: `time_step`, `satellite`, `elevation`, `azimuth`, `visible`.

* **Example `curl` Command:**
  ```bash
  curl -X POST "http://localhost:8000/visibility?format=csv" \
       -H "Content-Type: application/json" \
       -d '{
         "satellites": ["STARLINK-1008", "STARLINK-1009"],
         "ground_station": "Delhi",
         "duration": 3600,
         "step": 60,
         "min_elevation": 10.0
       }' --output visibility.csv
  ```

---

### 5. POST `/availability`
Computes link availability percentage and lists individual outage events.

* **Path:** `/availability`
* **Method:** `POST`
* **Request Body (JSON):**
  Same as `/simulate`, plus:
  * `snr_threshold` (float, optional, default: `5.0`): Minimum SNR in dB required to prevent an outage.

* **Example JSON Request:**
  ```json
  {
    "satellites": ["STARLINK-1008"],
    "ground_station": "Delhi",
    "frequency": 14e9,
    "duration": 86400,
    "step": 60,
    "snr_threshold": 5.0
  }
  ```

* **Example JSON Response:**
  ```json
  {
    "availability_fraction": 0.985,
    "total_duration_seconds": 86400,
    "outage_duration_seconds": 1296,
    "number_of_outages": 1,
    "outages": [
      {
        "start_step": 320,
        "end_step": 332,
        "duration_seconds": 720.0
      }
    ]
  }
  ```

* **Tabular Schema (CSV/Parquet):**
  Columns: `time_step`, `snr`, `available` (`1` if available, `0` if outage).
  
  *Note: For CSV and Parquet outputs, summary metrics (`availability_fraction`, `number_of_outages`, etc.) are returned inside custom HTTP response headers: `X-Availability-Fraction`, `X-Total-Duration-Seconds`, `X-Outage-Duration-Seconds`, and `X-Number-Of-Outages`.*

* **Example `curl` Command:**
  ```bash
  curl -X POST "http://localhost:8000/availability?format=json" \
       -H "Content-Type: application/json" \
       -d '{
         "satellites": ["STARLINK-1008"],
         "ground_station": "Delhi",
         "frequency": 14e9,
         "duration": 3600,
         "step": 60,
         "snr_threshold": 5.0
       }'
  ```

---

### 6. GET `/stations`
Returns profiles and RF hardware specs of predefined ground stations.

* **Path:** `/stations`
* **Method:** `GET`
* **Query Parameters:**
  * `name` (string, optional): Search/filter substring match.

* **Example JSON Response:**
  ```json
  [
    {
      "name": "Delhi",
      "eirp_dbw": 52.0,
      "g_rx_dbi": 45.0,
      "system_temp_k": 500.0,
      "antenna_diam_m": 2.4,
      "latitude": 28.6,
      "longitude": 77.2,
      "altitude_km": 0.216,
      "itu_rain": {
        "R001": 42.0,
        "R01": 19.0,
        "R1": 6.0,
        "P_rain": 0.053
      },
      "wv_g_m3": 12.0,
      "humidity_pct": 70.0,
      "v_radial_ms": -30.0
    }
  ]
  ```

* **Tabular Schema (CSV/Parquet):**
  Flat table with flattened `itu_rain` parameters:
  Columns: `name`, `eirp_dbw`, `g_rx_dbi`, `system_temp_k`, `antenna_diam_m`, `latitude`, `longitude`, `altitude_km`, `wv_g_m3`, `humidity_pct`, `v_radial_ms`, `sat_lon_deg`, `norad_id`, `sat_name`, `itu_rain_R001`, `itu_rain_R01`, `itu_rain_R1`, `itu_rain_P_rain`.

* **Example `curl` Command:**
  ```bash
  curl "http://localhost:8000/stations?name=Delhi&format=csv"
  ```

---

### 7. GET `/satellites`
Queries the local SQLite database for cached satellites TLEs.

* **Path:** `/satellites`
* **Method:** `GET`
* **Query Parameters:**
  * `query` (string, optional): Search filter matching name or NORAD ID.
  * `limit` (integer, optional, default: `100`): Maximum size of records to return.

* **Example JSON Response:**
  ```json
  [
    {
      "name": "INTELSAT 10 (IS-10)",
      "norad_id": 26766,
      "tle_line1": "1 26766U 01019A   26035.04021726 -.00000036  00000+0  00000+0 0  9994",
      "tle_line2": "2 26766   8.7570  60.6079 0008434 252.6678 104.5252  0.99072725 44254"
    }
  ]
  ```

* **Tabular Schema (CSV/Parquet):**
  Columns: `name`, `norad_id`, `tle_line1`, `tle_line2`.

* **Example `curl` Command:**
  ```bash
  curl "http://localhost:8000/satellites?query=GALAXY&limit=5&format=json"
  ```

---

## 8. POST `/batch`
Runs a batched simulation for multiple ground stations concurrently.

* **Path:** `/batch`
* **Method:** `POST`
* **Request Body (JSON):**
  * `satellites` (List, required): List of satellite names or NORAD IDs.
  * `ground_stations` (List of strings or objects, required): Predefined ground station names or custom specs.
  * `frequency`, `duration`, `step`, `rain`, `handoff` (same as `/simulate`).

* **Example JSON Request:**
  ```json
  {
    "satellites": ["GALAXY 16"],
    "ground_stations": ["Delhi", "Tokyo"],
    "duration": 3600,
    "step": 60
  }
  ```

* **Example JSON Response:**
  ```json
  {
    "Delhi": {
      "snr": [15.2, 14.8],
      "availability": [1, 1],
      "rain_loss": [0.0, 0.0],
      "handoffs": 0
    },
    "Tokyo": {
      "snr": [18.1, 18.2],
      "availability": [1, 1],
      "rain_loss": [0.0, 0.0],
      "handoffs": 0
    }
  }
  ```

* **Tabular Schema (CSV/Parquet):**
  Columns: `station`, `time_step`, `snr`, `availability`, `rain_loss`.

---

## 9. GET `/benchmarks`
Runs a live simulation throughput and propagation latency benchmark on the server, returning system performance statistics.

* **Path:** `/benchmarks`
* **Method:** `GET` or `POST`
* **Example JSON Response:**
  ```json
  {
    "throughput_timesteps_per_second": 32154.5,
    "avg_latency_per_step_ms": 0.031,
    "propagation_latency_ms": 0.0042,
    "memory_rss_mb": 112.4
  }
  ```

* **Tabular Schema (CSV/Parquet):**
  Columns: `throughput_timesteps_per_second`, `avg_latency_per_step_ms`, `propagation_latency_ms`, `memory_rss_mb`.

---

## 10. GET `/validation`
Validates physical invariants calculated by the simulator (FSPL, Rain Height, Slant Range) against their standard theoretical formulations.

* **Path:** `/validation`
* **Method:** `GET` or `POST`
* **Example JSON Response:**
  ```json
  [
    {
      "test_name": "Free Space Path Loss (FSPL) Correctness",
      "calculated": 207.3941,
      "reference": 207.3941,
      "difference": 0.0,
      "status": "passed"
    },
    {
      "test_name": "ITU-R P.839-4 Rain Height Correctness",
      "calculated": 4.58,
      "reference": 4.58,
      "difference": 0.0,
      "status": "passed"
    }
  ]
  ```

* **Tabular Schema (CSV/Parquet):**
  Columns: `test_name`, `calculated`, `reference`, `difference`, `status`.

---

## 11. GET `/datasets`
Provides details and descriptive statistics of the machine learning training datasets inside the project.

* **Path:** `/datasets`
* **Method:** `GET`
* **Example JSON Response:**
  ```json
  {
    "dataset_name": "link_training_data.parquet",
    "file_size_bytes": 35029,
    "total_rows": 1000,
    "columns": ["snr_db", "packet_loss", "load_factor", "link_quality"],
    "features": ["snr_db", "packet_loss", "load_factor"],
    "target": "link_quality",
    "summary_statistics": {
      "snr_db": { "mean": 14.2, "std": 3.1 },
      ...
    }
  }
  ```

* **Tabular Schema (CSV/Parquet):**
  Returns flattened statistical metrics:
  Columns: `metric` (count, mean, std, min, max), `snr_db`, `packet_loss`, `load_factor`, `link_quality`.

---

## 12. POST `/tle`
Triggers TLE database updates for starlink, oneweb, geo, iridium, and globalstar groups from CelesTrak.

* **Path:** `/tle`
* **Method:** `POST`
* **Request Body (JSON):**
  * `groups` (List of strings, optional): Groups to update (`"starlink"`, `"oneweb"`, `"geo"`, `"iridium"`, `"globalstar"`). If omitted, all groups are updated.
* **Example JSON Response:**
  ```json
  {
    "status": "success",
    "message": "TLE database updated successfully",
    "total_satellites": 1342
  }
  ```

*Note: You can query the TLE database using `GET /tle?query=STARLINK`.*

---

## 13. POST `/predict-rain`
Utilizes physics-guided Stage A analytical inversion to reconstruct rainfall intensity (mm/h) directly from link telemetry.

* **Path:** `/predict-rain`
* **Method:** `POST`
* **Request Body (JSON):**
  * `snr` (float or List, required): Observed link SNR values in dB.
  * `elevation` (float or List, required): Elevation angles in degrees.
  * `slant_range_km` (float or List, required): Slant ranges in km.
  * `ground_station` (string/object, required): Ground station name or profile.
  * `frequency` (float, optional, default: `14e9`): Link frequency in Hz.
  * `polarization` (string, optional, default: `"vertical"`): polarization.
  * `bandwidth_hz` (float, optional, default: `36e6`): Carrier bandwidth.

* **Example JSON Request:**
  ```json
  {
    "snr": [15.0, 10.0, 5.0],
    "elevation": [30.0, 30.0, 30.0],
    "slant_range_km": [38000.0, 38000.0, 38000.0],
    "ground_station": "Delhi",
    "frequency": 14e9
  }
  ```

* **Example JSON Response:**
  ```json
  {
    "predicted_rain_rate": [0.0, 1.25, 12.4]
  }
  ```

* **Tabular Schema (CSV/Parquet):**
  Columns: `time_step`, `observed_snr`, `excess_attenuation`, `filtered_attenuation`, `predicted_rain_rate`.

---

## 14. POST `/forecast-rain`
Generates stochastic forward forecasts of future rain rates utilizing the Maseng-Bakken first-order autoregressive process.

* **Path:** `/forecast-rain`
* **Method:** `POST`
* **Request Body (JSON):**
  * `current_rain_rate` (float, required): Initial rain rate in mm/h.
  * `ground_station` (string/object, required): Ground station name or profile.
  * `steps` (integer, optional, default: `10`): Number of future steps to forecast.
  * `step_size` (float, optional, default: `60.0`): Time-step in seconds.
  * `n_realizations` (integer, optional, default: `10`): Number of ensemble members.

* **Example JSON Response:**
  ```json
  {
    "mean_forecast": [4.8, 4.5, 4.2],
    "p90_forecast": [5.2, 5.5, 5.9],
    "p10_forecast": [4.0, 3.8, 3.5],
    "ensemble_members": [
      [4.9, 4.8, 4.6],
      [4.7, 4.2, 3.8]
    ]
  }
  ```

* **Tabular Schema (CSV/Parquet):**
  Returns the long-format ensemble forecast:
  Columns: `realization_id`, `time_step`, `predicted_rain_rate`.

---

## 15. POST `/handoff/live`
Evaluates a live handoff decision against candidate satellites.

* **Path:** `/handoff/live`
* **Method:** `POST`
* **Request Body (JSON):**
  * `current_satellite` (string, optional): Currently connected satellite.
  * `candidates_names` (List of strings, required): Candidate satellites.
  * `snr_metrics` (List of floats, required): SNR values of candidate satellites.
  * `el_metrics` (List of floats, required): Elevation angles of candidate satellites.
  * `dwell_timer` (integer, optional, default: `0`): Current dwell timer in steps.
  * `handoff_policy` (string, optional, default: `"highest_elevation"`): `"highest_elevation"` or `"highest_snr"`.
  * `hysteresis` (float, optional, default: `0.5`): Hysteresis margin in dB.
  * `min_dwell_steps` (integer, optional, default: `10`): Minimum dwell threshold.

* **Example JSON Request:**
  ```json
  {
    "current_satellite": "SAT-1",
    "candidates_names": ["SAT-1", "SAT-2"],
    "snr_metrics": [15.0, 20.0],
    "el_metrics": [20.0, 25.0],
    "dwell_timer": 15,
    "handoff_policy": "highest_snr"
  }
  ```

* **Example JSON Response:**
  ```json
  {
    "should_switch": true,
    "target_satellite": "SAT-2",
    "reason": "highest_snr",
    "metric_delta": 5.0
  }
  ```

---

## 16. POST `/orbit`
Predicts high-fidelity orbital geometry (slant range, azimuth, elevation, Doppler radial velocity) for a satellite over a target ground station.

* **Path:** `/orbit`
* **Method:** `POST`
* **Request Body (JSON):**
  * `satellite` (string/int, required): NORAD ID or name.
  * `ground_station` (string/object, required): Ground station coordinates/name.
  * `time` (datetime string, optional): Start timestamp.
  * `duration` (integer, optional): Duration in seconds.
  * `step` (float, optional, default: `60.0`): Time-step in seconds.

* **Example JSON Response:**
  ```json
  {
    "satellite": "STARLINK-1008",
    "time_step": [0, 1],
    "elevation": [45.2, 45.8],
    "slant_range_km": [850.4, 845.2],
    "radial_velocity_ms": [-120.4, -98.1],
    "azimuth": [12.4, 12.8]
  }
  ```

* **Tabular Schema (CSV/Parquet):**
  Columns: `time_step`, `satellite`, `elevation`, `slant_range_km`, `radial_velocity_ms`, `azimuth`.

---

## 17. POST `/coverage`
Computes visibility coverage fractions for a satellite constellation across multiple ground stations.

* **Path:** `/coverage`
* **Method:** `POST`
* **Request Body (JSON):**
  * `satellites` (List, required): Targets constellation satellites.
  * `ground_stations` (List, required): List of ground stations.
  * `duration`, `step` (same as `/simulate`).
  * `min_elevation` (float, optional, default: `10.0`): Minimum elevation angle.

* **Example JSON Response:**
  ```json
  {
    "Delhi": {
      "coverage_fraction": 0.85,
      "visible_duration_seconds": 3060.0,
      "total_duration_seconds": 3600.0
    },
    "Tokyo": {
      "coverage_fraction": 0.95,
      "visible_duration_seconds": 3420.0,
      "total_duration_seconds": 3600.0
    }
  }
  ```

* **Tabular Schema (CSV/Parquet):**
  Columns: `station`, `coverage_fraction`, `visible_duration_seconds`, `total_duration_seconds`.

---

## 18. GET/POST `/constellation`
Registers or retrieves satellite constellation layouts.

* **Path:** `/constellation`
* **Methods:**
  * `GET`: Lists all defined constellations.
  * `POST`: Registers a new constellation.
* **POST Request Body (JSON):**
  * `name` (string, required): Name of constellation.
  * `satellites` (List, required): NORAD IDs or names.
* **Example JSON Response (POST):**
  ```json
  {
    "status": "success",
    "constellation": "CustomConst",
    "satellites": [44057, 44059]
  }
  ```

---

## Python Integration Example

Here is a short Python script using the `requests` library to query the API and load the results into a Pandas DataFrame:

```python
import requests
import io
import pandas as pd

# Define simulation request payload
payload = {
    "satellites": ["STARLINK-1008"],
    "ground_station": "Delhi",
    "frequency": 14e9,
    "duration": 3600,
    "step": 60
}

# 1. Fetch JSON
res_json = requests.post("http://localhost:8000/simulate?format=json", json=payload)
data = res_json.json()
print("Mean SNR:", sum(data["snr"]) / len(data["snr"]))

# 2. Fetch CSV and load directly into Pandas DataFrame
res_csv = requests.post("http://localhost:8000/simulate?format=csv", json=payload)
df = pd.read_csv(io.StringIO(res_csv.text))
print(df.head())
```
