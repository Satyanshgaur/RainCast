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
