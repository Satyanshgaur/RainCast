import pytest
import json
import io
import pandas as pd
from fastapi import HTTPException
from satlinksim.infrastructure.api.server import (
    simulate, link_budget, attenuation, visibility, availability, stations, satellites
)
from satlinksim.infrastructure.api.schemas import (
    PublicSimulationRequest, LinkBudgetRequest, AttenuationRequest,
    VisibilityRequest, AvailabilityRequest
)

@pytest.mark.asyncio
async def test_public_simulate_json():
    request = PublicSimulationRequest(
        satellites=["GALAXY 16"],
        ground_station="Delhi",
        frequency=14e9,
        duration=3600,
        step=60,
        rain=True,
        handoff=True
    )
    # 1. Test JSON format
    response = await simulate(request, format="json")
    assert response.status_code == 200
    
    data = json.loads(response.body.decode())
    assert "snr" in data
    assert "availability" in data
    assert "handoffs" in data
    assert "rain_loss" in data
    assert "stations" in data
    assert len(data["snr"]) == 60
    assert data["stations"] == ["Delhi"]

@pytest.mark.asyncio
async def test_public_simulate_csv_and_parquet():
    request = PublicSimulationRequest(
        satellites=["GALAXY 16"],
        ground_station="Delhi",
        frequency=14e9,
        duration=600,
        step=60,
        rain=False,
        handoff=False
    )
    # 1. Test CSV format
    response = await simulate(request, format="csv")
    assert response.status_code == 200
    assert response.headers["Content-Disposition"] == "attachment; filename=simulation_results.csv"
    
    csv_content = response.body.decode()
    df = pd.read_csv(io.StringIO(csv_content))
    assert list(df.columns) == ["time_step", "station", "satellite", "snr", "availability", "rain_loss"]
    assert len(df) == 10
    
    # 2. Test Parquet format
    response_pq = await simulate(request, format="parquet")
    assert response_pq.status_code == 200
    assert response_pq.headers["Content-Disposition"] == "attachment; filename=simulation_results.parquet"
    
    df_pq = pd.read_parquet(io.BytesIO(response_pq.body))
    assert len(df_pq) == 10
    assert list(df_pq.columns) == ["time_step", "station", "satellite", "snr", "availability", "rain_loss"]

@pytest.mark.asyncio
async def test_public_link_budget():
    request = LinkBudgetRequest(
        satellites=["GALAXY 16"],
        ground_station="Delhi",
        frequency=14e9,
        duration=600,
        step=60,
        rain=True,
        handoff=True
    )
    response = await link_budget(request, format="json")
    assert response.status_code == 200
    data = json.loads(response.body.decode())
    assert "path_loss" in data
    assert "gas_loss" in data
    assert "rain_loss" in data
    assert "rx_power" in data
    assert "snr" in data
    assert len(data["path_loss"]) == 10

@pytest.mark.asyncio
async def test_public_attenuation():
    request = AttenuationRequest(
        satellites=["GALAXY 16"],
        ground_station="Delhi",
        frequency=14e9,
        duration=600,
        step=60,
        rain=True,
        handoff=True
    )
    response = await attenuation(request, format="json")
    assert response.status_code == 200
    data = json.loads(response.body.decode())
    assert "gaseous_attenuation" in data
    assert "rain_attenuation" in data
    assert "scintillation_attenuation" in data
    assert "total_attenuation" in data

@pytest.mark.asyncio
async def test_public_visibility():
    request = VisibilityRequest(
        satellites=["GALAXY 16"],
        ground_station="Delhi",
        duration=600,
        step=60,
        min_elevation=10.0
    )
    response = await visibility(request, format="json")
    assert response.status_code == 200
    data = json.loads(response.body.decode())
    assert "time_step" in data
    assert "satellites" in data
    assert "GALAXY 16 (G-16)" in data["satellites"]
    assert "elevation" in data["satellites"]["GALAXY 16 (G-16)"]
    assert "visible" in data["satellites"]["GALAXY 16 (G-16)"]

@pytest.mark.asyncio
async def test_public_availability():
    request = AvailabilityRequest(
        satellites=["GALAXY 16"],
        ground_station="Delhi",
        frequency=14e9,
        duration=1200,
        step=60,
        rain=True,
        handoff=True,
        snr_threshold=5.0
    )
    response = await availability(request, format="json")
    assert response.status_code == 200
    data = json.loads(response.body.decode())
    assert "availability_fraction" in data
    assert "total_duration_seconds" in data
    assert "outage_duration_seconds" in data
    assert "number_of_outages" in data
    assert "outages" in data

@pytest.mark.asyncio
async def test_get_stations():
    response = await stations(name="Delhi", format="json")
    assert response.status_code == 200
    data = json.loads(response.body.decode())
    assert len(data) == 1
    assert data[0]["name"] == "Delhi"

@pytest.mark.asyncio
async def test_get_satellites():
    response = await satellites(query="GALAXY 16", limit=5, format="json")
    assert response.status_code == 200
    data = json.loads(response.body.decode())
    assert len(data) >= 1
    assert data[0]["name"] == "GALAXY 16 (G-16)"

@pytest.mark.asyncio
async def test_not_found_errors():
    request = PublicSimulationRequest(
        satellites=["NON_EXISTENT_SAT"],
        ground_station="Delhi"
    )
    with pytest.raises(HTTPException) as exc_info:
        await simulate(request, format="json")
    assert exc_info.value.status_code == 404

from satlinksim.infrastructure.api.server import (
    batch_simulation, benchmarks, validation, get_datasets, update_tle, query_tle,
    predict_rain, forecast_rain, live_handoff, predict_orbit, constellation_coverage,
    list_constellations, add_constellation
)
from satlinksim.infrastructure.api.schemas import (
    BatchSimulationRequest, PredictRainRequest, ForecastRainRequest, LiveHandoffRequest,
    OrbitRequest, CoverageRequest, ConstellationRequest
)

@pytest.mark.asyncio
async def test_batch_simulation():
    req = BatchSimulationRequest(
        satellites=["GALAXY 16"],
        ground_stations=["Delhi", "Tokyo"],
        duration=600,
        step=60,
        rain=True,
        handoff=True
    )
    response = await batch_simulation(req, format="json")
    assert response.status_code == 200
    data = json.loads(response.body.decode())
    assert "Delhi" in data
    assert "Tokyo" in data
    assert len(data["Delhi"]["snr"]) == 10

@pytest.mark.asyncio
async def test_benchmarks():
    response = await benchmarks(format="json")
    assert response.status_code == 200
    data = json.loads(response.body.decode())
    assert "throughput_timesteps_per_second" in data
    assert "avg_latency_per_step_ms" in data
    assert "propagation_latency_ms" in data

@pytest.mark.asyncio
async def test_validation():
    response = await validation(format="json")
    assert response.status_code == 200
    data = json.loads(response.body.decode())
    assert len(data) >= 2
    assert data[0]["test_name"] == "Free Space Path Loss (FSPL) Correctness"
    assert data[0]["status"] == "passed"

@pytest.mark.asyncio
async def test_datasets():
    response = await get_datasets(format="json")
    assert response.status_code == 200
    data = json.loads(response.body.decode())
    assert data["dataset_name"] == "link_training_data.parquet"
    assert "columns" in data

@pytest.mark.asyncio
async def test_tle_endpoints():
    from satlinksim.infrastructure.api.server import TleUpdateRequest
    req = TleUpdateRequest(groups=["starlink"])
    response = await update_tle(req)
    assert response["status"] == "success"
    
    response_get = await query_tle(query="GALAXY 16", limit=5, format="json")
    assert response_get.status_code == 200
    data = json.loads(response_get.body.decode())
    assert len(data) >= 1

@pytest.mark.asyncio
async def test_predict_rain():
    req = PredictRainRequest(
        snr=[15.0, 10.0, 5.0],
        elevation=[30.0, 30.0, 30.0],
        slant_range_km=[38000.0, 38000.0, 38000.0],
        ground_station="Delhi",
        frequency=14e9
    )
    response = await predict_rain(req, format="json")
    assert response.status_code == 200
    data = json.loads(response.body.decode())
    assert "predicted_rain_rate" in data
    assert len(data["predicted_rain_rate"]) == 3

@pytest.mark.asyncio
async def test_forecast_rain():
    req = ForecastRainRequest(
        current_rain_rate=5.0,
        ground_station="Delhi",
        steps=5,
        step_size=60.0,
        n_realizations=5
    )
    response = await forecast_rain(req, format="json")
    assert response.status_code == 200
    data = json.loads(response.body.decode())
    assert "mean_forecast" in data
    assert "ensemble_members" in data
    assert len(data["mean_forecast"]) == 5

@pytest.mark.asyncio
async def test_live_handoff():
    req = LiveHandoffRequest(
        current_satellite="SAT-1",
        candidates_names=["SAT-1", "SAT-2"],
        snr_metrics=[15.0, 20.0],
        el_metrics=[20.0, 25.0],
        dwell_timer=15,
        handoff_policy="highest_snr",
        hysteresis=0.5,
        min_dwell_steps=10
    )
    response = await live_handoff(req)
    assert response["should_switch"] is True
    assert response["target_satellite"] == "SAT-2"

@pytest.mark.asyncio
async def test_orbit():
    req = OrbitRequest(
        satellite="GALAXY 16",
        ground_station="Delhi",
        duration=600,
        step=60
    )
    response = await predict_orbit(req, format="json")
    assert response.status_code == 200
    data = json.loads(response.body.decode())
    assert data["satellite"] == "GALAXY 16 (G-16)"
    assert "elevation" in data
    assert len(data["elevation"]) == 10

@pytest.mark.asyncio
async def test_coverage():
    req = CoverageRequest(
        satellites=["GALAXY 16"],
        ground_stations=["Delhi", "Tokyo"],
        duration=600,
        step=60,
        min_elevation=10.0
    )
    response = await constellation_coverage(req, format="json")
    assert response.status_code == 200
    data = json.loads(response.body.decode())
    assert "Delhi" in data
    assert "Tokyo" in data
    assert "coverage_fraction" in data["Delhi"]

@pytest.mark.asyncio
async def test_constellation_registry():
    response = await list_constellations()
    assert "Starlink" in response
    
    req = ConstellationRequest(
        name="CustomConst",
        satellites=["GALAXY 16"]
    )
    response_post = await add_constellation(req)
    assert response_post["status"] == "success"
    assert response_post["constellation"] == "CustomConst"
