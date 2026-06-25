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
