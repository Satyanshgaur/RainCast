import pytest
import json
import io
import pandas as pd
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from satlinksim.infrastructure.api.server import (
    verify_token, create_simulation, list_simulations, get_simulation,
    delete_simulation, get_simulation_results, get_simulation_attenuation,
    get_simulation_link_budget, get_simulation_visibility, get_simulation_availability,
    get_simulation_handoffs, get_simulation_orbit, create_job, get_job_v1,
    get_job_results_v1, calculator_link_budget, calculator_attenuation,
    calculator_availability, rain_predict_v1, rain_forecast_v1, coverage_v1,
    global_coverage, handoff_decision_v1, query_satellite_orbit, orbit_propagate_v1,
    stations_v1, satellites_v1, datasets_list_v1, dataset_detail_v1,
    dataset_download_v1, benchmarks_v1, validate_physics, validate_itu,
    validate_nasa, validate_regression, validate_all_v1, update_tle_v1,
    health_v1, simulations_store, jobs, GlobalCoverageRequest, TleUpdateRequest
)
from satlinksim.infrastructure.api.schemas import (
    PublicSimulationRequest, LinkBudgetRequest, AttenuationRequest,
    AvailabilityRequest, PredictRainRequest, ForecastRainRequest,
    LiveHandoffRequest, CoverageRequest, OrbitRequest,
    BatchSimulationRequest
)

@pytest.mark.asyncio
async def test_verify_token():
    # 1. Test valid token
    cred = HTTPAuthorizationCredentials(scheme="Bearer", credentials="sk_test_token")
    token = await verify_token(cred)
    assert token == "sk_test_token"
    
    # 2. Test invalid token
    cred_invalid = HTTPAuthorizationCredentials(scheme="Bearer", credentials="invalid_token")
    with pytest.raises(HTTPException) as exc_info:
        await verify_token(cred_invalid)
    assert exc_info.value.status_code == 401
    assert "Invalid or missing API Key" in exc_info.value.detail

@pytest.mark.asyncio
async def test_health_v1():
    response = await health_v1()
    assert response["status"] == "healthy"
    assert "timestamp" in response
    assert response["database"] == "connected"

@pytest.mark.asyncio
async def test_stations_v1():
    # Page 1, limit 2
    response = await stations_v1(page=1, limit=2)
    data = json.loads(response.body.decode())
    assert data["page"] == 1
    assert data["limit"] == 2
    assert len(data["data"]) <= 2

@pytest.mark.asyncio
async def test_satellites_v1():
    # Insert mock Starlink satellites to ensure the query returns data on a fresh DB
    import sqlite3
    import os
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src", "satlinksim", "satellites.db")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS satellites (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        norad_id INTEGER UNIQUE,
        tle_line1 TEXT,
        tle_line2 TEXT
    )
    """)
    mock_sats = [
        ("STARLINK-1", 44057, "1 44057U 19029A   20001.00000000  .00000000  00000-0  00000-0 0  9999", "2 44057  53.0000  0.0000 0001000   0.0000   0.0000 15.00000000    12"),
        ("STARLINK-2", 44059, "1 44059U 19029B   20001.00000000  .00000000  00000-0  00000-0 0  9999", "2 44059  53.0000  0.0000 0001000   0.0000   0.0000 15.00000000    12"),
        ("STARLINK-3", 44061, "1 44061U 19029C   20001.00000000  .00000000  00000-0  00000-0 0  9999", "2 44061  53.0000  0.0000 0001000   0.0000   0.0000 15.00000000    12")
    ]
    for name, norad_id, l1, l2 in mock_sats:
        cur.execute(
            "INSERT OR IGNORE INTO satellites (name, norad_id, tle_line1, tle_line2) VALUES (?, ?, ?, ?)",
            (name, norad_id, l1, l2)
        )
    conn.commit()
    conn.close()

    # Constellation search
    response = await satellites_v1(page=1, limit=5, constellation="Starlink", format="json")
    data = json.loads(response.body.decode())
    assert data["page"] == 1
    assert data["limit"] == 5
    assert len(data["data"]) >= 1
    
    # Verify that all are Starlink satellites
    from satlinksim.infrastructure.api.server import NAMED_CONSTELLATIONS
    starlink_ids = NAMED_CONSTELLATIONS["Starlink"]
    for sat in data["data"]:
        assert sat["norad_id"] in starlink_ids

@pytest.mark.asyncio
async def test_datasets_list_v1():
    data = await datasets_list_v1(page=1, limit=10)
    assert data["page"] == 1
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "link_training_data"

@pytest.mark.asyncio
async def test_dataset_detail_v1():
    detail = await dataset_detail_v1("link_training_data")
    assert detail["dataset_name"] == "link_training_data.parquet"
    assert "columns" in detail
    assert detail["target"] == "link_quality"

@pytest.mark.asyncio
async def test_dataset_download_v1():
    response = await dataset_download_v1("link_training_data")
    assert response.filename == "link_training_data.parquet"
    assert response.media_type == "application/octet-stream"

@pytest.mark.asyncio
async def test_benchmarks_v1():
    response = await benchmarks_v1(format="json")
    data = json.loads(response.body.decode())
    assert "cpu_utilization_pct" in data
    assert "numba_acceleration_ratio" in data
    assert "monte_carlo_speedup_factor" in data

@pytest.mark.asyncio
async def test_validation_endpoints_v1():
    # Individual category validation
    res_physics = await validate_physics(format="json")
    data_phys = json.loads(res_physics.body.decode())
    assert len(data_phys) >= 1
    
    res_itu = await validate_itu(format="json")
    data_itu = json.loads(res_itu.body.decode())
    assert len(data_itu) >= 1
    
    res_nasa = await validate_nasa(format="json")
    data_nasa = json.loads(res_nasa.body.decode())
    assert data_nasa[0]["test_name"] == "NASA GPM Rain Rate Comparison"
    
    res_regression = await validate_regression(format="json")
    data_reg = json.loads(res_regression.body.decode())
    assert data_reg[0]["test_name"] == "API Regression Verification"
    
    # Combined validation
    res_all = await validate_all_v1(format="json")
    data_all = json.loads(res_all.body.decode())
    categories = [x.get("category") for x in data_all]
    assert "physics" in categories
    assert "itu" in categories
    assert "nasa" in categories
    assert "regression" in categories

@pytest.mark.asyncio
async def test_orbit_v1():
    # 1. Orbit geodetic lookup
    data = await query_satellite_orbit("GALAXY 16")
    assert data["satellite"] == "GALAXY 16 (G-16)"
    assert "geodetic" in data
    assert "latitude" in data["geodetic"]
    assert "longitude" in data["geodetic"]
    
    # 2. Orbit propagation
    req = OrbitRequest(
        satellite="GALAXY 16",
        ground_station="Delhi",
        duration=600,
        step=60
    )
    res = await orbit_propagate_v1(req, format="json")
    resp_data = json.loads(res.body.decode())
    assert resp_data["satellite"] == "GALAXY 16 (G-16)"
    assert len(resp_data["elevation"]) == 10

@pytest.mark.asyncio
async def test_stateless_calculators_v1():
    # 1. Link budget
    req_lb = LinkBudgetRequest(
        satellites=["GALAXY 16"],
        ground_station="Delhi",
        frequency=14e9,
        duration=600,
        step=60,
        rain=True,
        handoff=True
    )
    res = await calculator_link_budget(req_lb, format="json")
    data = json.loads(res.body.decode())
    assert "path_loss" in data
    
    # 2. Attenuation
    req_att = AttenuationRequest(
        satellites=["GALAXY 16"],
        ground_station="Delhi",
        frequency=14e9,
        duration=600,
        step=60,
        rain=True,
        handoff=True
    )
    res = await calculator_attenuation(req_att, format="json")
    data = json.loads(res.body.decode())
    assert "total_attenuation" in data
    
    # 3. Availability
    req_av = AvailabilityRequest(
        satellites=["GALAXY 16"],
        ground_station="Delhi",
        frequency=14e9,
        duration=1200,
        step=60,
        rain=True,
        handoff=True,
        snr_threshold=5.0
    )
    res = await calculator_availability(req_av, format="json")
    data = json.loads(res.body.decode())
    assert "availability_fraction" in data

@pytest.mark.asyncio
async def test_rain_and_handoff_decision_v1():
    # 1. Predict rain
    req_pr = PredictRainRequest(
        snr=[15.0, 10.0, 5.0],
        elevation=[30.0, 30.0, 30.0],
        slant_range_km=[38000.0, 38000.0, 38000.0],
        ground_station="Delhi",
        frequency=14e9
    )
    res = await rain_predict_v1(req_pr, format="json")
    data = json.loads(res.body.decode())
    assert "predicted_rain_rate" in data
    
    # 2. Forecast rain
    req_fr = ForecastRainRequest(
        current_rain_rate=5.0,
        ground_station="Delhi",
        steps=5,
        step_size=60.0,
        n_realizations=5
    )
    res = await rain_forecast_v1(req_fr, format="json")
    data = json.loads(res.body.decode())
    assert "mean_forecast" in data
    
    # 3. Handoff decision
    req_ho = LiveHandoffRequest(
        current_satellite="SAT-1",
        candidates_names=["SAT-1", "SAT-2"],
        snr_metrics=[15.0, 20.0],
        el_metrics=[20.0, 25.0],
        dwell_timer=15,
        handoff_policy="highest_snr",
        hysteresis=0.5,
        min_dwell_steps=10
    )
    data = await handoff_decision_v1(req_ho)
    assert data["should_switch"] is True
    assert data["target_satellite"] == "SAT-2"

@pytest.mark.asyncio
async def test_coverage_v1():
    # Local coverage
    req = CoverageRequest(
        satellites=["GALAXY 16"],
        ground_stations=["Delhi", "Tokyo"],
        duration=600,
        step=60,
        min_elevation=10.0
    )
    res = await coverage_v1(req, format="json")
    data = json.loads(res.body.decode())
    assert "Delhi" in data
    assert "Tokyo" in data
    
    # Global coverage
    req_global = GlobalCoverageRequest(
        satellites=["GALAXY 16"],
        duration=600,
        step=60,
        min_elevation=10.0,
        grid_size=45
    )
    res = await global_coverage(req_global, format="json")
    data = json.loads(res.body.decode())
    assert "grid" in data
    assert len(data["grid"]) > 0

@pytest.mark.asyncio
async def test_simulation_resource_lifecycle():
    # 1. Create simulation
    req_pub = PublicSimulationRequest(
        satellites=["GALAXY 16"],
        ground_station="Delhi",
        frequency=14e9,
        duration=600,
        step=60,
        rain=True,
        handoff=True
    )
    sim_meta = await create_simulation(req_pub)
    assert "id" in sim_meta
    assert sim_meta["status"] == "completed"
    
    sim_id = sim_meta["id"]
    
    # 2. List simulations
    res_list = await list_simulations(page=1, limit=10)
    assert res_list["total_items"] >= 1
    assert any(s["id"] == sim_id for s in res_list["simulations"])
    
    # 3. Retrieve simulation detail
    detail = await get_simulation(sim_id)
    assert detail["id"] == sim_id
    assert detail["status"] == "completed"
    
    # 4. Get results
    res_out = await get_simulation_results(sim_id, format="json")
    data_out = json.loads(res_out.body.decode())
    assert "snr" in data_out
    
    # 5. Get attenuation
    res_att = await get_simulation_attenuation(sim_id, format="json")
    data_att = json.loads(res_att.body.decode())
    assert "total_attenuation" in data_att
    
    # 6. Get link budget
    res_lb = await get_simulation_link_budget(sim_id, format="json")
    data_lb = json.loads(res_lb.body.decode())
    assert "rx_power" in data_lb
    
    # 7. Get visibility
    res_vis = await get_simulation_visibility(sim_id, min_elevation=10.0, format="json")
    data_vis = json.loads(res_vis.body.decode())
    assert "satellites" in data_vis
    first_sat = list(data_vis["satellites"].keys())[0]
    assert "visible" in data_vis["satellites"][first_sat]
    
    # 8. Get availability
    res_av = await get_simulation_availability(sim_id, snr_threshold=5.0, format="json")
    data_av = json.loads(res_av.body.decode())
    assert "availability_fraction" in data_av
    
    # 9. Get handoffs
    res_ho = await get_simulation_handoffs(sim_id, format="json")
    data_ho = json.loads(res_ho.body.decode())
    # Should be a list
    assert isinstance(data_ho, list)
    
    # 10. Get orbit
    res_orb = await get_simulation_orbit(sim_id, format="json")
    data_orb = json.loads(res_orb.body.decode())
    assert "elevation" in data_orb
    
    # 11. Delete simulation
    del_res = await delete_simulation(sim_id)
    assert del_res["status"] == "success"
    
    # 12. Retrieve deleted simulation -> raise 404
    with pytest.raises(HTTPException) as exc_info:
        await get_simulation(sim_id)
    assert exc_info.value.status_code == 404

@pytest.mark.asyncio
async def test_jobs_v1():
    # 1. Create job
    req = BatchSimulationRequest(
        satellites=["GALAXY 16"],
        ground_stations=["Delhi", "Tokyo"],
        duration=600,
        step=60,
        rain=True,
        handoff=True
    )
    # mock BackgroundTasks
    from fastapi import BackgroundTasks
    bg = BackgroundTasks()
    job_resp = await create_job(req, bg)
    assert job_resp.job_id is not None
    assert len(job_resp.job_id) > 0
    
    job_id = job_resp.job_id
    
    # Since we bypassed background task execution in pytest by calling background task synchronously,
    # let's run the task function manually to simulate complete execution:
    from satlinksim.infrastructure.api.server import process_batch_job_task
    process_batch_job_task(job_id, req)
    
    # Check status -> completed
    status_resp = await get_job_v1(job_id)
    assert status_resp.status == "completed"
    
    # Get results
    results_resp = await get_job_results_v1(job_id, format="json")
    results = json.loads(results_resp.body.decode())
    assert "Delhi" in results
    assert "Tokyo" in results

@pytest.mark.asyncio
async def test_tle_update_v1():
    req = TleUpdateRequest(groups=["starlink"])
    response = await update_tle_v1(req)
    assert response["status"] == "success"
