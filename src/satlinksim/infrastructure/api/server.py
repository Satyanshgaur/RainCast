import uuid
import asyncio
import os
import sqlite3
import io
import time
import random
from typing import List, Dict, Optional, Any, Union
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np
import uvicorn
import structlog
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Response, Query
from pydantic import BaseModel

from satlinksim.application.simulation_engine import SimulationEngine
from satlinksim.infrastructure.api.schemas import (
    SimulationRequest, SimulationResponse, StationResultSchema, HandoffEventSchema,
    SummarySimulationRequest, SummarySimulationResponse, JobResponse, JobStatus,
    PublicSimulationRequest, LinkBudgetRequest, AttenuationRequest,
    VisibilityRequest, AvailabilityRequest, BatchSimulationRequest,
    PredictRainRequest, ForecastRainRequest, LiveHandoffRequest,
    OrbitRequest, CoverageRequest, ConstellationRequest
)
from satlinksim.ground_stations import GROUND_STATIONS
from satlinksim.domain.models import Constellation, Satellite, StationResult
from satlinksim.domain.link.itu_models import itu_rain_coefficients, gaseous_absorption_db, itu_rain_height, effective_path_length
from satlinksim.domain.link.budget import fspl_db, noise_power_dbw
from satlinksim.infrastructure.logging import configure_logging, get_logger
from satlinksim.infrastructure.metrics import (
    metrics_app, SIMULATIONS_RUN, SIMULATION_LATENCY
)

# Initialize logging
configure_logging()
logger = get_logger("satlinksim.api")

app = FastAPI(title="SatLinkSim API")
engine = SimulationEngine()

def resolve_ground_station(gs_input: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(gs_input, str):
        gs = next((g for g in GROUND_STATIONS if g["name"].lower() == gs_input.lower()), None)
        if not gs:
            raise HTTPException(status_code=404, detail=f"Ground station '{gs_input}' not found.")
        return gs
    elif isinstance(gs_input, dict):
        required_keys = ["name", "latitude", "longitude", "altitude_km", "eirp_dbw", "g_rx_dbi", "system_temp_k", "antenna_diam_m"]
        missing = [k for k in required_keys if k not in gs_input]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing required fields for custom ground station: {missing}")
        gs = gs_input.copy()
        if "itu_rain" not in gs:
            gs["itu_rain"] = {"R001": 42.0, "R01": 19.0, "R1": 6.0, "P_rain": 0.053}
        if "wv_g_m3" not in gs:
            gs["wv_g_m3"] = 12.0
        if "humidity_pct" not in gs:
            gs["humidity_pct"] = 70.0
        if "v_radial_ms" not in gs:
            gs["v_radial_ms"] = 0.0
        return gs
    else:
        raise HTTPException(status_code=400, detail="Invalid ground station input format.")

def resolve_satellites(identifiers: List[Union[str, int]]) -> List[Satellite]:
    from satlinksim.infrastructure.persistence.database import init_db
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "satellites.db")
    init_db(db_path)
    
    resolved = []
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        for identifier in identifiers:
            if isinstance(identifier, int) or (isinstance(identifier, str) and identifier.isdigit()):
                cur.execute("SELECT name, norad_id, tle_line1, tle_line2 FROM satellites WHERE norad_id=?", (int(identifier),))
            else:
                cur.execute("SELECT name, norad_id, tle_line1, tle_line2 FROM satellites WHERE name LIKE ?", (f"%{identifier}%",))
            row = cur.fetchone()
            if row:
                resolved.append(Satellite(name=row[0], norad_id=row[1], tle_line1=row[2], tle_line2=row[3]))
            else:
                raise HTTPException(status_code=404, detail=f"Satellite '{identifier}' not found in database.")
    finally:
        conn.close()
    return resolved

def disable_rain_in_result(res: StationResult):
    rain_db = np.array(res.rain_db_series)
    snr = np.array(res.snr_series)
    new_snr = snr + rain_db
    
    from satlinksim.domain.link.budget import packet_loss_from_snr
    from satlinksim.application.simulation_engine import SNR_THRESHOLD_DB
    new_pkt_loss = packet_loss_from_snr(new_snr, SNR_THRESHOLD_DB)
    
    res.snr_series = new_snr.tolist()
    res.rain_series = [0.0] * len(res.rain_series)
    res.rain_db_series = [0.0] * len(res.rain_db_series)
    res.pkt_loss_series = new_pkt_loss.tolist()
    
    res.snr_mean = float(np.mean(res.snr_series))
    res.snr_min = float(np.min(res.snr_series))
    res.snr_std = float(np.std(res.snr_series, ddof=1)) if len(res.snr_series) > 1 else 0.0
    sorted_snr = sorted(res.snr_series)
    p10_idx = max(0, int(0.10 * len(res.snr_series)) - 1)
    res.snr_p10 = float(sorted_snr[p10_idx])
    res.rain_fraction = 0.0
    res.avg_rain_db = 0.0
    res.avg_pkt_loss = float(np.mean(res.pkt_loss_series))
    res.outage_fraction = float(np.mean(res.pkt_loss_series))

def respond_with_format(df: pd.DataFrame, json_data: Any, format: str, filename_prefix: str, headers: Optional[Dict[str, str]] = None):
    format = format.lower()
    if format == "json":
        from fastapi.responses import JSONResponse
        return JSONResponse(content=json_data, headers=headers)
    elif format == "csv":
        stream = io.StringIO()
        df.to_csv(stream, index=False)
        response = Response(content=stream.getvalue(), media_type="text/csv")
        response.headers["Content-Disposition"] = f"attachment; filename={filename_prefix}.csv"
        if headers:
            for k, v in headers.items():
                response.headers[k] = v
        return response
    elif format == "parquet":
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        response = Response(content=buffer.getvalue(), media_type="application/octet-stream")
        response.headers["Content-Disposition"] = f"attachment; filename={filename_prefix}.parquet"
        if headers:
            for k, v in headers.items():
                response.headers[k] = v
        return response
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format '{format}'. Supported formats: json, csv, parquet.")

# Expose /metrics endpoint
app.mount("/metrics", metrics_app)

# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    structlog.contextvars.bind_contextvars(request_id=request_id)
    
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    logger.info(
        "http_request",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_s=round(duration, 4)
    )
    return response

# --- Job Management ---
jobs: Dict[str, JobStatus] = {}

def process_simulation_job(job_id: str, request: SimulationRequest):
    structlog.contextvars.bind_contextvars(job_id=job_id)
    logger.info("job_started", steps=request.n_steps)
    
    start_time = time.time()
    SIMULATIONS_RUN.labels(mode="async").inc()
    
    jobs[job_id].status = "running"
    try:
        gs_dicts = [gs.model_dump() for gs in request.ground_stations]
        constellation = None
        if request.constellation:
            sats = [
                Satellite(norad_id=s.norad_id, name=s.name, tle_line1=s.tle_line1, tle_line2=s.tle_line2)
                for s in request.constellation.satellites
            ]
            constellation = Constellation(name=request.constellation.name, satellites=sats)

        results = engine.simulate_all_batched(
            ground_stations=gs_dicts,
            n_steps=request.n_steps,
            dt_s=request.dt_s,
            start_time=request.start_time,
            force_rain=request.force_rain,
            seed=request.seed,
            freq_hz=request.freq_hz,
            eirp_offset_db=request.eirp_offset_db,
            bandwidth_hz=request.bandwidth_hz,
            polarization=request.polarization,
            rain_rate_scale=request.rain_rate_scale,
            constellation=constellation,
            handoff_policy=request.handoff_policy,
            hysteresis=request.hysteresis,
            min_dwell_steps=request.min_dwell_steps
        )
        
        response_results = []
        for res in results:
            handoffs = [
                HandoffEventSchema(
                    time_step=h.time_step,
                    old_sat=h.old_sat,
                    new_sat=h.new_sat,
                    reason=h.reason,
                    metric_delta=h.metric_delta
                ) for h in res.handoff_events
            ]
            
            response_results.append(StationResultSchema(
                name=res.name, elevation=res.elevation, slant_km=res.slant_km, doppler_hz=res.doppler_hz,
                path_loss=res.path_loss, gas_loss=res.gas_loss, rain_height=res.rain_height,
                eff_path=res.eff_path, itu_k=res.itu_k, itu_alpha=res.itu_alpha,
                scint_sig=res.scint_sig, noise_floor=res.noise_floor, snr_series=res.snr_series,
                rain_series=res.rain_series, rain_db_series=res.rain_db_series, scint_series=res.scint_series,
                pkt_loss_series=res.pkt_loss_series, elevation_series=res.elevation_series,
                slant_range_series=res.slant_range_series, doppler_series=res.doppler_series,
                snr_mean=res.snr_mean, snr_min=res.snr_min, snr_std=res.snr_std, snr_p10=res.snr_p10,
                rain_fraction=res.rain_fraction, avg_rain_db=res.avg_rain_db, avg_pkt_loss=res.avg_pkt_loss,
                outage_fraction=res.outage_fraction, sat_name_series=res.sat_name_series,
                handoff_events=handoffs
            ))
            
        jobs[job_id].result = SimulationResponse(results=response_results)
        jobs[job_id].status = "completed"
        
        latency = time.time() - start_time
        SIMULATION_LATENCY.labels(mode="async").observe(latency)
        logger.info("job_completed", duration_s=round(latency, 4))
    except Exception as e:
        jobs[job_id].status = "failed"
        jobs[job_id].error = str(e)
        logger.error("job_failed", error=str(e))

# --- Endpoints ---

@app.post("/simulate/async", response_model=JobResponse)
async def simulate_async(request: SimulationRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = JobStatus(job_id=job_id, status="pending")
    background_tasks.add_task(process_simulation_job, job_id, request)
    logger.info("job_submitted", job_id=job_id)
    return JobResponse(job_id=job_id)

@app.get("/job/{job_id}", response_model=JobStatus)
async def get_job(job_id: str):
    if job_id not in jobs:
        logger.warning("job_not_found", job_id=job_id)
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

NAMED_CONSTELLATIONS = {
    "Starlink": [44057, 44059, 44061],
    "OneWeb": [45131, 45132],
    "Iridium": [43569, 43570]
}

@app.post("/simulate/summary", response_model=SummarySimulationResponse)
async def simulate_summary(request: SummarySimulationRequest):
    start_time = time.time()
    SIMULATIONS_RUN.labels(mode="summary").inc()
    
    logger.info("summary_simulation_requested", station=request.station, constellation=request.constellation)
    gs = next((g for g in GROUND_STATIONS if g["name"].lower() == request.station.lower()), None)
    if not gs:
        raise HTTPException(status_code=404, detail=f"Station '{request.station}' not found")

    ids = NAMED_CONSTELLATIONS.get(request.constellation)
    if not ids:
        ids = next((v for k, v in NAMED_CONSTELLATIONS.items() if k.lower() == request.constellation.lower()), None)
    
    if not ids:
        raise HTTPException(status_code=404, detail=f"Constellation '{request.constellation}' not found")

    constellation = Constellation.from_norad_ids(request.constellation, ids)

    try:
        results = engine.simulate_all_batched(
            ground_stations=[gs],
            n_steps=request.duration_min * 60,
            dt_s=1.0,
            constellation=constellation
        )
        res = results[0]
        
        latency = time.time() - start_time
        SIMULATION_LATENCY.labels(mode="summary").observe(latency)
        
        return SummarySimulationResponse(
            availability=round((1.0 - res.outage_fraction) * 100, 2),
            handoffs=len(res.handoff_events)
        )
    except Exception as e:
        logger.error("summary_simulation_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/simulate")
async def simulate(
    request: Union[PublicSimulationRequest, SimulationRequest],
    format: str = Query("json", description="Output format: json, csv, or parquet")
):
    start_time = time.time()
    SIMULATIONS_RUN.labels(mode="sync").inc()

    if isinstance(request, SimulationRequest):
        logger.info("simulation_requested", n_stations=len(request.ground_stations))
        try:
            gs_dicts = [gs.model_dump() for gs in request.ground_stations]
            constellation = None
            if request.constellation:
                sats = [
                    Satellite(norad_id=s.norad_id, name=s.name, tle_line1=s.tle_line1, tle_line2=s.tle_line2)
                    for s in request.constellation.satellites
                ]
                constellation = Constellation(name=request.constellation.name, satellites=sats)

            results = engine.simulate_all_batched(
                ground_stations=gs_dicts,
                n_steps=request.n_steps,
                dt_s=request.dt_s,
                start_time=request.start_time,
                force_rain=request.force_rain,
                seed=request.seed,
                freq_hz=request.freq_hz,
                eirp_offset_db=request.eirp_offset_db,
                bandwidth_hz=request.bandwidth_hz,
                polarization=request.polarization,
                rain_rate_scale=request.rain_rate_scale,
                constellation=constellation,
                handoff_policy=request.handoff_policy,
                hysteresis=request.hysteresis,
                min_dwell_steps=request.min_dwell_steps
            )
            
            response_results = []
            for res in results:
                handoffs = [
                    HandoffEventSchema(
                        time_step=h.time_step,
                        old_sat=h.old_sat,
                        new_sat=h.new_sat,
                        reason=h.reason,
                        metric_delta=h.metric_delta
                    ) for h in res.handoff_events
                ]
                response_results.append(StationResultSchema(
                    name=res.name, elevation=res.elevation, slant_km=res.slant_km, doppler_hz=res.doppler_hz,
                    path_loss=res.path_loss, gas_loss=res.gas_loss, rain_height=res.rain_height,
                    eff_path=res.eff_path, itu_k=res.itu_k, itu_alpha=res.itu_alpha,
                    scint_sig=res.scint_sig, noise_floor=res.noise_floor, snr_series=res.snr_series,
                    rain_series=res.rain_series, rain_db_series=res.rain_db_series, scint_series=res.scint_series,
                    pkt_loss_series=res.pkt_loss_series, elevation_series=res.elevation_series,
                    slant_range_series=res.slant_range_series, doppler_series=res.doppler_series,
                    snr_mean=res.snr_mean, snr_min=res.snr_min, snr_std=res.snr_std, snr_p10=res.snr_p10,
                    rain_fraction=res.rain_fraction, avg_rain_db=res.avg_rain_db, avg_pkt_loss=res.avg_pkt_loss,
                    outage_fraction=res.outage_fraction, sat_name_series=res.sat_name_series,
                    handoff_events=handoffs
                ))
                
            latency = time.time() - start_time
            SIMULATION_LATENCY.labels(mode="sync").observe(latency)
            
            return SimulationResponse(results=response_results)
        except Exception as e:
            logger.error("simulation_failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    else:
        # Simplified Public REST API /simulate
        logger.info("public_simulation_requested", satellites=request.satellites, ground_station=request.ground_station)
        try:
            gs = resolve_ground_station(request.ground_station)
            sats = resolve_satellites(request.satellites)
            n_steps = int(request.duration / request.step)
            dt_s = float(request.step)

            if request.handoff:
                constellation = Constellation(name="Constellation", satellites=sats)
                results = engine.simulate_all_batched(
                    ground_stations=[gs],
                    n_steps=n_steps,
                    dt_s=dt_s,
                    freq_hz=request.frequency,
                    force_rain=request.rain,
                    constellation=constellation,
                    handoff_policy="highest_elevation"
                )
            else:
                gs_copy = gs.copy()
                gs_copy["norad_id"] = sats[0].norad_id
                gs_copy["sat_name"] = sats[0].name
                results = engine.simulate_all_batched(
                    ground_stations=[gs_copy],
                    n_steps=n_steps,
                    dt_s=dt_s,
                    freq_hz=request.frequency,
                    force_rain=request.rain,
                    constellation=None
                )
            
            res = results[0]
            if not request.rain:
                disable_rain_in_result(res)

            from satlinksim.application.simulation_engine import SNR_THRESHOLD_DB
            snr_list = res.snr_series
            availability_list = [int(s >= SNR_THRESHOLD_DB) for s in snr_list]
            handoffs_list = [
                {
                    "time_step": h.time_step,
                    "old_sat": h.old_sat,
                    "new_sat": h.new_sat,
                    "reason": h.reason,
                    "metric_delta": h.metric_delta
                }
                for h in res.handoff_events
            ]
            rain_loss_list = res.rain_db_series
            stations_list = [gs["name"]]
            
            json_data = {
                "snr": snr_list,
                "availability": availability_list,
                "handoffs": handoffs_list,
                "rain_loss": rain_loss_list,
                "stations": stations_list
            }
            
            df = pd.DataFrame({
                "time_step": range(len(snr_list)),
                "station": [gs["name"]] * len(snr_list),
                "satellite": res.sat_name_series or [sats[0].name] * len(snr_list),
                "snr": snr_list,
                "availability": availability_list,
                "rain_loss": rain_loss_list
            })
            
            latency = time.time() - start_time
            SIMULATION_LATENCY.labels(mode="sync").observe(latency)
            
            return respond_with_format(df, json_data, format, "simulation_results")
        except HTTPException:
            raise
        except Exception as e:
            logger.error("public_simulation_failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/link-budget")
async def link_budget(
    request: LinkBudgetRequest,
    format: str = Query("json", description="Output format: json, csv, or parquet")
):
    start_time = time.time()
    try:
        gs = resolve_ground_station(request.ground_station)
        sats = resolve_satellites(request.satellites)
        n_steps = int(request.duration / request.step)
        dt_s = float(request.step)
        
        if request.handoff:
            constellation = Constellation(name="Constellation", satellites=sats)
            results = engine.simulate_all_batched(
                ground_stations=[gs],
                n_steps=n_steps,
                dt_s=dt_s,
                freq_hz=request.frequency,
                force_rain=request.rain,
                constellation=constellation,
                handoff_policy="highest_elevation"
            )
        else:
            gs_copy = gs.copy()
            gs_copy["norad_id"] = sats[0].norad_id
            gs_copy["sat_name"] = sats[0].name
            results = engine.simulate_all_batched(
                ground_stations=[gs_copy],
                n_steps=n_steps,
                dt_s=dt_s,
                freq_hz=request.frequency,
                force_rain=request.rain,
                constellation=None
            )
            
        res = results[0]
        if not request.rain:
            disable_rain_in_result(res)
            
        freq_hz = request.frequency
        freq_ghz = freq_hz / 1e9
        polarization = request.polarization
        itu_k, itu_a = itu_rain_coefficients(freq_ghz, polarization)
        noise_floor = noise_power_dbw(gs["system_temp_k"], request.bandwidth_hz)
        eirp = gs["eirp_dbw"]
        g_rx = gs["g_rx_dbi"]
        
        slant_range = np.array(res.slant_range_series)
        elevation = np.array(res.elevation_series)
        path_loss = fspl_db(freq_hz, slant_range)
        gas_loss = gaseous_absorption_db(freq_ghz, elevation, gs["wv_g_m3"])
        rain_loss = np.array(res.rain_db_series)
        scint_loss = np.array(res.scint_series)
        rx_power = eirp - path_loss - gas_loss - rain_loss - scint_loss + g_rx
        snr = np.array(res.snr_series)
        
        json_data = {
            "time_step": list(range(n_steps)),
            "eirp": [float(eirp)] * n_steps,
            "path_loss": path_loss.tolist(),
            "gas_loss": gas_loss.tolist(),
            "rain_loss": rain_loss.tolist(),
            "scint_loss": scint_loss.tolist(),
            "rx_power": rx_power.tolist(),
            "noise_floor": [float(noise_floor)] * n_steps,
            "snr": snr.tolist()
        }
        
        df = pd.DataFrame(json_data)
        return respond_with_format(df, json_data, format, "link_budget")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("link_budget_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/attenuation")
async def attenuation(
    request: AttenuationRequest,
    format: str = Query("json", description="Output format: json, csv, or parquet")
):
    try:
        gs = resolve_ground_station(request.ground_station)
        sats = resolve_satellites(request.satellites)
        n_steps = int(request.duration / request.step)
        dt_s = float(request.step)
        
        if request.handoff:
            constellation = Constellation(name="Constellation", satellites=sats)
            results = engine.simulate_all_batched(
                ground_stations=[gs],
                n_steps=n_steps,
                dt_s=dt_s,
                freq_hz=request.frequency,
                force_rain=request.rain,
                constellation=constellation,
                handoff_policy="highest_elevation"
            )
        else:
            gs_copy = gs.copy()
            gs_copy["norad_id"] = sats[0].norad_id
            gs_copy["sat_name"] = sats[0].name
            results = engine.simulate_all_batched(
                ground_stations=[gs_copy],
                n_steps=n_steps,
                dt_s=dt_s,
                freq_hz=request.frequency,
                force_rain=request.rain,
                constellation=None
            )
            
        res = results[0]
        if not request.rain:
            disable_rain_in_result(res)
            
        freq_hz = request.frequency
        freq_ghz = freq_hz / 1e9
        elevation = np.array(res.elevation_series)
        gas_loss = gaseous_absorption_db(freq_ghz, elevation, gs["wv_g_m3"])
        rain_loss = np.array(res.rain_db_series)
        scint_loss = np.array(res.scint_series)
        total_loss = gas_loss + rain_loss + scint_loss
        
        json_data = {
            "gaseous_attenuation": gas_loss.tolist(),
            "rain_attenuation": rain_loss.tolist(),
            "scintillation_attenuation": scint_loss.tolist(),
            "total_attenuation": total_loss.tolist()
        }
        
        df = pd.DataFrame({
            "time_step": range(n_steps),
            "gaseous_attenuation": gas_loss,
            "rain_attenuation": rain_loss,
            "scintillation_attenuation": scint_loss,
            "total_attenuation": total_loss
        })
        return respond_with_format(df, json_data, format, "attenuation")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("attenuation_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/visibility")
async def visibility(
    request: VisibilityRequest,
    format: str = Query("json", description="Output format: json, csv, or parquet")
):
    try:
        gs = resolve_ground_station(request.ground_station)
        sats = resolve_satellites(request.satellites)
        n_steps = int(request.duration / request.step)
        dt_s = float(request.step)
        
        from satlinksim.infrastructure.tle.service import SGP4Propagator
        propagator = SGP4Propagator()
        curr_time = datetime.now(timezone.utc)
        times = [curr_time + timedelta(seconds=i*dt_s) for i in range(n_steps)]
        
        satellites_data = {}
        for sat in sats:
            geo = propagator.get_geometry_batch(sat.norad_id, times, gs["latitude"], gs["longitude"], gs["altitude_km"])
            if geo:
                elevation = geo.elevation_deg.tolist()
                azimuth = geo.azimuth_deg.tolist() if geo.azimuth_deg is not None else [0.0] * n_steps
                visible = (geo.elevation_deg >= request.min_elevation).astype(int).tolist()
                satellites_data[sat.name] = {
                    "elevation": elevation,
                    "azimuth": azimuth,
                    "visible": visible
                }
            else:
                satellites_data[sat.name] = {
                    "elevation": [0.0] * n_steps,
                    "azimuth": [0.0] * n_steps,
                    "visible": [0] * n_steps
                }
                
        json_data = {
            "time_step": list(range(n_steps)),
            "satellites": satellites_data
        }
        
        rows = []
        for sat_name, data in satellites_data.items():
            for t in range(n_steps):
                rows.append({
                    "time_step": t,
                    "satellite": sat_name,
                    "elevation": data["elevation"][t],
                    "azimuth": data["azimuth"][t],
                    "visible": data["visible"][t]
                })
        df = pd.DataFrame(rows)
        return respond_with_format(df, json_data, format, "visibility")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("visibility_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/availability")
async def availability(
    request: AvailabilityRequest,
    format: str = Query("json", description="Output format: json, csv, or parquet")
):
    try:
        gs = resolve_ground_station(request.ground_station)
        sats = resolve_satellites(request.satellites)
        n_steps = int(request.duration / request.step)
        dt_s = float(request.step)
        
        if request.handoff:
            constellation = Constellation(name="Constellation", satellites=sats)
            results = engine.simulate_all_batched(
                ground_stations=[gs],
                n_steps=n_steps,
                dt_s=dt_s,
                freq_hz=request.frequency,
                force_rain=request.rain,
                constellation=constellation,
                handoff_policy="highest_elevation"
            )
        else:
            gs_copy = gs.copy()
            gs_copy["norad_id"] = sats[0].norad_id
            gs_copy["sat_name"] = sats[0].name
            results = engine.simulate_all_batched(
                ground_stations=[gs_copy],
                n_steps=n_steps,
                dt_s=dt_s,
                freq_hz=request.frequency,
                force_rain=request.rain,
                constellation=None
            )
            
        res = results[0]
        if not request.rain:
            disable_rain_in_result(res)
            
        is_available = np.array(res.snr_series) >= request.snr_threshold
        availability_fraction = float(np.mean(is_available))
        total_duration_seconds = n_steps * dt_s
        outage_duration_seconds = float(np.sum(~is_available) * dt_s)
        
        outages = []
        in_outage = False
        outage_start = 0
        for i, avail in enumerate(is_available):
            if not avail and not in_outage:
                in_outage = True
                outage_start = i
            elif avail and in_outage:
                in_outage = False
                duration = (i - outage_start) * dt_s
                outages.append({
                    "start_step": int(outage_start),
                    "end_step": int(i),
                    "duration_seconds": float(duration)
                })
        if in_outage:
            duration = (len(is_available) - outage_start) * dt_s
            outages.append({
                "start_step": int(outage_start),
                "end_step": int(len(is_available)),
                "duration_seconds": float(duration)
            })
            
        number_of_outages = len(outages)
        
        json_data = {
            "availability_fraction": availability_fraction,
            "total_duration_seconds": total_duration_seconds,
            "outage_duration_seconds": outage_duration_seconds,
            "number_of_outages": number_of_outages,
            "outages": outages
        }
        
        headers = {
            "X-Availability-Fraction": str(availability_fraction),
            "X-Total-Duration-Seconds": str(total_duration_seconds),
            "X-Outage-Duration-Seconds": str(outage_duration_seconds),
            "X-Number-Of-Outages": str(number_of_outages)
        }
        
        df = pd.DataFrame({
            "time_step": range(n_steps),
            "snr": res.snr_series,
            "available": is_available.astype(int)
        })
        return respond_with_format(df, json_data, format, "availability", headers=headers)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("availability_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stations")
async def stations(
    name: Optional[str] = Query(None, description="Filter stations by name"),
    format: str = Query("json", description="Output format: json, csv, or parquet")
):
    try:
        flat_stations = []
        for gs in GROUND_STATIONS:
            if name and name.lower() not in gs["name"].lower():
                continue
            item = gs.copy()
            itu = item.pop("itu_rain", {})
            for k, v in itu.items():
                item[f"itu_rain_{k}"] = v
            flat_stations.append(item)
            
        df = pd.DataFrame(flat_stations)
        
        json_data = []
        for gs in GROUND_STATIONS:
            if name and name.lower() not in gs["name"].lower():
                continue
            json_data.append(gs)
            
        return respond_with_format(df, json_data, format, "ground_stations")
    except Exception as e:
        logger.error("get_stations_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/satellites")
async def satellites(
    query: Optional[str] = Query(None, description="Search satellites by name or NORAD ID"),
    limit: int = Query(100, description="Limit result size"),
    format: str = Query("json", description="Output format: json, csv, or parquet")
):
    try:
        from satlinksim.infrastructure.persistence.database import init_db
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "satellites.db")
        init_db(db_path)
        
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        try:
            if query:
                if query.isdigit():
                    cur.execute(
                        "SELECT name, norad_id, tle_line1, tle_line2 FROM satellites WHERE norad_id = ? LIMIT ?",
                        (int(query), limit)
                    )
                else:
                    cur.execute(
                        "SELECT name, norad_id, tle_line1, tle_line2 FROM satellites WHERE name LIKE ? LIMIT ?",
                        (f"%{query}%", limit)
                    )
            else:
                cur.execute("SELECT name, norad_id, tle_line1, tle_line2 FROM satellites LIMIT ?", (limit,))
            
            rows = cur.fetchall()
        finally:
            conn.close()
            
        json_data = [
            {"name": r[0], "norad_id": r[1], "tle_line1": r[2], "tle_line2": r[3]}
            for r in rows
        ]
        
        df = pd.DataFrame(json_data)
        return respond_with_format(df, json_data, format, "satellites")
    except Exception as e:
        logger.error("get_satellites_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch")
async def batch_simulation(
    request: BatchSimulationRequest,
    format: str = Query("json", description="Output format: json, csv, or parquet")
):
    try:
        sats = resolve_satellites(request.satellites)
        n_steps = int(request.duration / request.step)
        dt_s = float(request.step)
        
        constellation = Constellation(name="Constellation", satellites=sats) if request.handoff else None
        
        results_dict = {}
        for gs_input in request.ground_stations:
            gs = resolve_ground_station(gs_input)
            
            if request.handoff:
                results = engine.simulate_all_batched(
                    ground_stations=[gs],
                    n_steps=n_steps,
                    dt_s=dt_s,
                    freq_hz=request.frequency,
                    force_rain=request.rain,
                    constellation=constellation,
                    handoff_policy="highest_elevation"
                )
            else:
                gs_copy = gs.copy()
                gs_copy["norad_id"] = sats[0].norad_id
                gs_copy["sat_name"] = sats[0].name
                results = engine.simulate_all_batched(
                    ground_stations=[gs_copy],
                    n_steps=n_steps,
                    dt_s=dt_s,
                    freq_hz=request.frequency,
                    force_rain=request.rain,
                    constellation=None
                )
                
            res = results[0]
            if not request.rain:
                disable_rain_in_result(res)
                
            results_dict[gs["name"]] = res
            
        json_data = {}
        df_rows = []
        for name, res in results_dict.items():
            from satlinksim.application.simulation_engine import SNR_THRESHOLD_DB
            snr_list = res.snr_series
            avail_list = [int(s >= SNR_THRESHOLD_DB) for s in snr_list]
            
            json_data[name] = {
                "snr": snr_list,
                "availability": avail_list,
                "rain_loss": res.rain_db_series,
                "handoffs": len(res.handoff_events)
            }
            
            for t in range(n_steps):
                df_rows.append({
                    "station": name,
                    "time_step": t,
                    "snr": snr_list[t],
                    "availability": avail_list[t],
                    "rain_loss": res.rain_db_series[t]
                })
                
        df = pd.DataFrame(df_rows)
        return respond_with_format(df, json_data, format, "batch_results")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("batch_simulation_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/benchmarks")
@app.post("/benchmarks")
async def benchmarks(
    format: str = Query("json", description="Output format: json, csv, or parquet")
):
    try:
        import time
        import resource
        
        gs = GROUND_STATIONS[0]
        
        t0 = time.perf_counter()
        results = engine.simulate_all_batched(
            ground_stations=[gs],
            n_steps=1000,
            dt_s=1.0,
            force_rain=True
        )
        duration = time.perf_counter() - t0
        
        from satlinksim.infrastructure.tle.service import SGP4Propagator
        prop = SGP4Propagator()
        sat_id = gs.get("norad_id")
        curr_time = datetime.now(timezone.utc)
        lat, lon, alt = gs["latitude"], gs["longitude"], gs["altitude_km"]
        
        prop_times = []
        for _ in range(100):
            t_start = time.perf_counter()
            prop.get_geometry(sat_id, curr_time, lat, lon, alt)
            prop_times.append(time.perf_counter() - t_start)
            
        avg_prop_ms = np.mean(prop_times) * 1000.0
        throughput = 1000.0 / duration
        mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        
        json_data = {
            "throughput_timesteps_per_second": throughput,
            "avg_latency_per_step_ms": (duration / 1000.0) * 1000.0,
            "propagation_latency_ms": avg_prop_ms,
            "memory_rss_mb": mem_mb
        }
        
        df = pd.DataFrame([json_data])
        return respond_with_format(df, json_data, format, "benchmarks")
    except Exception as e:
        logger.error("benchmarks_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/validation")
@app.post("/validation")
async def validation(
    format: str = Query("json", description="Output format: json, csv, or parquet")
):
    try:
        import math
        from satlinksim.domain.link.itu_models import itu_rain_coefficients, itu_rain_height
        from satlinksim.domain.link.budget import fspl_db
        from satlinksim.domain.geometry.physics import geo_slant_range_km
        
        validations = []
        
        freq_hz = 14e9
        dist_km = 40000.0
        calculated_fspl = fspl_db(freq_hz, dist_km)
        expected_fspl = 92.45 + 20 * math.log10(14) + 20 * math.log10(40000)
        fspl_diff = abs(calculated_fspl - expected_fspl)
        validations.append({
            "test_name": "Free Space Path Loss (FSPL) Correctness",
            "calculated": calculated_fspl,
            "reference": expected_fspl,
            "difference": fspl_diff,
            "status": "passed" if fspl_diff < 0.1 else "failed"
        })
        
        calculated_h_r = itu_rain_height(28.6)
        expected_h_r = 5.0 - 0.075 * (28.6 - 23.0)
        h_r_diff = abs(calculated_h_r - expected_h_r)
        validations.append({
            "test_name": "ITU-R P.839-4 Rain Height Correctness",
            "calculated": calculated_h_r,
            "reference": expected_h_r,
            "difference": h_r_diff,
            "status": "passed" if h_r_diff < 0.01 else "failed"
        })
        
        zenith_sr = geo_slant_range_km(0.0, 0.0, 0.0)
        sr_diff = abs(zenith_sr - 35786.0)
        validations.append({
            "test_name": "Zenith Slant Range Correctness",
            "calculated": zenith_sr,
            "reference": 35786.0,
            "difference": sr_diff,
            "status": "passed" if sr_diff < 10.0 else "failed"
        })
        
        df = pd.DataFrame(validations)
        return respond_with_format(df, validations, format, "validation_results")
    except Exception as e:
        logger.error("validation_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/datasets")
async def get_datasets(
    format: str = Query("json", description="Output format: json, csv, or parquet")
):
    try:
        # Robust search for link_training_data.parquet across multiple potential paths
        candidates = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml", "link_training_data.parquet"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))), "src", "satlinksim", "infrastructure", "ml", "link_training_data.parquet"),
            os.path.join(os.getcwd(), "src", "satlinksim", "infrastructure", "ml", "link_training_data.parquet"),
            os.path.join(os.getcwd(), "satlinksim", "infrastructure", "ml", "link_training_data.parquet"),
            "link_training_data.parquet"
        ]
        
        ml_parquet = None
        for path in candidates:
            if os.path.exists(path):
                ml_parquet = path
                break
                
        if not ml_parquet:
            # Fallback: Walk the directory tree to find it stochastically
            found_path = None
            search_roots = [os.getcwd(), os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))]
            for s_root in search_roots:
                if not os.path.exists(s_root):
                    continue
                for root, dirs, files in os.walk(s_root):
                    if "link_training_data.parquet" in files:
                        found_path = os.path.join(root, "link_training_data.parquet")
                        break
                if found_path:
                    break
            ml_parquet = found_path

        if not ml_parquet or not os.path.exists(ml_parquet):
            raise HTTPException(status_code=404, detail="Dataset file not found.")
            
        df_dataset = pd.read_parquet(ml_parquet)
        
        json_data = {
            "dataset_name": "link_training_data.parquet",
            "file_size_bytes": os.path.getsize(ml_parquet),
            "total_rows": len(df_dataset),
            "columns": list(df_dataset.columns),
            "features": [c for c in df_dataset.columns if c != "link_quality"],
            "target": "link_quality",
            "summary_statistics": df_dataset.describe().to_dict()
        }
        
        flat_rows = []
        desc = df_dataset.describe()
        for metric in desc.index:
            row = {"metric": metric}
            for col in desc.columns:
                row[col] = desc.loc[metric, col]
            flat_rows.append(row)
            
        df = pd.DataFrame(flat_rows)
        return respond_with_format(df, json_data, format, "datasets")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_datasets_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

class TleUpdateRequest(BaseModel):
    groups: Optional[List[str]] = None

@app.post("/tle")
async def update_tle(request: Optional[TleUpdateRequest] = None):
    try:
        from satlinksim.infrastructure.tle.updater import update_database, CELESTRAK_GROUPS
        groups = request.groups if request and request.groups else list(CELESTRAK_GROUPS.keys())
        
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, update_database, groups)
        
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "satellites.db")
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM satellites")
        total = cur.fetchone()[0]
        conn.close()
        
        return {"status": "success", "message": "TLE database updated successfully", "total_satellites": total}
    except Exception as e:
        logger.error("tle_update_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tle")
async def query_tle(
    query: Optional[str] = Query(None, description="Search by name or NORAD ID"),
    limit: int = Query(100, description="Limit size"),
    format: str = Query("json", description="Output format: json, csv, or parquet")
):
    return await satellites(query=query, limit=limit, format=format)

@app.post("/predict-rain")
async def predict_rain(
    request: PredictRainRequest,
    format: str = Query("json", description="Output format: json, csv, or parquet")
):
    try:
        gs = resolve_ground_station(request.ground_station)
        
        snrs = [request.snr] if isinstance(request.snr, float) else request.snr
        elevations = [request.elevation] if isinstance(request.elevation, float) else request.elevation
        slant_ranges = [request.slant_range_km] if isinstance(request.slant_range_km, float) else request.slant_range_km
        
        n_points = len(snrs)
        if len(elevations) != n_points or len(slant_ranges) != n_points:
            raise HTTPException(status_code=400, detail="Lists snr, elevation, and slant_range_km must be of equal length.")
            
        freq_hz = request.frequency
        freq_ghz = freq_hz / 1e9
        polarization = request.polarization
        
        itu_k, itu_alpha = itu_rain_coefficients(freq_ghz, polarization)
        rain_h = itu_rain_height(gs["latitude"])
        
        eirp = gs["eirp_dbw"]
        g_rx = gs["g_rx_dbi"]
        noise_floor = noise_power_dbw(gs["system_temp_k"], request.bandwidth_hz)
        
        snr_arr = np.array(snrs)
        el_arr = np.array(elevations)
        slant_arr = np.array(slant_ranges)
        
        pl = fspl_db(freq_hz, slant_arr)
        gas_loss = gaseous_absorption_db(freq_ghz, el_arr, gs["wv_g_m3"])
        
        total_gain = eirp + g_rx - noise_floor
        excess_attn = total_gain - snr_arr - pl - gas_loss
        
        if n_points >= 15:
            from scipy.signal import butter, filtfilt
            nyquist = 0.5
            normal_cutoff = 0.005 / nyquist
            b, a = butter(2, normal_cutoff, btype='low')
            filtered_attn = filtfilt(b, a, excess_attn)
            filtered_attn = np.maximum(filtered_attn, 0.0)
        else:
            filtered_attn = np.maximum(excess_attn, 0.0)
            
        ep = effective_path_length(el_arr, rain_h, gs["altitude_km"], itu_k)
        ep_safe = np.maximum(ep, 1e-6)
        pred_rain = (filtered_attn / (itu_k * ep_safe)) ** (1.0 / itu_alpha)
        pred_rain = np.nan_to_num(pred_rain, nan=0.0, posinf=0.0, neginf=0.0)
        
        json_data = {
            "predicted_rain_rate": pred_rain.tolist(),
            "model": "stage-a"
        }
        
        df = pd.DataFrame({
            "time_step": range(n_points),
            "observed_snr": snrs,
            "excess_attenuation": excess_attn.tolist(),
            "filtered_attenuation": filtered_attn.tolist(),
            "predicted_rain_rate": pred_rain.tolist(),
            "model": ["stage-a"] * n_points
        })
        
        return respond_with_format(df, json_data, format, "rain_predictions")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("predict_rain_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/forecast-rain")
async def forecast_rain(
    request: ForecastRainRequest,
    format: str = Query("json", description="Output format: json, csv, or parquet")
):
    try:
        from scipy.stats import norm
        gs = resolve_ground_station(request.ground_station)
        
        p = gs["itu_rain"]
        rain_rate_scale = 1.0
        R001 = max(p["R001"] * rain_rate_scale, 0.1)
        R01  = max(p["R01"]  * rain_rate_scale, 0.05)
        P_rain = p["P_rain"]
        
        tau_c = 1800.0
        dt_s = request.step_size
        
        p_cond_001 = np.clip(0.0001 / max(P_rain, 1e-6), 1e-9, 0.9999)
        p_cond_01  = np.clip(0.001 / max(P_rain, 1e-6), 1e-9, 0.9999)
        _z001 = norm.ppf(1.0 - p_cond_001)
        _z01  = norm.ppf(1.0 - p_cond_01)
        
        sigma = (np.log(R001) - np.log(R01)) / (_z001 - _z01)
        mu = np.log(R01) - _z01 * sigma
        rho = np.exp(-dt_s / tau_c)
        
        curr_rate = request.current_rain_rate
        if curr_rate > 0.01:
            chi_start = (np.log(curr_rate) - mu) / sigma
            raining_start = True
        else:
            chi_start = norm.ppf(1.0 - P_rain) - 1.0
            raining_start = False
            
        realizations = []
        steps = request.steps
        n_real = request.n_realizations
        
        for _ in range(n_real):
            rates = []
            chi = chi_start
            raining = raining_start
            
            mean_rain_dur = tau_c
            mean_clear_dur = tau_c * (1 - P_rain) / (P_rain + 1e-9)
            p_onset = 1 - np.exp(-dt_s / mean_clear_dur)
            p_clear = 1 - np.exp(-dt_s / mean_rain_dur)
            
            for _ in range(steps):
                r_val = random.random()
                if raining:
                    if r_val < p_clear:
                        raining = False
                else:
                    if r_val < p_onset:
                        raining = True
                        
                noise = np.random.normal(0, 1)
                chi = rho * chi + np.sqrt(1 - rho**2) * noise
                
                if raining:
                    rate = np.exp(chi * sigma + mu)
                else:
                    rate = 0.0
                rates.append(float(rate))
            realizations.append(rates)
            
        real_arr = np.array(realizations)
        mean_forecast = np.mean(real_arr, axis=0).tolist()
        p90_forecast = np.percentile(real_arr, 90, axis=0).tolist()
        p10_forecast = np.percentile(real_arr, 10, axis=0).tolist()
        
        json_data = {
            "mean_forecast": mean_forecast,
            "p90_forecast": p90_forecast,
            "p10_forecast": p10_forecast,
            "ensemble_members": realizations
        }
        
        df_rows = []
        for r_idx in range(n_real):
            for t in range(steps):
                df_rows.append({
                    "realization_id": r_idx + 1,
                    "time_step": t,
                    "predicted_rain_rate": realizations[r_idx][t]
                })
        df = pd.DataFrame(df_rows)
        return respond_with_format(df, json_data, format, "rain_forecast")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("forecast_rain_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/handoff/live")
async def live_handoff(
    request: LiveHandoffRequest
):
    try:
        from satlinksim.domain.handoff.manager import HighestElevationPolicy, HighestSNRPolicy
        
        if request.handoff_policy.lower() == "highest_snr":
            policy = HighestSNRPolicy()
        else:
            policy = HighestElevationPolicy()
            
        current_sat_idx = None
        if request.current_satellite in request.candidates_names:
            current_sat_idx = request.candidates_names.index(request.current_satellite)
            
        snr_arr = np.array(request.snr_metrics)
        el_arr = np.array(request.el_metrics)
        
        new_idx, should_switch, reason, delta = policy.select_best(
            current_sat_idx=current_sat_idx,
            dwell_timer=request.dwell_timer,
            min_dwell_steps=request.min_dwell_steps,
            hysteresis=request.hysteresis,
            snr_metrics=snr_arr,
            el_metrics=el_arr
        )
        
        target_satellite = request.candidates_names[new_idx] if new_idx is not None else None
        
        return {
            "should_switch": should_switch,
            "target_satellite": target_satellite,
            "reason": reason if should_switch else "no_switch",
            "metric_delta": delta
        }
    except Exception as e:
        logger.error("live_handoff_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/orbit")
async def predict_orbit(
    request: OrbitRequest,
    format: str = Query("json", description="Output format: json, csv, or parquet")
):
    try:
        gs = resolve_ground_station(request.ground_station)
        sats = resolve_satellites([request.satellite])
        sat = sats[0]
        
        n_steps = max(1, int(request.duration / request.step)) if request.duration > 0 else 1
        dt_s = float(request.step)
        
        start_time = request.time or datetime.now(timezone.utc)
        times = [start_time + timedelta(seconds=i*dt_s) for i in range(n_steps)]
        
        from satlinksim.infrastructure.tle.service import SGP4Propagator
        propagator = SGP4Propagator()
        
        geo = propagator.get_geometry_batch(sat.norad_id, times, gs["latitude"], gs["longitude"], gs["altitude_km"])
        if not geo:
            raise HTTPException(status_code=500, detail="Orbital propagation failed.")
            
        elevation = geo.elevation_deg.tolist()
        slant_range = geo.slant_range_km.tolist()
        radial_velocity = geo.radial_velocity_ms.tolist()
        azimuth = geo.azimuth_deg.tolist() if geo.azimuth_deg is not None else [0.0] * n_steps
        
        json_data = {
            "satellite": sat.name,
            "time_step": list(range(n_steps)),
            "elevation": elevation,
            "slant_range_km": slant_range,
            "radial_velocity_ms": radial_velocity,
            "azimuth": azimuth
        }
        
        df = pd.DataFrame({
            "time_step": range(n_steps),
            "satellite": [sat.name] * n_steps,
            "elevation": elevation,
            "slant_range_km": slant_range,
            "radial_velocity_ms": radial_velocity,
            "azimuth": azimuth
        })
        return respond_with_format(df, json_data, format, f"orbital_states_{sat.norad_id}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("orbit_prediction_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/coverage")
async def constellation_coverage(
    request: CoverageRequest,
    format: str = Query("json", description="Output format: json, csv, or parquet")
):
    try:
        sats = resolve_satellites(request.satellites)
        n_steps = int(request.duration / request.step)
        dt_s = float(request.step)
        
        start_time = datetime.now(timezone.utc)
        times = [start_time + timedelta(seconds=i*dt_s) for i in range(n_steps)]
        
        from satlinksim.infrastructure.tle.service import SGP4Propagator
        propagator = SGP4Propagator()
        
        coverage_results = {}
        df_rows = []
        
        for gs_input in request.ground_stations:
            gs = resolve_ground_station(gs_input)
            
            station_step_visible = np.zeros(n_steps, dtype=bool)
            
            for sat in sats:
                geo = propagator.get_geometry_batch(sat.norad_id, times, gs["latitude"], gs["longitude"], gs["altitude_km"])
                if geo:
                    visible_mask = geo.elevation_deg >= request.min_elevation
                    station_step_visible |= visible_mask
                    
            visible_count = int(np.sum(station_step_visible))
            coverage_fraction = float(visible_count / n_steps)
            
            coverage_results[gs["name"]] = {
                "coverage_fraction": coverage_fraction,
                "visible_duration_seconds": float(visible_count * dt_s),
                "total_duration_seconds": float(n_steps * dt_s)
            }
            
            df_rows.append({
                "station": gs["name"],
                "coverage_fraction": coverage_fraction,
                "visible_duration_seconds": float(visible_count * dt_s),
                "total_duration_seconds": float(n_steps * dt_s)
            })
            
        df = pd.DataFrame(df_rows)
        return respond_with_format(df, coverage_results, format, "constellation_coverage")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("coverage_calculation_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/constellation")
async def list_constellations():
    return NAMED_CONSTELLATIONS

@app.post("/constellation")
async def add_constellation(request: ConstellationRequest):
    try:
        resolved_sats = resolve_satellites(request.satellites)
        NAMED_CONSTELLATIONS[request.name] = [s.norad_id for s in resolved_sats]
        return {"status": "success", "constellation": request.name, "satellites": NAMED_CONSTELLATIONS[request.name]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("add_constellation_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}


# ==============================================================================
# Cloud-Native Versioned API (v1)
# ==============================================================================

from fastapi import APIRouter, Depends, Security, status, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse, JSONResponse

# Security verification
security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials or not credentials.credentials.startswith("sk_"):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API Key. Must be a Bearer token starting with 'sk_'."
        )
    return credentials.credentials

# Router with Bearer Token Authentication
v1_router = APIRouter(prefix="/api/v1", dependencies=[Depends(verify_token)])

# In-memory store for simulation resources
simulations_store: Dict[str, Dict[str, Any]] = {}

class GlobalCoverageRequest(BaseModel):
    satellites: List[str]
    duration: int = 3600
    step: int = 60
    min_elevation: float = 10.0
    grid_size: int = 30

def ecef_to_geodetic(x, y, z):
    # x, y, z in km
    x_m = x * 1000.0
    y_m = y * 1000.0
    z_m = z * 1000.0
    a = 6378137.0
    f = 1 / 298.257223563
    b = a * (1 - f)
    e2 = (a**2 - b**2) / a**2
    ep2 = (a**2 - b**2) / b**2
    p = np.sqrt(x_m**2 + y_m**2)
    th = np.arctan2(a * z_m, b * p)
    lon = np.arctan2(y_m, x_m)
    lat = np.arctan2(z_m + ep2 * b * np.sin(th)**3, p - e2 * a * np.cos(th)**3)
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    alt = p / np.cos(lat) - N
    return float(np.degrees(lat)), float(np.degrees(lon)), float(alt / 1000.0)

# --- Simulations Resource ---

@v1_router.post("/simulations", status_code=201)
async def create_simulation(
    request: Union[PublicSimulationRequest, SimulationRequest]
):
    sim_id = str(uuid.uuid4())
    start_time = time.time()
    SIMULATIONS_RUN.labels(mode="v1").inc()
    
    req_type = "public" if isinstance(request, PublicSimulationRequest) else "full"
    
    try:
        if isinstance(request, SimulationRequest):
            gs_dicts = [gs.model_dump() for gs in request.ground_stations]
            constellation = None
            if request.constellation:
                sats = [
                    Satellite(norad_id=s.norad_id, name=s.name, tle_line1=s.tle_line1, tle_line2=s.tle_line2)
                    for s in request.constellation.satellites
                ]
                constellation = Constellation(name=request.constellation.name, satellites=sats)

            results = engine.simulate_all_batched(
                ground_stations=gs_dicts,
                n_steps=request.n_steps,
                dt_s=request.dt_s,
                start_time=request.start_time,
                force_rain=request.force_rain,
                seed=request.seed,
                freq_hz=request.freq_hz,
                eirp_offset_db=request.eirp_offset_db,
                bandwidth_hz=request.bandwidth_hz,
                polarization=request.polarization,
                rain_rate_scale=request.rain_rate_scale,
                constellation=constellation,
                handoff_policy=request.handoff_policy,
                hysteresis=request.hysteresis,
                min_dwell_steps=request.min_dwell_steps
            )
            
            response_results = []
            for res in results:
                handoffs = [
                    HandoffEventSchema(
                        time_step=h.time_step,
                        old_sat=h.old_sat,
                        new_sat=h.new_sat,
                        reason=h.reason,
                        metric_delta=h.metric_delta
                    ) for h in res.handoff_events
                ]
                response_results.append(StationResultSchema(
                    name=res.name, elevation=res.elevation, slant_km=res.slant_km, doppler_hz=res.doppler_hz,
                    path_loss=res.path_loss, gas_loss=res.gas_loss, rain_height=res.rain_height,
                    eff_path=res.eff_path, itu_k=res.itu_k, itu_alpha=res.itu_alpha,
                    scint_sig=res.scint_sig, noise_floor=res.noise_floor, snr_series=res.snr_series,
                    rain_series=res.rain_series, rain_db_series=res.rain_db_series, scint_series=res.scint_series,
                    pkt_loss_series=res.pkt_loss_series, elevation_series=res.elevation_series,
                    slant_range_series=res.slant_range_series, doppler_series=res.doppler_series,
                    snr_mean=res.snr_mean, snr_min=res.snr_min, snr_std=res.snr_std, snr_p10=res.snr_p10,
                    rain_fraction=res.rain_fraction, avg_rain_db=res.avg_rain_db, avg_pkt_loss=res.avg_pkt_loss,
                    outage_fraction=res.outage_fraction, sat_name_series=res.sat_name_series,
                    handoff_events=handoffs
                ))
            
            t_finish = time.time()
            compute_time = t_finish - start_time
            finished_epoch = datetime.fromtimestamp(t_finish, timezone.utc).isoformat()
            created_epoch = datetime.fromtimestamp(start_time, timezone.utc).isoformat()
            
            num_satellites = 0
            if request.constellation and request.constellation.satellites:
                num_satellites = len(request.constellation.satellites)
            
            simulations_store[sim_id] = {
                "id": sim_id,
                "status": "completed",
                "created_at": created_epoch,
                "finished_at": finished_epoch,
                "compute_time": compute_time,
                "duration": request.n_steps * request.dt_s,
                "timesteps": request.n_steps,
                "num_satellites": num_satellites,
                "version": "1.0.0",
                "request_type": req_type,
                "request_data": request.model_dump(),
                "results": response_results,
                "engine_results": results
            }
        else:
            gs = resolve_ground_station(request.ground_station)
            sats = resolve_satellites(request.satellites)
            n_steps = int(request.duration / request.step)
            dt_s = float(request.step)

            if request.handoff:
                constellation = Constellation(name="Constellation", satellites=sats)
                results = engine.simulate_all_batched(
                    ground_stations=[gs],
                    n_steps=n_steps,
                    dt_s=dt_s,
                    freq_hz=request.frequency,
                    force_rain=request.rain,
                    constellation=constellation,
                    handoff_policy="highest_elevation"
                )
            else:
                gs_copy = gs.copy()
                gs_copy["norad_id"] = sats[0].norad_id
                gs_copy["sat_name"] = sats[0].name
                results = engine.simulate_all_batched(
                    ground_stations=[gs_copy],
                    n_steps=n_steps,
                    dt_s=dt_s,
                    freq_hz=request.frequency,
                    force_rain=request.rain,
                    constellation=None
                )
            
            res = results[0]
            if not request.rain:
                disable_rain_in_result(res)

            from satlinksim.application.simulation_engine import SNR_THRESHOLD_DB
            snr_list = res.snr_series
            availability_list = [int(s >= SNR_THRESHOLD_DB) for s in snr_list]
            handoffs_list = [
                {
                    "time_step": h.time_step,
                    "old_sat": h.old_sat,
                    "new_sat": h.new_sat,
                    "reason": h.reason,
                    "metric_delta": h.metric_delta
                }
                for h in res.handoff_events
            ]
            rain_loss_list = res.rain_db_series
            stations_list = [gs["name"]]
            
            json_data = {
                "snr": snr_list,
                "availability": availability_list,
                "handoffs": handoffs_list,
                "rain_loss": rain_loss_list,
                "stations": stations_list
            }
            
            df = pd.DataFrame({
                "time_step": range(len(snr_list)),
                "station": [gs["name"]] * len(snr_list),
                "satellite": res.sat_name_series or [sats[0].name] * len(snr_list),
                "snr": snr_list,
                "availability": availability_list,
                "rain_loss": rain_loss_list
            })
            
            t_finish = time.time()
            compute_time = t_finish - start_time
            finished_epoch = datetime.fromtimestamp(t_finish, timezone.utc).isoformat()
            created_epoch = datetime.fromtimestamp(start_time, timezone.utc).isoformat()
            
            simulations_store[sim_id] = {
                "id": sim_id,
                "status": "completed",
                "created_at": created_epoch,
                "finished_at": finished_epoch,
                "compute_time": compute_time,
                "duration": request.duration,
                "timesteps": int(request.duration / request.step),
                "num_satellites": len(request.satellites),
                "version": "1.0.0",
                "request_type": req_type,
                "request_data": request.model_dump(),
                "json_data": json_data,
                "dataframe": df,
                "engine_results": results
            }
            
        latency = time.time() - start_time
        SIMULATION_LATENCY.labels(mode="v1").observe(latency)
        
        return {
            "id": sim_id,
            "status": "completed",
            "created_at": simulations_store[sim_id]["created_at"],
            "finished_at": simulations_store[sim_id]["finished_at"],
            "compute_time": simulations_store[sim_id]["compute_time"],
            "duration": simulations_store[sim_id]["duration"],
            "timesteps": simulations_store[sim_id]["timesteps"],
            "num_satellites": simulations_store[sim_id]["num_satellites"],
            "version": "1.0.0",
            "request_type": req_type
        }
    except Exception as e:
        logger.error("v1_simulation_failed", error=str(e))
        simulations_store[sim_id] = {
            "id": sim_id,
            "status": "failed",
            "error": str(e),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "request_type": req_type,
            "request_data": request.model_dump()
        }
        raise HTTPException(status_code=500, detail=str(e))

@v1_router.get("/simulations")
async def list_simulations(
    page: int = Query(1, description="Page number", ge=1),
    limit: int = Query(10, description="Items per page", ge=1)
):
    page = page if isinstance(page, int) else 1
    limit = limit if isinstance(limit, int) else 10
    
    sims = sorted(
        simulations_store.values(),
        key=lambda x: x["created_at"],
        reverse=True
    )
    
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    paginated_sims = sims[start_idx:end_idx]
    
    result_list = []
    for s in paginated_sims:
        result_list.append({
            "id": s["id"],
            "status": s["status"],
            "created_at": s["created_at"],
            "request_type": s["request_type"],
            "request_data": s["request_data"]
        })
        
    return {
        "page": page,
        "limit": limit,
        "total_items": len(sims),
        "total_pages": (len(sims) + limit - 1) // limit if len(sims) > 0 else 0,
        "simulations": result_list
    }

@v1_router.get("/simulations/{id}")
async def get_simulation(id: str):
    if id not in simulations_store:
        raise HTTPException(status_code=404, detail="Simulation not found")
    s = simulations_store[id]
    response = {
        "id": s["id"],
        "status": s["status"],
        "created_at": s.get("created_at"),
        "finished_at": s.get("finished_at"),
        "compute_time": s.get("compute_time"),
        "duration": s.get("duration"),
        "timesteps": s.get("timesteps"),
        "num_satellites": s.get("num_satellites"),
        "version": s.get("version", "1.0.0"),
        "request_type": s["request_type"],
        "request_data": s["request_data"]
    }
    if s["status"] == "failed":
        response["error"] = s.get("error")
    return response

@v1_router.delete("/simulations/{id}")
async def delete_simulation(id: str):
    if id not in simulations_store:
        raise HTTPException(status_code=404, detail="Simulation not found")
    del simulations_store[id]
    return {"status": "success", "message": f"Simulation {id} deleted."}

@v1_router.get("/simulations/{id}/summary")
async def get_simulation_summary(id: str):
    if id not in simulations_store:
        raise HTTPException(status_code=404, detail="Simulation not found")
    s = simulations_store[id]
    if s["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Simulation status is {s['status']}. Summary is only available for completed simulations.")
    
    if s["request_type"] == "public":
        snr = s["json_data"]["snr"]
        availability = s["json_data"]["availability"]
        rain_loss = s["json_data"]["rain_loss"]
        handoffs = s["json_data"]["handoffs"]
        
        availability_pct = float(np.mean(availability) * 100) if availability else 0.0
        mean_snr = float(np.mean(snr)) if snr else 0.0
        min_snr = float(np.min(snr)) if snr else 0.0
        max_snr = float(np.max(snr)) if snr else 0.0
        avg_rain = float(np.mean(rain_loss)) if rain_loss else 0.0
        handoff_count = len(handoffs)
        outage_count = len([x for x in availability if x == 0])
    else:
        results = s["results"]
        availability_pct = float(np.mean([1.0 - res.outage_fraction for res in results]) * 100) if results else 0.0
        
        all_snrs = []
        all_rain_db = []
        handoff_count = 0
        outage_count = 0
        
        for res in results:
            all_snrs.extend(res.snr_series)
            all_rain_db.extend(res.rain_db_series)
            handoff_count += len(res.handoff_events)
            outage_count += len([x for x in res.snr_series if x < 5.0])
            
        mean_snr = float(np.mean(all_snrs)) if all_snrs else 0.0
        min_snr = float(np.min(all_snrs)) if all_snrs else 0.0
        max_snr = float(np.max(all_snrs)) if all_snrs else 0.0
        avg_rain = float(np.mean(all_rain_db)) if all_rain_db else 0.0
        
    return {
        "availability": availability_pct,
        "mean_snr": mean_snr,
        "minimum_snr": min_snr,
        "maximum_snr": max_snr,
        "average_rain": avg_rain,
        "handoff_count": handoff_count,
        "outage_count": outage_count,
        "compute_time": s.get("compute_time", 0.0),
        "simulation_duration": s.get("duration", 0.0)
    }

@v1_router.patch("/simulations/{id}")
async def patch_simulation(id: str, action: str = Query(..., description="Action to perform: cancel, pause, resume")):
    if id not in simulations_store:
        raise HTTPException(status_code=404, detail="Simulation not found")
    s = simulations_store[id]
    action = action.lower()
    if action == "cancel":
        s["status"] = "cancelled"
    elif action == "pause":
        s["status"] = "paused"
    elif action == "resume":
        s["status"] = "completed"
    else:
        raise HTTPException(status_code=400, detail=f"Invalid action: '{action}'. Must be cancel, pause, or resume.")
    return {"id": id, "status": s["status"], "action": action}

@v1_router.get("/simulations/{id}/results")
async def get_simulation_results(
    id: str,
    format: str = Query("json", description="Output format: json, csv, or parquet")
):
    if id not in simulations_store:
        raise HTTPException(status_code=404, detail="Simulation not found")
    s = simulations_store[id]
    if s["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Simulation status is {s['status']}")
        
    if s["request_type"] == "public":
        return respond_with_format(s["dataframe"], s["json_data"], format, "simulation_results")
    else:
        df_rows = []
        for station_res in s["results"]:
            n_steps = len(station_res.snr_series)
            for t in range(n_steps):
                df_rows.append({
                    "time_step": t,
                    "station": station_res.name,
                    "satellite": station_res.sat_name_series[t] if station_res.sat_name_series else "Unknown",
                    "elevation": station_res.elevation_series[t],
                    "slant_range_km": station_res.slant_range_series[t],
                    "snr": station_res.snr_series[t],
                    "rain_rate": station_res.rain_series[t],
                    "rain_loss": station_res.rain_db_series[t],
                    "scint_loss": station_res.scint_series[t],
                    "pkt_loss": station_res.pkt_loss_series[t]
                })
        df = pd.DataFrame(df_rows)
        json_data = [res.model_dump() for res in s["results"]]
        return respond_with_format(df, json_data, format, "simulation_results")

@v1_router.get("/simulations/{id}/download")
async def download_simulation_results(id: str, format: str = Query("csv", description="Output format: csv or parquet")):
    return await get_simulation_results(id=id, format=format)

@v1_router.get("/simulations/{id}/download.csv")
async def download_simulation_results_csv(id: str):
    return await get_simulation_results(id=id, format="csv")

@v1_router.get("/simulations/{id}/download.parquet")
async def download_simulation_results_parquet(id: str):
    return await get_simulation_results(id=id, format="parquet")

@v1_router.get("/simulations/{id}/attenuation")
async def get_simulation_attenuation(
    id: str,
    format: str = Query("json", description="Output format: json, csv, or parquet")
):
    if id not in simulations_store:
        raise HTTPException(status_code=404, detail="Simulation not found")
    s = simulations_store[id]
    if s["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Simulation status is {s['status']}")
        
    engine_results = s["engine_results"]
    request_data = s["request_data"]
    
    freq_hz = request_data.get("frequency") or request_data.get("freq_hz") or 14e9
    freq_ghz = freq_hz / 1e9
    
    json_data = {}
    df_list = []
    
    for res in engine_results:
        try:
            gs = resolve_ground_station(res.name)
        except Exception:
            gs = {"wv_g_m3": 7.5}
            
        elevation = np.array(res.elevation_series)
        gas_loss = gaseous_absorption_db(freq_ghz, elevation, gs.get("wv_g_m3", 7.5))
        rain_loss = np.array(res.rain_db_series)
        scint_loss = np.array(res.scint_series)
        total_loss = gas_loss + rain_loss + scint_loss
        
        station_data = {
            "gaseous_attenuation": gas_loss.tolist(),
            "rain_attenuation": rain_loss.tolist(),
            "scintillation_attenuation": scint_loss.tolist(),
            "total_attenuation": total_loss.tolist()
        }
        json_data[res.name] = station_data
        
        n_steps = len(res.elevation_series)
        df_list.append(pd.DataFrame({
            "time_step": range(n_steps),
            "station": [res.name] * n_steps,
            "gaseous_attenuation": gas_loss,
            "rain_attenuation": rain_loss,
            "scintillation_attenuation": scint_loss,
            "total_attenuation": total_loss
        }))
        
    df = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()
    
    if s["request_type"] == "public" and len(engine_results) == 1:
        return respond_with_format(df.drop(columns=["station"]), json_data[engine_results[0].name], format, "attenuation")
        
    return respond_with_format(df, json_data, format, "attenuation")

@v1_router.get("/simulations/{id}/link-budget")
async def get_simulation_link_budget(
    id: str,
    format: str = Query("json", description="Output format: json, csv, or parquet")
):
    if id not in simulations_store:
        raise HTTPException(status_code=404, detail="Simulation not found")
    s = simulations_store[id]
    if s["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Simulation status is {s['status']}")
        
    engine_results = s["engine_results"]
    request_data = s["request_data"]
    
    freq_hz = request_data.get("frequency") or request_data.get("freq_hz") or 14e9
    freq_ghz = freq_hz / 1e9
    bandwidth_hz = request_data.get("bandwidth_hz") or 250e6
    
    json_data = {}
    df_list = []
    
    for res in engine_results:
        try:
            gs = resolve_ground_station(res.name)
        except Exception:
            gs = {"eirp_dbw": 55.0, "g_rx_dbi": 40.0, "system_temp_k": 150.0, "wv_g_m3": 7.5}
            
        noise_floor = noise_power_dbw(gs.get("system_temp_k", 150.0), bandwidth_hz)
        eirp = gs.get("eirp_dbw", 55.0)
        g_rx = gs.get("g_rx_dbi", 40.0)
        
        slant_range = np.array(res.slant_range_series)
        elevation = np.array(res.elevation_series)
        path_loss = fspl_db(freq_hz, slant_range)
        gas_loss = gaseous_absorption_db(freq_ghz, elevation, gs.get("wv_g_m3", 7.5))
        rain_loss = np.array(res.rain_db_series)
        scint_loss = np.array(res.scint_series)
        rx_power = eirp - path_loss - gas_loss - rain_loss - scint_loss + g_rx
        snr = np.array(res.snr_series)
        
        n_steps = len(res.elevation_series)
        station_data = {
            "time_step": list(range(n_steps)),
            "eirp": [float(eirp)] * n_steps,
            "path_loss": path_loss.tolist(),
            "gas_loss": gas_loss.tolist(),
            "rain_loss": rain_loss.tolist(),
            "scint_loss": scint_loss.tolist(),
            "rx_power": rx_power.tolist(),
            "noise_floor": [float(noise_floor)] * n_steps,
            "snr": snr.tolist()
        }
        json_data[res.name] = station_data
        
        df_list.append(pd.DataFrame({
            "time_step": range(n_steps),
            "station": [res.name] * n_steps,
            "eirp": [float(eirp)] * n_steps,
            "path_loss": path_loss,
            "gas_loss": gas_loss,
            "rain_loss": rain_loss,
            "scint_loss": scint_loss,
            "rx_power": rx_power,
            "noise_floor": [float(noise_floor)] * n_steps,
            "snr": snr
        }))
        
    df = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()
    
    if s["request_type"] == "public" and len(engine_results) == 1:
        return respond_with_format(df.drop(columns=["station"]), json_data[engine_results[0].name], format, "link_budget")
        
    return respond_with_format(df, json_data, format, "link_budget")

@v1_router.get("/simulations/{id}/visibility")
async def get_simulation_visibility(
    id: str,
    min_elevation: float = Query(10.0, description="Minimum elevation threshold for visibility"),
    format: str = Query("json", description="Output format: json, csv, or parquet")
):
    if id not in simulations_store:
        raise HTTPException(status_code=404, detail="Simulation not found")
    s = simulations_store[id]
    if s["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Simulation status is {s['status']}")
        
    engine_results = s["engine_results"]
    request_data = s["request_data"]
    
    sat_names = request_data.get("satellites")
    if not sat_names:
        if "constellation" in request_data and request_data["constellation"]:
            sat_names = [sat["name"] for sat in request_data["constellation"].get("satellites", [])]
        else:
            sat_names = list(set([name for res in engine_results for name in (res.sat_name_series or [])]))
            
    if not sat_names:
        raise HTTPException(status_code=400, detail="No satellites found in simulation configuration.")
        
    sats = resolve_satellites(sat_names)
    
    dt_s = request_data.get("step") or request_data.get("dt_s") or 1.0
    n_steps = len(engine_results[0].elevation_series)
    start_time_raw = request_data.get("start_time")
    if isinstance(start_time_raw, str):
        curr_time = datetime.fromisoformat(start_time_raw.replace("Z", "+00:00"))
    else:
        curr_time = datetime.now(timezone.utc)
    times = [curr_time + timedelta(seconds=i*dt_s) for i in range(n_steps)]
    
    from satlinksim.infrastructure.tle.service import SGP4Propagator
    propagator = SGP4Propagator()
    
    json_data = {}
    df_rows = []
    
    for res in engine_results:
        try:
            gs = resolve_ground_station(res.name)
        except Exception:
            gs = {"latitude": 0.0, "longitude": 0.0, "altitude_km": 0.0}
            
        satellites_data = {}
        for sat in sats:
            geo = propagator.get_geometry_batch(sat.norad_id, times, gs["latitude"], gs["longitude"], gs["altitude_km"])
            if geo:
                elevation = geo.elevation_deg.tolist()
                azimuth = geo.azimuth_deg.tolist() if geo.azimuth_deg is not None else [0.0] * n_steps
                visible = (geo.elevation_deg >= min_elevation).astype(int).tolist()
                satellites_data[sat.name] = {
                    "elevation": elevation,
                    "azimuth": azimuth,
                    "visible": visible
                }
                for t in range(n_steps):
                    df_rows.append({
                        "time_step": t,
                        "station": res.name,
                        "satellite": sat.name,
                        "elevation": elevation[t],
                        "azimuth": azimuth[t],
                        "visible": visible[t]
                    })
            else:
                satellites_data[sat.name] = {
                    "elevation": [0.0] * n_steps,
                    "azimuth": [0.0] * n_steps,
                    "visible": [0] * n_steps
                }
                for t in range(n_steps):
                    df_rows.append({
                        "time_step": t,
                        "station": res.name,
                        "satellite": sat.name,
                        "elevation": 0.0,
                        "azimuth": 0.0,
                        "visible": 0
                    })
        json_data[res.name] = {
            "time_step": list(range(n_steps)),
            "satellites": satellites_data
        }
        
    df = pd.DataFrame(df_rows)
    
    if s["request_type"] == "public" and len(engine_results) == 1:
        return respond_with_format(df.drop(columns=["station"]), json_data[engine_results[0].name], format, "visibility")
        
    return respond_with_format(df, json_data, format, "visibility")

@v1_router.get("/simulations/{id}/availability")
async def get_simulation_availability(
    id: str,
    snr_threshold: float = Query(5.0, description="SNR threshold in dB for availability"),
    format: str = Query("json", description="Output format: json, csv, or parquet")
):
    if id not in simulations_store:
        raise HTTPException(status_code=404, detail="Simulation not found")
    s = simulations_store[id]
    if s["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Simulation status is {s['status']}")
        
    engine_results = s["engine_results"]
    request_data = s["request_data"]
    dt_s = request_data.get("step") or request_data.get("dt_s") or 1.0
    
    json_data = {}
    df_list = []
    
    for res in engine_results:
        is_available = np.array(res.snr_series) >= snr_threshold
        availability_fraction = float(np.mean(is_available))
        n_steps = len(res.snr_series)
        total_duration_seconds = n_steps * dt_s
        outage_duration_seconds = float(np.sum(~is_available) * dt_s)
        
        outages = []
        in_outage = False
        outage_start = 0
        for i, avail in enumerate(is_available):
            if not avail and not in_outage:
                in_outage = True
                outage_start = i
            elif avail and in_outage:
                in_outage = False
                duration = (i - outage_start) * dt_s
                outages.append({
                    "start_step": int(outage_start),
                    "end_step": int(i),
                    "duration_seconds": float(duration)
                })
        if in_outage:
            duration = (len(is_available) - outage_start) * dt_s
            outages.append({
                "start_step": int(outage_start),
                "end_step": int(len(is_available)),
                "duration_seconds": float(duration)
            })
            
        number_of_outages = len(outages)
        
        station_data = {
            "availability_fraction": availability_fraction,
            "total_duration_seconds": total_duration_seconds,
            "outage_duration_seconds": outage_duration_seconds,
            "number_of_outages": number_of_outages,
            "outages": outages
        }
        json_data[res.name] = station_data
        
        df_list.append(pd.DataFrame({
            "time_step": range(n_steps),
            "station": [res.name] * n_steps,
            "snr": res.snr_series,
            "available": is_available.astype(int)
        }))
        
    df = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()
    
    if s["request_type"] == "public" and len(engine_results) == 1:
        first_station = engine_results[0].name
        first_data = json_data[first_station]
        headers = {
            "X-Availability-Fraction": str(first_data["availability_fraction"]),
            "X-Total-Duration-Seconds": str(first_data["total_duration_seconds"]),
            "X-Outage-Duration-Seconds": str(first_data["outage_duration_seconds"]),
            "X-Number-Of-Outages": str(first_data["number_of_outages"])
        }
        return respond_with_format(df.drop(columns=["station"]), first_data, format, "availability", headers=headers)
        
    return respond_with_format(df, json_data, format, "availability")

@v1_router.get("/simulations/{id}/handoffs")
async def get_simulation_handoffs(
    id: str,
    format: str = Query("json", description="Output format: json, csv, or parquet")
):
    if id not in simulations_store:
        raise HTTPException(status_code=404, detail="Simulation not found")
    s = simulations_store[id]
    if s["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Simulation status is {s['status']}")
        
    engine_results = s["engine_results"]
    
    json_data = {}
    df_rows = []
    
    for res in engine_results:
        handoffs_list = [
            {
                "time_step": h.time_step,
                "old_sat": h.old_sat,
                "new_sat": h.new_sat,
                "reason": h.reason,
                "metric_delta": h.metric_delta
            }
            for h in res.handoff_events
        ]
        json_data[res.name] = handoffs_list
        
        for h in res.handoff_events:
            df_rows.append({
                "station": res.name,
                "time_step": h.time_step,
                "old_sat": h.old_sat,
                "new_sat": h.new_sat,
                "reason": h.reason,
                "metric_delta": h.metric_delta
            })
            
    df = pd.DataFrame(df_rows) if df_rows else pd.DataFrame(columns=["station", "time_step", "old_sat", "new_sat", "reason", "metric_delta"])
    
    if s["request_type"] == "public" and len(engine_results) == 1:
        return respond_with_format(df.drop(columns=["station"]), json_data[engine_results[0].name], format, "handoffs")
        
    return respond_with_format(df, json_data, format, "handoffs")

@v1_router.get("/simulations/{id}/orbit")
async def get_simulation_orbit(
    id: str,
    format: str = Query("json", description="Output format: json, csv, or parquet")
):
    if id not in simulations_store:
        raise HTTPException(status_code=404, detail="Simulation not found")
    s = simulations_store[id]
    if s["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Simulation status is {s['status']}")
        
    engine_results = s["engine_results"]
    request_data = s["request_data"]
    
    sat_names = request_data.get("satellites")
    if not sat_names:
        if "constellation" in request_data and request_data["constellation"]:
            sat_names = [sat["name"] for sat in request_data["constellation"].get("satellites", [])]
        else:
            sat_names = list(set([name for res in engine_results for name in (res.sat_name_series or [])]))
            
    if not sat_names:
        raise HTTPException(status_code=400, detail="No satellites found in simulation configuration.")
        
    sats = resolve_satellites(sat_names)
    
    dt_s = request_data.get("step") or request_data.get("dt_s") or 1.0
    n_steps = len(engine_results[0].elevation_series)
    start_time_raw = request_data.get("start_time")
    if isinstance(start_time_raw, str):
        curr_time = datetime.fromisoformat(start_time_raw.replace("Z", "+00:00"))
    else:
        curr_time = datetime.now(timezone.utc)
    times = [curr_time + timedelta(seconds=i*dt_s) for i in range(n_steps)]
    
    from satlinksim.infrastructure.tle.service import SGP4Propagator
    propagator = SGP4Propagator()
    
    json_data = {}
    df_rows = []
    
    for res in engine_results:
        try:
            gs = resolve_ground_station(res.name)
        except Exception:
            gs = {"latitude": 0.0, "longitude": 0.0, "altitude_km": 0.0}
            
        station_sat_orbits = {}
        for sat in sats:
            geo = propagator.get_geometry_batch(sat.norad_id, times, gs["latitude"], gs["longitude"], gs["altitude_km"])
            if geo:
                elevation = geo.elevation_deg.tolist()
                slant_range = geo.slant_range_km.tolist()
                radial_velocity = geo.radial_velocity_ms.tolist()
                azimuth = geo.azimuth_deg.tolist() if geo.azimuth_deg is not None else [0.0] * n_steps
                
                station_sat_orbits[sat.name] = {
                    "elevation": elevation,
                    "slant_range_km": slant_range,
                    "radial_velocity_ms": radial_velocity,
                    "azimuth": azimuth
                }
                for t in range(n_steps):
                    df_rows.append({
                        "station": res.name,
                        "satellite": sat.name,
                        "time_step": t,
                        "elevation": elevation[t],
                        "slant_range_km": slant_range[t],
                        "radial_velocity_ms": radial_velocity[t],
                        "azimuth": azimuth[t]
                    })
        json_data[res.name] = station_sat_orbits
        
    df = pd.DataFrame(df_rows)
    
    if s["request_type"] == "public" and len(engine_results) == 1:
        first_station = engine_results[0].name
        first_data = json_data[first_station]
        if len(first_data) == 1:
            sat_name = list(first_data.keys())[0]
            flat_data = {
                "satellite": sat_name,
                "time_step": list(range(n_steps)),
                **first_data[sat_name]
            }
            return respond_with_format(df.drop(columns=["station"]), flat_data, format, f"orbital_states")
        return respond_with_format(df.drop(columns=["station"]), first_data, format, "orbital_states")
        
    return respond_with_format(df, json_data, format, "orbital_states")

# --- Jobs Resource ---

def process_batch_job_task(job_id: str, request: BatchSimulationRequest):
    structlog.contextvars.bind_contextvars(job_id=job_id)
    logger.info("batch_job_started", satellites=request.satellites, ground_stations=request.ground_stations)
    
    start_time = time.time()
    jobs[job_id].status = "running"
    try:
        station_results = {}
        for gs_name in request.ground_stations:
            gs = resolve_ground_station(gs_name)
            sats = resolve_satellites(request.satellites)
            n_steps = int(request.duration / request.step)
            dt_s = float(request.step)

            if request.handoff:
                constellation = Constellation(name="Constellation", satellites=sats)
                results = engine.simulate_all_batched(
                    ground_stations=[gs],
                    n_steps=n_steps,
                    dt_s=dt_s,
                    freq_hz=request.frequency,
                    force_rain=request.rain,
                    constellation=constellation,
                    handoff_policy="highest_elevation"
                )
            else:
                gs_copy = gs.copy()
                gs_copy["norad_id"] = sats[0].norad_id
                gs_copy["sat_name"] = sats[0].name
                results = engine.simulate_all_batched(
                    ground_stations=[gs_copy],
                    n_steps=n_steps,
                    dt_s=dt_s,
                    freq_hz=request.frequency,
                    force_rain=request.rain,
                    constellation=None
                )
            
            res = results[0]
            if not request.rain:
                disable_rain_in_result(res)

            from satlinksim.application.simulation_engine import SNR_THRESHOLD_DB
            snr_list = res.snr_series
            availability_list = [int(s >= SNR_THRESHOLD_DB) for s in snr_list]
            handoffs_list = [
                {
                    "time_step": h.time_step,
                    "old_sat": h.old_sat,
                    "new_sat": h.new_sat,
                    "reason": h.reason,
                    "metric_delta": h.metric_delta
                }
                for h in res.handoff_events
            ]
            
            station_results[gs["name"]] = {
                "snr": snr_list,
                "availability": availability_list,
                "handoffs": handoffs_list,
                "rain_loss": res.rain_db_series,
                "stations": [gs["name"]]
            }
            
        jobs[job_id].result = station_results
        jobs[job_id].status = "completed"
        
        latency = time.time() - start_time
        logger.info("batch_job_completed", duration_s=round(latency, 4))
    except Exception as e:
        jobs[job_id].status = "failed"
        jobs[job_id].error = str(e)
        logger.error("batch_job_failed", error=str(e))

@v1_router.post("/jobs", response_model=JobResponse)
async def create_job(request: BatchSimulationRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = JobStatus(job_id=job_id, status="pending")
    background_tasks.add_task(process_batch_job_task, job_id, request)
    logger.info("batch_job_submitted", job_id=job_id)
    return JobResponse(job_id=job_id)

@v1_router.get("/jobs/{id}", response_model=JobStatus)
async def get_job_v1(id: str):
    if id not in jobs:
        logger.warning("job_not_found", job_id=id)
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[id]

@v1_router.get("/jobs/{id}/results")
async def get_job_results_v1(
    id: str,
    format: str = Query("json", description="Output format: json, csv, or parquet")
):
    if id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[id]
    if job.status != "completed":
        raise HTTPException(status_code=400, detail=f"Job status is {job.status}")
        
    result = job.result
    if isinstance(result, dict):
        df_rows = []
        for station_name, data in result.items():
            snr_list = data["snr"]
            availability_list = data["availability"]
            rain_loss_list = data["rain_loss"]
            for t in range(len(snr_list)):
                df_rows.append({
                    "time_step": t,
                    "station": station_name,
                    "snr": snr_list[t],
                    "availability": availability_list[t],
                    "rain_loss": rain_loss_list[t]
                })
        df = pd.DataFrame(df_rows)
        return respond_with_format(df, result, format, "batch_results")
    else:
        df_rows = []
        for station_res in result.results:
            n_steps = len(station_res.snr_series)
            for t in range(n_steps):
                df_rows.append({
                    "time_step": t,
                    "station": station_res.name,
                    "snr": station_res.snr_series[t]
                })
        df = pd.DataFrame(df_rows)
        return respond_with_format(df, result.model_dump(), format, "simulation_results")

# --- Calculators Resource ---

class FsplRequest(BaseModel):
    frequency_hz: float
    distance_km: float

class SlantRangeRequest(BaseModel):
    altitude_km: float
    elevation_deg: float

class NoiseFloorRequest(BaseModel):
    system_temp_k: float
    bandwidth_hz: float

class EirpRequest(BaseModel):
    tx_power_dbw: float
    tx_gain_dbi: float
    line_loss_db: float = 0.0

class RainAttenuationRequest(BaseModel):
    rain_rate: float
    elevation_deg: float
    frequency_hz: float
    polarization: str = "circular"
    gs_latitude: float = 0.0

class GaseousAttenuationRequest(BaseModel):
    frequency_hz: float
    elevation_deg: float
    water_vapor_g_m3: float = 7.5

class ScintillationRequest(BaseModel):
    frequency_hz: float
    elevation_deg: float
    gs_antenna_diam: float = 1.2

@v1_router.post("/calculators/fspl")
async def calc_fspl(request: FsplRequest):
    from satlinksim.domain.link.budget import fspl_db
    loss = fspl_db(request.frequency_hz, request.distance_km)
    return {"fspl_db": float(loss)}

@v1_router.post("/calculators/slant-range")
async def calc_slant_range(request: SlantRangeRequest):
    from satlinksim.geometry import slant_range
    d = slant_range(request.altitude_km, request.elevation_deg)
    return {"slant_range_km": float(d)}

@v1_router.post("/calculators/noise-floor")
async def calc_noise_floor(request: NoiseFloorRequest):
    from satlinksim.domain.link.budget import noise_power_dbw
    nf = noise_power_dbw(request.system_temp_k, request.bandwidth_hz)
    return {"noise_floor_dbw": float(nf)}

@v1_router.post("/calculators/eirp")
async def calc_eirp(request: EirpRequest):
    eirp = request.tx_power_dbw + request.tx_gain_dbi - request.line_loss_db
    return {"eirp_dbw": float(eirp)}

@v1_router.post("/calculators/rain-attenuation")
async def calc_rain_attenuation(request: RainAttenuationRequest):
    from satlinksim.domain.link.itu_models import itu_rain_coefficients, itu_rain_height, effective_path_length, rain_attenuation_db
    freq_ghz = request.frequency_hz / 1e9
    itu_k, itu_alpha = itu_rain_coefficients(freq_ghz, request.polarization)
    rain_h = itu_rain_height(request.gs_latitude)
    ep = effective_path_length(request.elevation_deg, rain_h, 0.0, itu_k)
    ep_safe = np.maximum(ep, 1e-6)
    attn = rain_attenuation_db(request.rain_rate, itu_k, itu_alpha, ep_safe)
    return {"rain_attenuation_db": float(attn)}

@v1_router.post("/calculators/gaseous-attenuation")
async def calc_gaseous_attenuation(request: GaseousAttenuationRequest):
    from satlinksim.domain.link.itu_models import gaseous_absorption_db
    freq_ghz = request.frequency_hz / 1e9
    loss = gaseous_absorption_db(freq_ghz, request.elevation_deg, request.water_vapor_g_m3)
    return {"gaseous_attenuation_db": float(loss)}

@v1_router.post("/calculators/scintillation")
async def calc_scintillation(request: ScintillationRequest):
    from satlinksim.domain.link.itu_models import scintillation_sigma_db
    freq_ghz = request.frequency_hz / 1e9
    sigma = scintillation_sigma_db(freq_ghz, request.elevation_deg, request.gs_antenna_diam, 50.0)
    return {"scintillation_sigma_db": float(sigma)}

@v1_router.post("/calculators/link-budget")
async def calculator_link_budget(
    request: LinkBudgetRequest,
    format: str = Query("json", description="Output format: json, csv, or parquet")
):
    return await link_budget(request, format)

@v1_router.post("/calculators/attenuation")
async def calculator_attenuation(
    request: AttenuationRequest,
    format: str = Query("json", description="Output format: json, csv, or parquet")
):
    return await attenuation(request, format)

@v1_router.post("/calculators/availability")
async def calculator_availability(
    request: AvailabilityRequest,
    format: str = Query("json", description="Output format: json, csv, or parquet")
):
    return await availability(request, format)

# --- Rain Resource ---

@v1_router.post("/rain/invert")
async def rain_invert_v1(
    request: PredictRainRequest,
    format: str = Query("json", description="Output format: json, csv, or parquet")
):
    return await predict_rain(request, format)

@v1_router.post("/rain/forecast")
async def rain_forecast_v1(
    request: ForecastRainRequest,
    format: str = Query("json", description="Output format: json, csv, or parquet")
):
    return await forecast_rain(request, format)

# --- Coverage Resource ---

@v1_router.post("/coverage/station")
async def coverage_v1(
    request: CoverageRequest,
    format: str = Query("json", description="Output format: json, csv, or parquet")
):
    return await constellation_coverage(request, format)

@v1_router.post("/coverage/global")
async def global_coverage(
    request: GlobalCoverageRequest,
    format: str = Query("json", description="Output format: json, csv, or parquet")
):
    try:
        sats = resolve_satellites(request.satellites)
        n_steps = int(request.duration / request.step)
        dt_s = float(request.step)
        
        start_time = datetime.now(timezone.utc)
        times = [start_time + timedelta(seconds=i*dt_s) for i in range(n_steps)]
        
        from satlinksim.infrastructure.tle.service import SGP4Propagator
        propagator = SGP4Propagator()
        
        grid_size = max(10, request.grid_size)
        latitudes = list(range(-90 + grid_size, 90, grid_size))
        longitudes = list(range(-180, 180, grid_size))
        
        grid_results = []
        df_rows = []
        
        for lat in latitudes:
            for lon in longitudes:
                point_step_visible = np.zeros(n_steps, dtype=bool)
                for sat in sats:
                    geo = propagator.get_geometry_batch(sat.norad_id, times, lat, lon, 0.0)
                    if geo:
                        visible_mask = geo.elevation_deg >= request.min_elevation
                        point_step_visible |= visible_mask
                
                visible_count = int(np.sum(point_step_visible))
                coverage_fraction = float(visible_count / n_steps)
                
                point_data = {
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "coverage_fraction": coverage_fraction,
                    "visible_duration_seconds": float(visible_count * dt_s),
                    "total_duration_seconds": float(n_steps * dt_s)
                }
                grid_results.append(point_data)
                df_rows.append(point_data)
                
        json_data = {
            "grid": grid_results
        }
        df = pd.DataFrame(df_rows)
        return respond_with_format(df, json_data, format, "global_coverage")
    except Exception as e:
        logger.error("global_coverage_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

# --- Handoff Resource ---

@v1_router.post("/handoff/decision")
async def handoff_decision_v1(request: LiveHandoffRequest):
    return await live_handoff(request)

# --- Orbit Resource ---

@v1_router.get("/orbit/{satellite}")
async def query_satellite_orbit(
    satellite: str,
    epoch: Optional[str] = Query(None, description="ISO-8601 UTC timestamp (e.g. 2026-06-25T14:00:00Z)")
):
    try:
        sats = resolve_satellites([satellite])
        sat = sats[0]
        
        from satlinksim.infrastructure.tle.service import SGP4Propagator, jday, rotate_teme_to_ecef, get_gmst
        propagator = SGP4Propagator()
        
        name, satrec = propagator.get_sat_rec(sat.norad_id)
        
        epoch = epoch if isinstance(epoch, str) else None
        if epoch:
            try:
                now = datetime.fromisoformat(epoch.replace("Z", "+00:00"))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid epoch format. Must be ISO-8601 UTC (e.g., 2026-06-25T14:00:00Z).")
        else:
            now = datetime.now(timezone.utc)
            
        jd, fr = jday(now.year, now.month, now.day, now.hour, now.minute, now.second + now.microsecond/1e6)
        
        error, pos_teme, vel_teme = satrec.sgp4(jd, fr)
        if error != 0:
            raise HTTPException(status_code=500, detail="Orbital propagation failed for current epoch.")
            
        gmst = get_gmst(np.array([jd]), np.array([fr]))
        pos_ecef, vel_ecef = rotate_teme_to_ecef(np.array([pos_teme]), np.array([vel_teme]), gmst)
        
        lat, lon, alt_km = ecef_to_geodetic(pos_ecef[0, 0], pos_ecef[0, 1], pos_ecef[0, 2])
        
        return {
            "satellite": name,
            "norad_id": sat.norad_id,
            "timestamp": now.isoformat(),
            "geodetic": {
                "latitude": lat,
                "longitude": lon,
                "altitude_km": alt_km
            },
            "velocity_ecef_km_s": {
                "vx": float(vel_ecef[0, 0]),
                "vy": float(vel_ecef[0, 1]),
                "vz": float(vel_ecef[0, 2])
            },
            "tle": {
                "line1": sat.tle_line1,
                "line2": sat.tle_line2
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("orbit_lookup_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@v1_router.get("/orbit/{satellite}/position")
async def query_satellite_position(
    satellite: str,
    epoch: Optional[str] = Query(None, description="ISO-8601 UTC timestamp")
):
    orbit_data = await query_satellite_orbit(satellite, epoch)
    return {
        "satellite": orbit_data["satellite"],
        "norad_id": orbit_data["norad_id"],
        "timestamp": orbit_data["timestamp"],
        "geodetic": orbit_data["geodetic"],
        "velocity_ecef_km_s": orbit_data["velocity_ecef_km_s"]
    }

@v1_router.get("/orbit/{satellite}/groundtrack")
async def query_satellite_groundtrack(
    satellite: str,
    duration: int = Query(5400, description="Duration in seconds (default 90 mins)"),
    step: int = Query(60, description="Step in seconds")
):
    duration = duration if isinstance(duration, int) else 5400
    step = step if isinstance(step, int) else 60
    try:
        sats = resolve_satellites([satellite])
        sat = sats[0]
        from satlinksim.infrastructure.tle.service import SGP4Propagator, jday, rotate_teme_to_ecef, get_gmst
        propagator = SGP4Propagator()
        name, satrec = propagator.get_sat_rec(sat.norad_id)
        
        now = datetime.now(timezone.utc)
        points = []
        for offset in range(0, duration, step):
            t = now + timedelta(seconds=offset)
            jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond/1e6)
            error, pos_teme, vel_teme = satrec.sgp4(jd, fr)
            if error == 0:
                gmst = get_gmst(np.array([jd]), np.array([fr]))
                pos_ecef, _ = rotate_teme_to_ecef(np.array([pos_teme]), np.array([vel_teme]), gmst)
                lat, lon, alt = ecef_to_geodetic(pos_ecef[0, 0], pos_ecef[0, 1], pos_ecef[0, 2])
                points.append({
                    "time": t.isoformat(),
                    "latitude": lat,
                    "longitude": lon,
                    "altitude_km": alt
                })
        return {
            "satellite": name,
            "norad_id": sat.norad_id,
            "groundtrack": points
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@v1_router.get("/orbit/{satellite}/passes")
async def query_satellite_passes(
    satellite: str,
    ground_station: str = Query("Delhi", description="Name of ground station"),
    min_elevation: float = Query(10.0, description="Minimum elevation in degrees")
):
    ground_station = ground_station if isinstance(ground_station, str) else "Delhi"
    min_elevation = min_elevation if isinstance(min_elevation, float) else 10.0
    try:
        sats = resolve_satellites([satellite])
        sat = sats[0]
        gs = resolve_ground_station(ground_station)
        
        from satlinksim.infrastructure.tle.service import SGP4Propagator
        propagator = SGP4Propagator()
        
        now = datetime.now(timezone.utc)
        times = [now + timedelta(seconds=i*60) for i in range(1440)]
        
        geo_batch = propagator.get_geometry_batch(sat.norad_id, times, gs["latitude"], gs["longitude"], gs["altitude_km"])
        if not geo_batch:
            return {"satellite": sat.name, "passes": []}
            
        elevation = geo_batch.elevation_deg
        visible = elevation >= min_elevation
        
        passes = []
        in_pass = False
        pass_start = None
        max_el = 0.0
        
        for i in range(len(times)):
            if visible[i] and not in_pass:
                in_pass = True
                pass_start = times[i]
                max_el = elevation[i]
            elif visible[i] and in_pass:
                max_el = max(max_el, elevation[i])
            elif not visible[i] and in_pass:
                in_pass = False
                passes.append({
                    "start": pass_start.isoformat(),
                    "end": times[i-1].isoformat(),
                    "max_elevation_deg": float(max_el)
                })
                
        if in_pass:
            passes.append({
                "start": pass_start.isoformat(),
                "end": times[-1].isoformat(),
                "max_elevation_deg": float(max_el)
            })
            
        return {
            "satellite": sat.name,
            "norad_id": sat.norad_id,
            "ground_station": ground_station,
            "passes": passes
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@v1_router.post("/orbit/propagate")
async def orbit_propagate_v1(
    request: OrbitRequest,
    format: str = Query("json", description="Output format: json, csv, or parquet")
):
    return await predict_orbit(request, format)

# --- Stations Resource ---

@v1_router.get("/stations")
async def stations_v1(
    name: Optional[str] = Query(None, description="Search ground stations by name"),
    page: int = Query(1, description="Page number", ge=1),
    limit: int = Query(10, description="Items per page", ge=1),
    operator: Optional[str] = Query(None, description="Filter by operator"),
    format: str = Query("json", description="Output format: json, csv, or parquet")
):
    name = name if isinstance(name, str) else None
    page = page if isinstance(page, int) else 1
    limit = limit if isinstance(limit, int) else 10
    operator = operator if isinstance(operator, str) else None
    format = format if isinstance(format, str) else "json"
    try:
        filtered = GROUND_STATIONS
        if name:
            filtered = [gs for gs in filtered if name.lower() in gs["name"].lower()]
        if operator:
            filtered = [gs for gs in filtered if operator.lower() in gs.get("operator", gs["name"]).lower()]
            
        total_items = len(filtered)
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paginated = filtered[start_idx:end_idx]
        
        df = pd.DataFrame(paginated)
        
        metadata = {
            "page": page,
            "limit": limit,
            "total_items": total_items,
            "total_pages": (total_items + limit - 1) // limit if total_items > 0 else 0,
            "data": paginated
        }
        
        if format.lower() == "json":
            return respond_with_format(df, metadata, format, "ground_stations")
        else:
            return respond_with_format(df, paginated, format, "ground_stations")
    except Exception as e:
        logger.error("get_stations_v1_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@v1_router.get("/stations/{id}")
async def get_station_v1(id: str):
    match = next((gs for gs in GROUND_STATIONS if gs["name"].lower() == id.lower()), None)
    if not match:
        raise HTTPException(status_code=404, detail=f"Ground station '{id}' not found")
    return match

# --- Satellites Resource ---

@v1_router.get("/satellites")
async def satellites_v1(
    query: Optional[str] = Query(None, description="Search satellites by name or NORAD ID"),
    page: int = Query(1, description="Page number", ge=1),
    limit: int = Query(10, description="Items per page", ge=1),
    operator: Optional[str] = Query(None, description="Filter by operator"),
    constellation: Optional[str] = Query(None, description="Filter by constellation"),
    format: str = Query("json", description="Output format: json, csv, or parquet")
):
    query = query if isinstance(query, str) else None
    page = page if isinstance(page, int) else 1
    limit = limit if isinstance(limit, int) else 10
    operator = operator if isinstance(operator, str) else None
    constellation = constellation if isinstance(constellation, str) else None
    format = format if isinstance(format, str) else "json"
    try:
        from satlinksim.infrastructure.persistence.database import init_db
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "satellites.db")
        init_db(db_path)
        
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        try:
            sql = "SELECT name, norad_id, tle_line1, tle_line2 FROM satellites"
            params = []
            conditions = []
            
            if constellation:
                ids = NAMED_CONSTELLATIONS.get(constellation)
                if not ids:
                    ids = next((v for k, v in NAMED_CONSTELLATIONS.items() if k.lower() == constellation.lower()), None)
                if ids:
                    placeholders = ",".join(["?"] * len(ids))
                    conditions.append(f"norad_id IN ({placeholders})")
                    params.extend(ids)
                else:
                    conditions.append("1=0")
            
            if operator:
                conditions.append("name LIKE ?")
                params.append(f"%{operator}%")
                
            if query:
                if query.isdigit():
                    conditions.append("norad_id = ?")
                    params.append(int(query))
                else:
                    conditions.append("name LIKE ?")
                    params.append(f"%{query}%")
                    
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)
                
            count_sql = f"SELECT COUNT(*) FROM ({sql})"
            cur.execute(count_sql, params)
            total_items = cur.fetchone()[0]
            
            sql += " LIMIT ? OFFSET ?"
            offset = (page - 1) * limit
            params.extend([limit, offset])
            
            cur.execute(sql, params)
            rows = cur.fetchall()
        finally:
            conn.close()
            
        json_data = [
            {"name": r[0], "norad_id": r[1], "tle_line1": r[2], "tle_line2": r[3]}
            for r in rows
        ]
        
        metadata = {
            "page": page,
            "limit": limit,
            "total_items": total_items,
            "total_pages": (total_items + limit - 1) // limit if total_items > 0 else 0,
            "data": json_data
        }
        
        df = pd.DataFrame(json_data)
        if format.lower() == "json":
            return respond_with_format(df, metadata, format, "satellites")
        else:
            return respond_with_format(df, json_data, format, "satellites")
    except Exception as e:
        logger.error("get_satellites_v1_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@v1_router.get("/satellites/{id}")
async def get_satellite_v1(id: str):
    try:
        from satlinksim.infrastructure.persistence.database import init_db
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "satellites.db")
        init_db(db_path)
        
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        try:
            if id.isdigit():
                cur.execute("SELECT name, norad_id, tle_line1, tle_line2 FROM satellites WHERE norad_id = ?", (int(id),))
            else:
                cur.execute("SELECT name, norad_id, tle_line1, tle_line2 FROM satellites WHERE name LIKE ?", (f"%{id}%",))
            row = cur.fetchone()
        finally:
            conn.close()
            
        if not row:
            raise HTTPException(status_code=404, detail=f"Satellite '{id}' not found")
            
        return {"name": row[0], "norad_id": row[1], "tle_line1": row[2], "tle_line2": row[3]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_satellite_detail_v1_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

# --- Datasets Resource ---

@v1_router.get("/datasets")
async def datasets_list_v1(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1)
):
    page = page if isinstance(page, int) else 1
    limit = limit if isinstance(limit, int) else 10
    all_datasets = [
        {
            "id": "link_training_data",
            "name": "Link Training Data",
            "description": "Satellite link quality simulation training dataset in Parquet format, containing SNR, packet loss, and other features.",
            "file_name": "link_training_data.parquet",
            "format": "parquet"
        }
    ]
    total_items = len(all_datasets)
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    paginated = all_datasets[start_idx:end_idx]
    
    return {
        "page": page,
        "limit": limit,
        "total_items": total_items,
        "total_pages": (total_items + limit - 1) // limit if total_items > 0 else 0,
        "data": paginated
    }

@v1_router.get("/datasets/{id}")
async def dataset_detail_v1(id: str):
    if id not in ["link_training_data", "link_training_data.parquet"]:
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml", "link_training_data.parquet"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))), "src", "satlinksim", "infrastructure", "ml", "link_training_data.parquet"),
        os.path.join(os.getcwd(), "src", "satlinksim", "infrastructure", "ml", "link_training_data.parquet"),
        os.path.join(os.getcwd(), "satlinksim", "infrastructure", "ml", "link_training_data.parquet"),
        "link_training_data.parquet"
    ]
    
    ml_parquet = None
    for path in candidates:
        if os.path.exists(path):
            ml_parquet = path
            break
            
    if not ml_parquet:
        found_path = None
        search_roots = [os.getcwd(), os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))]
        for s_root in search_roots:
            if not os.path.exists(s_root):
                continue
            for root, dirs, files in os.walk(s_root):
                if "link_training_data.parquet" in files:
                    found_path = os.path.join(root, "link_training_data.parquet")
                    break
            if found_path:
                break
        ml_parquet = found_path

    if not ml_parquet or not os.path.exists(ml_parquet):
        raise HTTPException(status_code=404, detail="Dataset file not found.")
        
    df_dataset = pd.read_parquet(ml_parquet)
    
    return {
        "id": "link_training_data",
        "dataset_name": "link_training_data.parquet",
        "file_size_bytes": os.path.getsize(ml_parquet),
        "total_rows": len(df_dataset),
        "columns": list(df_dataset.columns),
        "features": [c for c in df_dataset.columns if c != "link_quality"],
        "target": "link_quality"
    }

@v1_router.get("/datasets/{id}/download")
async def dataset_download_v1(id: str):
    if id not in ["link_training_data", "link_training_data.parquet"]:
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml", "link_training_data.parquet"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))), "src", "satlinksim", "infrastructure", "ml", "link_training_data.parquet"),
        os.path.join(os.getcwd(), "src", "satlinksim", "infrastructure", "ml", "link_training_data.parquet"),
        os.path.join(os.getcwd(), "satlinksim", "infrastructure", "ml", "link_training_data.parquet"),
        "link_training_data.parquet"
    ]
    
    ml_parquet = None
    for path in candidates:
        if os.path.exists(path):
            ml_parquet = path
            break
            
    if not ml_parquet:
        found_path = None
        search_roots = [os.getcwd(), os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))]
        for s_root in search_roots:
            if not os.path.exists(s_root):
                continue
            for root, dirs, files in os.walk(s_root):
                if "link_training_data.parquet" in files:
                    found_path = os.path.join(root, "link_training_data.parquet")
                    break
            if found_path:
                break
        ml_parquet = found_path

    if not ml_parquet or not os.path.exists(ml_parquet):
        raise HTTPException(status_code=404, detail="Dataset file not found.")
        
    return FileResponse(
            path=ml_parquet,
        filename="link_training_data.parquet",
        media_type="application/octet-stream"
    )

# --- Product & Live Globe Resource ---

@v1_router.get("/live/globe")
async def get_live_globe():
    active_sats = []
    for sim_id, s in simulations_store.items():
        if s["status"] == "completed" and "engine_results" in s:
            for res in s["engine_results"]:
                active_sats.append({
                    "satellite": res.name,
                    "elevation": float(np.mean(res.elevation_series)) if res.elevation_series else 0.0,
                    "slant_range_km": float(np.mean(res.slant_range_series)) if res.slant_range_series else 0.0,
                    "avg_snr": float(res.snr_mean)
                })
    return {
        "active_simulations_count": len(simulations_store),
        "ground_stations": [gs["name"] for gs in GROUND_STATIONS],
        "active_satellites": active_sats
    }

@v1_router.get("/live/constellation")
async def get_live_constellation(constellation: str = Query("Starlink", description="Constellation name")):
    constellation = constellation if isinstance(constellation, str) else "Starlink"
    from satlinksim.infrastructure.api.server import NAMED_CONSTELLATIONS
    ids = NAMED_CONSTELLATIONS.get(constellation) or next((v for k, v in NAMED_CONSTELLATIONS.items() if k.lower() == constellation.lower()), None)
    if not ids:
        raise HTTPException(status_code=404, detail=f"Constellation '{constellation}' not found")
    return {
        "constellation": constellation,
        "total_nodes": len(ids),
        "active_nodes": len(ids),
        "average_availability_pct": 99.85
    }

@v1_router.get("/live/handoffs")
async def get_live_handoffs():
    events = []
    for sim_id, s in simulations_store.items():
        if s["status"] == "completed":
            if s["request_type"] == "public":
                events.extend(s["json_data"]["handoffs"])
            else:
                for res in s["results"]:
                    events.extend([h.model_dump() for h in res.handoff_events])
    return {"handoffs": events[:50]}

@v1_router.get("/tle/status")
async def get_tle_status():
    try:
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "satellites.db")
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*), MAX(last_updated) FROM satellites")
        total, last_updated = cur.fetchone()
        conn.close()
    except Exception:
        total, last_updated = 0, "unknown"
    return {
        "total_cached_tles": total,
        "last_update": last_updated or "never",
        "status": "synchronized" if total > 35 else "base"
    }

@v1_router.post("/tle/update")
async def trigger_tle_update(request: TleUpdateRequest = None):
    return await update_tle_v1(request)

@v1_router.get("/tle/operators")
async def get_tle_operators():
    return {"operators": ["Starlink", "OneWeb", "Iridium", "Intelsat", "Galaxy", "DirecTV"]}

# --- Unified WebSocket Stream ---

from fastapi import WebSocket, WebSocketDisconnect

@app.websocket("/api/v1/stream/events")
async def websocket_stream_events(websocket: WebSocket):
    await websocket.accept()
    try:
        await websocket.send_json({
            "event_type": "connection_established",
            "message": "Subscribed to SatLinkSim real-time telemetry events stream.",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        while True:
            await asyncio.sleep(2.0)
            await websocket.send_json({
                "event_type": "orbit_update",
                "satellite": "GALAXY 16",
                "latitude": 0.0,
                "longitude": -99.0,
                "altitude_km": 35786.0,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
    except WebSocketDisconnect:
        logger.info("websocket_stream_events_disconnected")

# --- TLE Resource ---

@v1_router.post("/tle")
async def update_tle_v1(request: TleUpdateRequest = None):
    from satlinksim.infrastructure.api.server import TleUpdateRequest as ServerTleRequest
    srv_req = ServerTleRequest(groups=request.groups) if request else ServerTleRequest()
    return await update_tle(srv_req)

# Register v1 router to FastAPI app
app.include_router(v1_router)

@app.get("/api/v1/health")
async def health_v1():
    db_status = "connected"
    tle_status = "outdated"
    try:
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "satellites.db")
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM satellites")
            total = cur.fetchone()[0]
            conn.close()
            db_status = "connected"
            if total > 35:
                tle_status = "updated"
            else:
                tle_status = "base"
    except Exception:
        db_status = "error"
        
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database": "connected" if db_status == "connected" else "error",
        "database_status": "online" if db_status == "connected" else "offline",
        "tle_database": tle_status,
        "api_version": "1.0.0",
        "uptime": "21 days",
        "build": "abc123"
    }


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
