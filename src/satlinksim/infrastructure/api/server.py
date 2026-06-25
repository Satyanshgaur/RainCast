import uuid
import asyncio
import os
import sqlite3
import io
import time
from typing import List, Dict, Optional, Any, Union
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np
import uvicorn
import structlog
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Response, Query

from satlinksim.application.simulation_engine import SimulationEngine
from satlinksim.infrastructure.api.schemas import (
    SimulationRequest, SimulationResponse, StationResultSchema, HandoffEventSchema,
    SummarySimulationRequest, SummarySimulationResponse, JobResponse, JobStatus,
    PublicSimulationRequest, LinkBudgetRequest, AttenuationRequest,
    VisibilityRequest, AvailabilityRequest
)
from satlinksim.ground_stations import GROUND_STATIONS
from satlinksim.domain.models import Constellation, Satellite, StationResult
from satlinksim.domain.link.itu_models import itu_rain_coefficients, gaseous_absorption_db
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

@app.get("/health")
async def health():
    return {"status": "ok"}

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
