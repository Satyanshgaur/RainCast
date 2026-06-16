import uuid
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from satlinksim.application.simulation_engine import SimulationEngine
from satlinksim.infrastructure.api.schemas import (
    SimulationRequest, SimulationResponse, StationResultSchema, HandoffEventSchema,
    SummarySimulationRequest, SummarySimulationResponse, JobResponse, JobStatus
)
from satlinksim.ground_stations import GROUND_STATIONS
from satlinksim.domain.models import Constellation, Satellite
from satlinksim.infrastructure.logging import configure_logging, get_logger
from satlinksim.infrastructure.metrics import (
    metrics_app, SIMULATIONS_RUN, SIMULATION_LATENCY
)
from typing import List, Dict, Optional
import time
import uvicorn
import structlog

# Initialize logging
configure_logging()
logger = get_logger("satlinksim.api")

app = FastAPI(title="SatLinkSim API")
engine = SimulationEngine()

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

@app.post("/simulate", response_model=SimulationResponse)
async def simulate(request: SimulationRequest):
    start_time = time.time()
    SIMULATIONS_RUN.labels(mode="sync").inc()
    
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

@app.get("/health")
async def health():
    return {"status": "ok"}

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
