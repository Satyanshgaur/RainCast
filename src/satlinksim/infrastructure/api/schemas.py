from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union
from datetime import datetime

class ItuRain(BaseModel):
    R001: float
    R01: float
    R1: float
    P_rain: float

class GroundStation(BaseModel):
    name: str
    eirp_dbw: float
    g_rx_dbi: float
    system_temp_k: float
    antenna_diam_m: float
    latitude: float
    longitude: float
    altitude_km: float
    itu_rain: ItuRain
    wv_g_m3: float
    humidity_pct: float
    v_radial_ms: float
    norad_id: Optional[int] = None
    sat_name: Optional[str] = None
    sat_lon_deg: Optional[float] = None

class SatelliteSchema(BaseModel):
    norad_id: int
    name: str
    tle_line1: Optional[str] = None
    tle_line2: Optional[str] = None

class ConstellationSchema(BaseModel):
    name: str
    satellites: List[SatelliteSchema]

class SimulationRequest(BaseModel):
    ground_stations: List[GroundStation]
    n_steps: int = 3600
    dt_s: float = 1.0
    start_time: Optional[datetime] = None
    force_rain: bool = False
    seed: Optional[int] = None
    freq_hz: float = 14e9
    eirp_offset_db: float = 0.0
    bandwidth_hz: float = 36e6
    polarization: str = "vertical"
    rain_rate_scale: float = 1.0
    constellation: Optional[ConstellationSchema] = None
    handoff_policy: str = "highest_elevation"
    hysteresis: float = 0.5
    min_dwell_steps: int = 10

class HandoffEventSchema(BaseModel):
    time_step: int
    old_sat: str
    new_sat: str
    reason: str
    metric_delta: float

class StationResultSchema(BaseModel):
    name: str
    elevation: float
    slant_km: float
    doppler_hz: float
    path_loss: float
    gas_loss: float
    rain_height: float
    eff_path: float
    itu_k: float
    itu_alpha: float
    scint_sig: float
    noise_floor: float
    snr_series: List[float]
    rain_series: List[float]
    rain_db_series: List[float]
    scint_series: List[float]
    pkt_loss_series: List[float]
    elevation_series: List[float]
    slant_range_series: List[float]
    doppler_series: List[float]
    snr_mean: float
    snr_min: float
    snr_std: float
    snr_p10: float
    rain_fraction: float
    avg_rain_db: float
    avg_pkt_loss: float
    outage_fraction: float
    sat_name_series: Optional[List[str]] = None
    handoff_events: List[HandoffEventSchema]

class SimulationResponse(BaseModel):
    results: List[StationResultSchema]

class SummarySimulationRequest(BaseModel):
    station: str
    constellation: str
    duration_min: int = Field(default=1440, alias="duration")

    class Config:
        populate_by_name = True

class SummarySimulationResponse(BaseModel):
    availability: float
    handoffs: int

class JobResponse(BaseModel):
    job_id: str

class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, running, completed, failed
    result: Optional[SimulationResponse] = None
    error: Optional[str] = None
