import random
import numpy as np
import asyncio
import pickle
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Union, Callable

from satlinksim.domain.models import StationResult, Constellation
from satlinksim.domain.interfaces import Propagator, RainModel
from satlinksim.domain.rain.engine import CorrelatedRainProcess
from satlinksim.domain.geometry.physics import geo_elevation_deg, geo_slant_range_km
from satlinksim.domain.link.itu_models import (
    itu_rain_coefficients, itu_rain_height, effective_path_length,
    rain_attenuation_db, gaseous_absorption_db, scintillation_sigma_db
)
from satlinksim.domain.link.budget import (
    fspl_db, noise_power_dbw, doppler_shift_hz, packet_loss_from_snr
)
from satlinksim.domain.handoff.manager import HandoffManager, HandoffPolicy, HighestElevationPolicy, HighestSNRPolicy
from satlinksim.infrastructure.tle.service import SGP4Propagator
from satlinksim.config import config
from satlinksim.ground_stations import GROUND_STATIONS

# Defaults
DEFAULT_CARRIER_FREQ_HZ = config.simulation.link.carrier_freq_hz
DEFAULT_BANDWIDTH_HZ    = config.simulation.link.bandwidth_hz
DEFAULT_POLARIZATION    = config.simulation.link.polarization
DEFAULT_DT_S            = config.simulation.dt_s
DEFAULT_N_STEPS         = config.simulation.n_steps
SNR_THRESHOLD_DB        = config.simulation.link.snr_threshold_db

class SimulationEngine:
    def __init__(self, propagator: Optional[Propagator] = None):
        self.propagator = propagator or SGP4Propagator()

    def resume(self, checkpoint_file: str = "checkpoint.pkl") -> list[StationResult]:
        """Resume a simulation from a saved checkpoint."""
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"Checkpoint file {checkpoint_file} not found.")
        with open(checkpoint_file, "rb") as f:
            state = pickle.load(f)
        return self.simulate_all_batched(**state["kwargs"], _resume_state=state)

    def simulate_all_batched(self, ground_stations: list[dict],
                             n_steps:         int   = DEFAULT_N_STEPS,
                             dt_s:            float = DEFAULT_DT_S,
                             start_time:      Optional[datetime] = None,
                             force_rain:      bool  = False,
                             seed:            Optional[int] = None,
                             freq_hz:         float = DEFAULT_CARRIER_FREQ_HZ,
                             eirp_offset_db:  float = 0.0,
                             bandwidth_hz:    float = DEFAULT_BANDWIDTH_HZ,
                             polarization:    str   = DEFAULT_POLARIZATION,
                             rain_rate_scale: float = 1.0,
                             constellation:   Optional[Constellation] = None,
                             handoff_policy:  Union[str, HandoffPolicy] = "highest_elevation",
                             hysteresis:      float = config.simulation.handoff.hysteresis_db,
                             min_dwell_steps: int = config.simulation.handoff.dwell_steps,
                             rain_model_factory: Optional[Callable[[dict], RainModel]] = None,
                             checkpoint_interval: int = 10000,
                             checkpoint_file: str = "checkpoint.pkl",
                             _resume_state: dict = None
                             ) -> list[StationResult]:
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        if _resume_state:
            results = _resume_state["results"]
            start_gs_idx = _resume_state["gs_idx"]
            start_chunk_idx = _resume_state["chunk_idx"]
            rain_proc = _resume_state.get("rain_proc")
            hm = _resume_state.get("hm")
            acc_state = _resume_state.get("acc_state")
            kwargs_to_save = _resume_state["kwargs"]
        else:
            results = []
            start_gs_idx = 0
            start_chunk_idx = 0
            rain_proc = None
            hm = None
            acc_state = None
            # Exclude callables from kwargs
            kwargs_to_save = {
                "ground_stations": ground_stations, "n_steps": n_steps, "dt_s": dt_s, 
                "start_time": start_time, "force_rain": force_rain, "seed": seed, 
                "freq_hz": freq_hz, "eirp_offset_db": eirp_offset_db, 
                "bandwidth_hz": bandwidth_hz, "polarization": polarization, 
                "rain_rate_scale": rain_rate_scale, "constellation": constellation, 
                "handoff_policy": handoff_policy, "hysteresis": hysteresis, 
                "min_dwell_steps": min_dwell_steps, "checkpoint_interval": checkpoint_interval,
                "checkpoint_file": checkpoint_file
            }

        curr_time = start_time or datetime.now(timezone.utc)
        times = [curr_time + timedelta(seconds=i*dt_s) for i in range(n_steps)]
        freq_ghz = freq_hz / 1e9

        itu_k, itu_a = itu_rain_coefficients(freq_ghz, polarization)

        for i in range(start_gs_idx, len(ground_stations)):
            gs = ground_stations[i]
            noise_dbw = noise_power_dbw(gs["system_temp_k"], bandwidth_hz)
            eirp = gs["eirp_dbw"] + eirp_offset_db
            g_rx = gs["g_rx_dbi"]
            rain_h = itu_rain_height(gs["latitude"])

            candidates_geo = []
            if constellation:
                for sat in constellation.satellites:
                    geo = self.propagator.get_geometry_batch(sat, times, gs["latitude"], gs["longitude"], gs["altitude_km"])
                    if geo: candidates_geo.append(geo)
            else:
                sat_id = gs.get("norad_id") or gs.get("sat_name")
                if sat_id:
                    geo = self.propagator.get_geometry_batch(sat_id, times, gs["latitude"], gs["longitude"], gs["altitude_km"])
                    if geo: candidates_geo.append(geo)

            if not candidates_geo:
                # Fallback to GEO
                el_s = np.full(n_steps, geo_elevation_deg(gs["latitude"], gs["longitude"], gs.get("sat_lon_deg", 0)))
                slant_s = np.full(n_steps, geo_slant_range_km(gs["latitude"], gs["longitude"], gs.get("sat_lon_deg", 0)))
                dop_s = np.full(n_steps, doppler_shift_hz(gs.get("v_radial_ms", 0), freq_hz))
                sat_names = [gs.get("sat_name") or f"SAT-LAT{gs.get('sat_lon_deg',0)}"] * n_steps
                handoff_events = []
                rain_rate_s = np.zeros(n_steps).tolist()
                rain_db_s = np.zeros(n_steps).tolist()
                scint_s = np.zeros(n_steps).tolist()
                snr_s = np.full(n_steps, eirp - fspl_db(freq_hz, slant_s[0]) + g_rx - noise_dbw)
                pkt_s = packet_loss_from_snr(snr_s, SNR_THRESHOLD_DB).tolist()
                snr_s = snr_s.tolist()
                el_s = el_s.tolist()
                slant_s = slant_s.tolist()
                dop_s = dop_s.tolist()
                sorted_snr = sorted(snr_s)
                p10_idx = max(0, int(0.10 * n_steps) - 1)
            else:
                n_cands = len(candidates_geo)
                cand_names = [g.sat_name for g in candidates_geo]

                if rain_proc is None:
                    if rain_model_factory:
                        rain_proc = rain_model_factory(gs)
                    else:
                        rain_proc = CorrelatedRainProcess([gs], dt_s=dt_s, force_rain=force_rain, 
                                                          rain_rate_scale=rain_rate_scale)

                if hm is None:
                    if isinstance(handoff_policy, str):
                        if handoff_policy == "highest_snr":
                            policy_obj = HighestSNRPolicy()
                        else:
                            policy_obj = HighestElevationPolicy()
                    else:
                        policy_obj = handoff_policy
                    hm = HandoffManager(policy=policy_obj, hysteresis=hysteresis, min_dwell_steps=min_dwell_steps)

                if acc_state is None:
                    acc_state = {
                        "el_s": [], "slant_s": [], "dop_s": [], "snr_s": [],
                        "rain_db_s": [], "scint_db_s": [], "sat_names": [],
                        "rain_rate_s": []
                    }

                for chunk_start in range(start_chunk_idx * checkpoint_interval, n_steps, checkpoint_interval):
                    chunk_end = min(n_steps, chunk_start + checkpoint_interval)
                    c_steps = chunk_end - chunk_start
                    
                    rain_rate_chunk = rain_proc.generate_batch(c_steps)[:, 0]
                    acc_state["rain_rate_s"].extend(rain_rate_chunk.tolist())

                    cand_snr_matrix = np.zeros((n_cands, c_steps))
                    cand_el_matrix = np.zeros((n_cands, c_steps))
                    cand_slant_matrix = np.zeros((n_cands, c_steps))
                    cand_dop_matrix = np.zeros((n_cands, c_steps))
                    cand_rain_db_matrix = np.zeros((n_cands, c_steps))
                    cand_scint_db_matrix = np.zeros((n_cands, c_steps))
                    
                    for c_idx, geo in enumerate(candidates_geo):
                        g_slant = geo.slant_range_km[chunk_start:chunk_end]
                        g_el = geo.elevation_deg[chunk_start:chunk_end]
                        g_rad = geo.radial_velocity_ms[chunk_start:chunk_end]
                        
                        pl = fspl_db(freq_hz, g_slant)
                        gl = gaseous_absorption_db(freq_ghz, g_el, gs["wv_g_m3"])
                        ep = effective_path_length(g_el, rain_h, gs["altitude_km"], itu_k)
                        ra = rain_attenuation_db(rain_rate_chunk, itu_k, itu_a, ep)
                        ss = scintillation_sigma_db(freq_ghz, g_el, gs["antenna_diam_m"], gs["humidity_pct"])
                        scint_db = np.random.normal(0, ss)
                        
                        snr = eirp - pl - gl - ra - scint_db + g_rx - noise_dbw
                        cand_snr_matrix[c_idx] = snr
                        cand_el_matrix[c_idx] = g_el
                        cand_slant_matrix[c_idx] = g_slant
                        cand_dop_matrix[c_idx] = doppler_shift_hz(g_rad, freq_hz)
                        cand_rain_db_matrix[c_idx] = ra
                        cand_scint_db_matrix[c_idx] = scint_db

                    selected_indices = []
                    for t in range(c_steps):
                        idx = hm.select(chunk_start + t, cand_names, cand_snr_matrix[:, t], cand_el_matrix[:, t])
                        selected_indices.append(idx)
                    
                    t_idx = np.arange(c_steps)
                    s_idx = np.array(selected_indices)
                    
                    acc_state["el_s"].extend(cand_el_matrix[s_idx, t_idx].tolist())
                    acc_state["slant_s"].extend(cand_slant_matrix[s_idx, t_idx].tolist())
                    acc_state["dop_s"].extend(cand_dop_matrix[s_idx, t_idx].tolist())
                    acc_state["snr_s"].extend(cand_snr_matrix[s_idx, t_idx].tolist())
                    acc_state["rain_db_s"].extend(cand_rain_db_matrix[s_idx, t_idx].tolist())
                    acc_state["scint_db_s"].extend(cand_scint_db_matrix[s_idx, t_idx].tolist())
                    acc_state["sat_names"].extend([cand_names[j] for j in s_idx])
                    
                    # CHECKPOINTING
                    if chunk_end < n_steps:
                        state = {
                            "kwargs": kwargs_to_save,
                            "results": results,
                            "gs_idx": i,
                            "chunk_idx": (chunk_start // checkpoint_interval) + 1,
                            "rain_proc": rain_proc,
                            "hm": hm,
                            "acc_state": acc_state
                        }
                        with open(checkpoint_file, "wb") as f:
                            pickle.dump(state, f)

                el_s = acc_state["el_s"]
                slant_s = acc_state["slant_s"]
                dop_s = acc_state["dop_s"]
                snr_s = acc_state["snr_s"]
                rain_db_s = acc_state["rain_db_s"]
                scint_s = acc_state["scint_db_s"]
                sat_names = acc_state["sat_names"]
                rain_rate_s = acc_state["rain_rate_s"]
                handoff_events = hm.events

                pkt_s = packet_loss_from_snr(np.array(snr_s), SNR_THRESHOLD_DB).tolist()

                sorted_snr = sorted(snr_s)
                p10_idx = max(0, int(0.10 * n_steps) - 1)

            results.append(StationResult(
                name=gs["name"], elevation=el_s[0], slant_km=slant_s[0], doppler_hz=dop_s[0],
                path_loss=fspl_db(freq_hz, slant_s[0]),
                gas_loss=gaseous_absorption_db(freq_ghz, el_s[0], gs["wv_g_m3"]),
                rain_height=rain_h,
                eff_path=effective_path_length(el_s[0], rain_h, gs["altitude_km"], itu_k),
                itu_k=itu_k, itu_alpha=itu_a,
                scint_sig=scintillation_sigma_db(freq_ghz, el_s[0], gs["antenna_diam_m"], gs["humidity_pct"]),
                noise_floor=noise_dbw,
                snr_series=snr_s, rain_series=rain_rate_s, rain_db_series=rain_db_s,
                scint_series=scint_s, pkt_loss_series=pkt_s,
                elevation_series=el_s, slant_range_series=slant_s, doppler_series=dop_s,
                snr_mean=float(np.mean(snr_s)),
                snr_min=float(np.min(snr_s)),
                snr_std=float(np.std(snr_s, ddof=1)) if len(snr_s) > 1 else 0.0,
                snr_p10=float(sorted_snr[p10_idx]),
                rain_fraction=float(np.sum(np.array(rain_rate_s) > 0) / n_steps),
                avg_rain_db=float(np.mean([db for db in rain_db_s if db > 0])) if any(db > 0 for db in rain_db_s) else 0.0,
                avg_pkt_loss=float(np.mean(pkt_s)),
                outage_fraction=float(np.mean(pkt_s)),
                sat_name_series=sat_names,
                handoff_events=handoff_events
            ))
            
            # Reset state for next station
            rain_proc = None
            hm = None
            acc_state = None
            start_chunk_idx = 0

            # Checkpoint at end of station
            if i < len(ground_stations) - 1:
                state = {
                    "kwargs": kwargs_to_save,
                    "results": results,
                    "gs_idx": i + 1,
                    "chunk_idx": 0,
                    "rain_proc": None,
                    "hm": None,
                    "acc_state": None
                }
                with open(checkpoint_file, "wb") as f:
                    pickle.dump(state, f)

        # Cleanup checkpoint if finished successfully
        if os.path.exists(checkpoint_file):
            try:
                os.remove(checkpoint_file)
            except OSError:
                pass

        return results

    async def simulate_all_concurrent(self, ground_stations: List[Dict], 
                                      constellation: Optional[Constellation] = None,
                                      **kwargs) -> List[StationResult]:
        n_steps = kwargs.get("n_steps", DEFAULT_N_STEPS)
        dt_s = kwargs.get("dt_s", DEFAULT_DT_S)
        start_time = kwargs.get("start_time") or datetime.now(timezone.utc)
        times = [start_time + timedelta(seconds=i*dt_s) for i in range(n_steps)]
        
        tasks = []
        for gs in ground_stations:
            if constellation:
                tasks.append(asyncio.sleep(0, result=None))
            else:
                sat_id = gs.get("norad_id") or gs.get("sat_name")
                if sat_id:
                    tasks.append(self.propagator.get_geometry_batch_async(
                        sat_id, times, gs["latitude"], gs["longitude"], gs["altitude_km"]
                    ))
                else:
                    tasks.append(asyncio.sleep(0, result=None))
                
        await asyncio.gather(*tasks)
        
        return self.simulate_all_batched(ground_stations, constellation=constellation, **kwargs)

    def run_monte_carlo(self, n_iterations: int, ground_stations: List[Dict], 
                        constellation: Optional[Constellation] = None,
                        **kwargs) -> List[List[StationResult]]:
        base_seed = kwargs.pop("seed", 42) or 42
        seeds = [base_seed + i for i in range(n_iterations)]
        
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(self.simulate_all_batched, ground_stations, seed=s, constellation=constellation, **kwargs)
                for s in seeds
            ]
            results = [f.result() for f in futures]
        return results

def run_simulation(*args, **kwargs):
    engine = SimulationEngine()
    return engine.simulate_all_batched(*args, **kwargs)
