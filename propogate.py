import math
import numpy as np
import sqlite3
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from sgp4.api import Satrec, jday
from typing import List, Optional

# ── WGS84 Constants ──────────────────────────────────────────────────────────
WGS84_A = 6378.137             # Semi-major axis (km)
WGS84_F = 1/298.257223563      # Flattening
WGS84_E2 = (2*WGS84_F - WGS84_F**2) # Squared eccentricity
EARTH_OMEGA = 7.292115e-5      # Earth rotation rate (rad/s)

@dataclass
class SatelliteGeometry:
    elevation_deg: np.ndarray  # Shape: (n_steps,)
    slant_range_km: np.ndarray # Shape: (n_steps,)
    radial_velocity_ms: np.ndarray # Shape: (n_steps,)
    azimuth_deg: np.ndarray = None # Shape: (n_steps,)
    sat_name: str | list[str] = None # Name(s) of the satellite(s) providing this geometry

@dataclass
class Satellite:
    norad_id: int
    name: str
    tle_line1: Optional[str] = None
    tle_line2: Optional[str] = None

    @property
    def tle(self):
        return (self.tle_line1, self.tle_line2)

@dataclass
class Constellation:
    name: str
    satellites: List[Satellite]

    @classmethod
    def from_norad_ids(cls, name: str, ids: List[int]):
        """Create a constellation from a list of NORAD IDs (will be fetched from DB)."""
        return cls(name=name, satellites=[Satellite(norad_id=i, name=str(i)) for i in ids])

    @classmethod
    def from_tle_file(cls, name: str, file_path: str):
        """Parse a TLE file (3-line format or 2-line) into a constellation."""
        sats = []
        with open(file_path, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        
        # Determine if 2-line or 3-line format
        if len(lines) >= 3 and not lines[0].startswith("1 ") and not lines[0].startswith("2 "):
            # 3-line format (Name, L1, L2)
            for i in range(0, len(lines) - 2, 3):
                sat_name = lines[i]
                l1 = lines[i+1]
                l2 = lines[i+2]
                try:
                    norad_id = int(l1[2:7])
                    sats.append(Satellite(norad_id=norad_id, name=sat_name, tle_line1=l1, tle_line2=l2))
                except: continue
        else:
            # 2-line format
            for i in range(0, len(lines) - 1, 2):
                l1 = lines[i]
                l2 = lines[i+1]
                try:
                    norad_id = int(l1[2:7])
                    sats.append(Satellite(norad_id=norad_id, name=f"SAT-{norad_id}", tle_line1=l1, tle_line2=l2))
                except: continue
        return cls(name=name, satellites=sats)

def geodetic_to_ecef(lat_deg, lon_deg, alt_km):
    """Convert geodetic (lat, lon, alt) to ECEF (x, y, z) in km. Supports arrays."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat)**2)
    
    x = (n + alt_km) * np.cos(lat) * np.cos(lon)
    y = (n + alt_km) * np.cos(lat) * np.sin(lon)
    z = (n * (1 - WGS84_E2) + alt_km) * np.sin(lat)
    return np.stack([x, y, z], axis=-1)

def ecef_to_enu_matrix(lat_deg, lon_deg):
    """Create a rotation matrix from ECEF to ENU. Supports arrays."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)
    
    # Return shape (..., 3, 3)
    return np.array([
        [-sin_lon,           cos_lon,            np.zeros_like(lat)],
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [cos_lat * cos_lon,  cos_lat * sin_lon,  sin_lat]
    ]).transpose(2, 0, 1) if np.ndim(lat) > 0 else np.array([
        [-sin_lon,           cos_lon,            0],
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [cos_lat * cos_lon,  cos_lat * sin_lon,  sin_lat]
    ])

class Propagator:
    def __init__(self, db_path=None):
        if db_path is None:
            # Default to satellites.db in the same directory as this file
            db_path = os.path.join(os.path.dirname(__file__), "satellites.db")
        self.db_path = db_path
        self.cache = {}
        self._executor = ThreadPoolExecutor(max_workers=os.cpu_count())

    def get_sat_rec(self, identifier):
        """Fetch Satrec by name, norad_id, or Satellite object."""
        if isinstance(identifier, Satellite):
            if identifier.tle_line1 and identifier.tle_line2:
                sat = Satrec.twoline2rv(identifier.tle_line1, identifier.tle_line2)
                return identifier.name, sat
            identifier = identifier.norad_id

        if identifier in self.cache:
            return self.cache[identifier]
        
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        if isinstance(identifier, (int, np.integer)):
            cur.execute("SELECT name, tle_line1, tle_line2 FROM satellites WHERE norad_id=?", (int(identifier),))
        else:
            cur.execute("SELECT name, tle_line1, tle_line2 FROM satellites WHERE name LIKE ?", (f"%{identifier}%",))
        
        row = cur.fetchone()
        conn.close()
        
        if row:
            name, l1, l2 = row
            sat = Satrec.twoline2rv(l1, l2)
            self.cache[identifier] = (name, sat)
            return name, sat
        return None

    def get_geometry_batch(self, identifier, dts: list[datetime], gs_lat, gs_lon, gs_alt):
        """Propagate for a batch of times for a single ground station."""
        res = self.get_sat_rec(identifier)
        if not res: return None
        name, sat = res
        
        # SGP4 propagation (batched)
        jds = []
        frs = []
        for dt in dts:
            jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond/1e6)
            jds.append(jd)
            frs.append(fr)
        
        jds = np.array(jds)
        frs = np.array(frs)
        
        # sgp4 array API: sat.sgp4_array(jd, fr) -> error, pos, vel
        # pos/vel are (n, 3)
        error, sat_pos, sat_vel = sat.sgp4_array(jds, frs)
        if np.any(error != 0): 
            return None
        
        # Ground station ECEF
        gs_pos = geodetic_to_ecef(gs_lat, gs_lon, gs_alt) # (3,)
        
        # Relative vector (n, 3)
        rho_ecef = sat_pos - gs_pos
        slant_range = np.linalg.norm(rho_ecef, axis=1) # (n,)
        
        # Topocentric (ENU)
        m = ecef_to_enu_matrix(gs_lat, gs_lon) # (3, 3)
        rho_enu = np.einsum('ij,nj->ni', m, rho_ecef) # (n, 3)
        
        # Elevation & Azimuth
        el = np.arctan2(rho_enu[:, 2], np.sqrt(rho_enu[:, 0]**2 + rho_enu[:, 1]**2))
        az = np.arctan2(rho_enu[:, 0], rho_enu[:, 1])
        
        # Radial Velocity (Doppler)
        gs_vel = np.array([-EARTH_OMEGA * gs_pos[1], EARTH_OMEGA * gs_pos[0], 0])
        rel_vel = sat_vel - gs_vel # (n, 3)
        v_radial = np.einsum('ni,ni->n', rel_vel, rho_ecef) / slant_range # (n,) km/s
        
        return SatelliteGeometry(
            elevation_deg=np.degrees(el),
            slant_range_km=slant_range,
            radial_velocity_ms=v_radial * 1000.0,
            azimuth_deg=np.degrees(az),
            sat_name=name
        )

    def get_constellation_geometry(self, constellation: Constellation, dts: list[datetime], gs_lat, gs_lon, gs_alt):
        """Pick the best satellite (highest elevation) at each time step."""
        n_steps = len(dts)
        best_el = np.full(n_steps, -90.0)
        best_slant = np.zeros(n_steps)
        best_v_radial = np.zeros(n_steps)
        best_az = np.zeros(n_steps)
        best_name = [None] * n_steps

        for sat in constellation.satellites:
            geo = self.get_geometry_batch(sat, dts, gs_lat, gs_lon, gs_alt)
            if geo is None: continue
            
            mask = geo.elevation_deg > best_el
            best_el[mask] = geo.elevation_deg[mask]
            best_slant[mask] = geo.slant_range_km[mask]
            best_v_radial[mask] = geo.radial_velocity_ms[mask]
            best_az[mask] = geo.azimuth_deg[mask]
            for i in range(n_steps):
                if mask[i]:
                    best_name[i] = geo.sat_name

        return SatelliteGeometry(
            elevation_deg=best_el,
            slant_range_km=best_slant,
            radial_velocity_ms=best_v_radial,
            azimuth_deg=best_az,
            sat_name=best_name
        )

    async def get_geometry_batch_async(self, identifier, dts: list[datetime], gs_lat, gs_lon, gs_alt):
        """Async wrapper for get_geometry_batch, running in a thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, 
            self.get_geometry_batch, 
            identifier, dts, gs_lat, gs_lon, gs_alt
        )

    def get_geometry(self, identifier, dt: datetime, gs_lat, gs_lon, gs_alt):
        """Maintains backward compatibility for single-step calls."""
        res = self.get_geometry_batch(identifier, [dt], gs_lat, gs_lon, gs_alt)
        if not res: return None
        return SatelliteGeometry(
            elevation_deg=res.elevation_deg[0],
            slant_range_km=res.slant_range_km[0],
            radial_velocity_ms=res.radial_velocity_ms[0],
            azimuth_deg=res.azimuth_deg[0]
        )

