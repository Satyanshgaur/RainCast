import math
import numpy as np
import sqlite3
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from sgp4.api import Satrec, jday

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

    def get_sat_rec(self, identifier):
        """Fetch Satrec by name or norad_id."""
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
            # If any step fails, we might want to handle it more gracefully, 
            # but for now return None if any error occurs
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
            azimuth_deg=np.degrees(az)
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
