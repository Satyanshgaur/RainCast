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
    elevation_deg: float
    slant_range_km: float
    radial_velocity_ms: float
    azimuth_deg: float = 0.0

def geodetic_to_ecef(lat_deg, lon_deg, alt_km):
    """Convert geodetic (lat, lon, alt) to ECEF (x, y, z) in km."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    n = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(lat)**2)
    
    x = (n + alt_km) * math.cos(lat) * math.cos(lon)
    y = (n + alt_km) * math.cos(lat) * math.sin(lon)
    z = (n * (1 - WGS84_E2) + alt_km) * math.sin(lat)
    return np.array([x, y, z])

def ecef_to_enu_matrix(lat_deg, lon_deg):
    """Create a rotation matrix from ECEF to ENU at a given location."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    
    return np.array([
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
        if isinstance(identifier, int):
            cur.execute("SELECT name, tle_line1, tle_line2 FROM satellites WHERE norad_id=?", (identifier,))
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

    def get_geometry(self, identifier, dt: datetime, gs_lat, gs_lon, gs_alt):
        """Propagate and compute geometry relative to a ground station."""
        res = self.get_sat_rec(identifier)
        if not res: return None
        name, sat = res
        
        # SGP4 propagation
        jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond/1e6)
        error, sat_pos, sat_vel = sat.sgp4(jd, fr)
        if error != 0: return None
        
        sat_pos = np.array(sat_pos)
        sat_vel = np.array(sat_vel)
        
        # Ground station ECEF
        gs_pos = geodetic_to_ecef(gs_lat, gs_lon, gs_alt)
        
        # Relative vector
        rho_ecef = sat_pos - gs_pos
        slant_range = np.linalg.norm(rho_ecef)
        
        # Topocentric (ENU)
        m = ecef_to_enu_matrix(gs_lat, gs_lon)
        rho_enu = m @ rho_ecef
        
        # Elevation & Azimuth
        el = math.atan2(rho_enu[2], math.sqrt(rho_enu[0]**2 + rho_enu[1]**2))
        az = math.atan2(rho_enu[0], rho_enu[1])
        
        # Radial Velocity (Doppler)
        # Account for Earth rotation in ground station velocity
        gs_vel = np.array([-EARTH_OMEGA * gs_pos[1], EARTH_OMEGA * gs_pos[0], 0])
        rel_vel = sat_vel - gs_vel
        v_radial = np.dot(rel_vel, rho_ecef) / slant_range # km/s
        
        return SatelliteGeometry(
            elevation_deg=math.degrees(el),
            slant_range_km=slant_range,
            radial_velocity_ms=v_radial * 1000.0,
            azimuth_deg=math.degrees(az)
        )
