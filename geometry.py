import math

def slant_range(altitude_km, elevation_deg):
    Re = 6371.0  # Earth radius (km)
    el = math.radians(elevation_deg)
    h = altitude_km
    return math.sqrt(
        (Re + h)**2 - (Re * math.cos(el))**2
    ) - Re * math.sin(el)
#Calculating slant range angles for different sattelites 

def effective_elevation(global_elev, latitude):
    # stations far from equator see lower effective elevation
    penalty = abs(latitude) / 90.0 * 20.0
    return max(5.0, global_elev - penalty)

