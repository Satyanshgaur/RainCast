import math
import random

C = 3e8
CARRIER_FREQ = 14e9
SAT_DISTANCE_KM = 40000
BANDWIDTH_HZ = 36e6
SYSTEM_TEMP_K = 500

RAIN_K = 0.02
RAIN_ALPHA = 1.1
RAIN_PATH_KM = 3.0

P_CLEAR_TO_RAIN = 0.05
P_RAIN_TO_CLEAR = 0.30

def fspl(freq_hz, distance_km):
    return (
        92.45
        + 20 * math.log10(freq_hz / 1e9)
        + 20 * math.log10(distance_km)
    )

def noise_power_dbw(T, B):
    return -228.6 + 10 * math.log10(T) + 10 * math.log10(B)

def rain_attenuation(rate):
    return RAIN_K * (rate ** RAIN_ALPHA) * RAIN_PATH_KM

def sample_rain_rate():
    return min(random.lognormvariate(2.0, 0.6), 50)

ground_station = {
    "name": "Delhi",
    "eirp_dbw": 52,
    "g_rx_dbi": 45,
    "atm_loss_db": 1.2,
    "v_radial": -30
}

path_loss = fspl(CARRIER_FREQ, SAT_DISTANCE_KM)
noise_dbw = noise_power_dbw(SYSTEM_TEMP_K, BANDWIDTH_HZ)

rain_state = 0

print(f"Simulating link for {ground_station['name']}\n")

for t in range(10):
    if rain_state == 0 and random.random() < P_CLEAR_TO_RAIN:
        rain_state = 1
    elif rain_state == 1 and random.random() < P_RAIN_TO_CLEAR:
        rain_state = 0

    if rain_state:
        rain_rate = sample_rain_rate()
        rain_fade = rain_attenuation(rain_rate)
    else:
        rain_rate = 0.0
        rain_fade = 0.0

    snr = (
        ground_station["eirp_dbw"]
        - path_loss
        - ground_station["atm_loss_db"]
        - rain_fade
        + ground_station["g_rx_dbi"]
        - noise_dbw
    )

    doppler = (ground_station["v_radial"] / C) * CARRIER_FREQ

    print(
        f"t={t:02d} | "
        f"{'RAIN' if rain_state else 'CLEAR'} | "
        f"Rain={rain_rate:4.1f} | "
        f"SNR={snr:6.2f} dB | "
        f"Doppler={doppler:6.1f} Hz"
    )

