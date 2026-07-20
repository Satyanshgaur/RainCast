import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, "src"))

from satlinksim.satellite_link_sim import simulate_all_batched
from satlinksim.domain.models import Constellation

c = Constellation.from_norad_ids("OneWeb-Mock", [45131, 45132, 44057, 44058, 44059, 44060])
res = simulate_all_batched(
    [{"name": "Test", "latitude": 28.6, "longitude": 77.2, "altitude_km": 0.2, "eirp_dbw": 60, "g_rx_dbi": 40, "system_temp_k": 290, "wv_g_m3": 7.5, "antenna_diam_m": 1.2, "humidity_pct": 50, "itu_rain": {"R001": 50.0, "R01": 25.0, "R1": 10.0, "P_rain": 0.01}}],
    n_steps=180,
    constellation=c
)
print("Handoffs:", len(res[0].handoff_events))
