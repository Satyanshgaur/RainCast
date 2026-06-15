import os
import sys
from satlinksim.satellite_link_sim import SimulationEngine, GROUND_STATIONS

# We'll patch HandoffManager to crash at step 15,000 for Station 0
from satlinksim.domain.handoff.manager import HandoffManager
original_select = HandoffManager.select

def crashing_select(self, step_idx, *args, **kwargs):
    if step_idx == 15000:
        print(f"CRASHING INTENTIONALLY at step {step_idx}")
        sys.exit(1)
    return original_select(self, step_idx, *args, **kwargs)

HandoffManager.select = crashing_select

engine = SimulationEngine()
try:
    print("Running simulation (will crash)...")
    engine.simulate_all_batched(GROUND_STATIONS[:2], n_steps=30000, checkpoint_interval=10000)
except SystemExit:
    print("Simulation crashed as expected.")

# Restore the original method
HandoffManager.select = original_select

print("Resuming simulation from checkpoint...")
engine2 = SimulationEngine()
results = engine2.resume()
print(f"Resumed successfully. Processed {len(results)} stations.")
for r in results:
    print(f" - {r.name}: {len(r.snr_series)} steps recorded.")
