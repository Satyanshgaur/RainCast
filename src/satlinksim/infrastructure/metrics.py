from prometheus_client import Counter, Histogram, make_asgi_app

# Counters
SIMULATIONS_RUN = Counter(
    "simulations_run_total", 
    "Total number of simulations executed",
    ["mode"] # e.g., sync, async, summary
)

HANDOFFS_TOTAL = Counter(
    "handoffs_total", 
    "Total number of handoffs occurred across all simulations"
)

STEPS_PROCESSED = Counter(
    "steps_processed_total", 
    "Total number of simulation steps processed"
)

# Histograms
SIMULATION_LATENCY = Histogram(
    "simulation_latency_seconds", 
    "Time taken to complete a simulation",
    ["mode"]
)

RAIN_GENERATION_TIME = Histogram(
    "rain_generation_seconds", 
    "Time taken for rain process generation"
)

# ASGI app to expose metrics
metrics_app = make_asgi_app()
