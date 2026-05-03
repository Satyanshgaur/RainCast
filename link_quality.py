import csv
import random
from datetime import datetime

def normalize_snr(snr, min_snr=-10, max_snr=20):
    return max(0.0, min(1.0, (snr - min_snr) / (max_snr - min_snr)))

def link_quality_score(snr, packet_loss, load):
    snr_n = normalize_snr(snr)
    return (
        0.5 * snr_n
        + 0.3 * (1 - packet_loss)
        + 0.2 * (1 - load)
    )

def synthetic_packet_loss(snr):
    if snr > 10:
        return random.uniform(0.0, 0.01)
    elif snr > 0:
        return random.uniform(0.01, 0.05)
    else:
        return random.uniform(0.05, 0.20)

def synthetic_load():
    return random.uniform(0.1, 0.9)

with open("link_training_data.csv", "a", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "timestamp",
        "snr_db",
        "packet_loss",
        "load_factor",
        "link_quality"
    ])

    for _ in range(200):
        snr = random.uniform(-5, 15)
        packet_loss = synthetic_packet_loss(snr)
        load = synthetic_load()

        score = link_quality_score(snr, packet_loss, load)

        writer.writerow([
            datetime.utcnow().isoformat(),
            snr,
            packet_loss,
            load,
            score
        ])

