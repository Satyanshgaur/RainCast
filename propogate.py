from sgp4.api import Satrec
from sgp4.api import jday
import sqlite3
from datetime import datetime

conn = sqlite3.connect("satellites.db")
cur = conn.cursor()

cur.execute("SELECT name, tle_line1, tle_line2 FROM satellites")
satellites = cur.fetchall()

for name, l1, l2 in satellites:
    sat = Satrec.twoline2rv(l1, l2)

    now = datetime.utcnow()
    jd, fr = jday(
        now.year, now.month, now.day,
        now.hour, now.minute, now.second
    )

    error, position, velocity = sat.sgp4(jd, fr)

    if error == 0:
        print(f"{name}")
        print("Position (km):", position)
        print("Velocity (km/s):", velocity)
        print()

