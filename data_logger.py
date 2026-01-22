import pandas as pd
import os

CSV_PATH = "data/telemetry_log.csv"

COLUMNS = [
    "device_model",
    "timestamp",
    "effort_metric",
    "efficiency_metric",
    "usage_metric",
    "maintenance_event"
]

def log_to_csv(row):
    df = pd.DataFrame([row])

    file_exists = os.path.exists(CSV_PATH)
    write_header = not file_exists or os.path.getsize(CSV_PATH) == 0

    df.to_csv(
        CSV_PATH,
        mode="a",
        index=False,
        header=write_header,
        columns=COLUMNS
    )
