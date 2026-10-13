import os

# Number of IA workers (default 5). Can be overridden via .env or environment.
NUM_WORKERS_IA = int(os.getenv("NUM_WORKERS_IA", "5"))

# Autoscaling control – set AUTOSCALE_ENABLED=false to keep a fixed pool.
AUTOSCALE_ENABLED = os.getenv("AUTOSCALE_ENABLED", "true").lower() == "true"
# How many queued tasks trigger adding one extra worker.
AUTOSCALE_STEP = int(os.getenv("AUTOSCALE_STEP", "5"))
# Maximum multiplier of the base worker count.
AUTOSCALE_MAX_FACTOR = int(os.getenv("AUTOSCALE_MAX_FACTOR", "2"))
