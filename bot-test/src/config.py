import os

# Number of IA workers (default 5). Can be overridden via .env or environment.
NUM_WORKERS_IA = 1

# Autoscaling control – set AUTOSCALE_ENABLED=false to keep a fixed pool.
AUTOSCALE_ENABLED = False
# How many queued tasks trigger adding one extra worker.
AUTOSCALE_STEP = int(os.getenv("AUTOSCALE_STEP", "5"))
# Maximum multiplier of the base worker count.
AUTOSCALE_MAX_FACTOR = int(os.getenv("AUTOSCALE_MAX_FACTOR", "2"))
