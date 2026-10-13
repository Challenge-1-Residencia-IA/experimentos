import asyncio
from src.config import NUM_WORKERS_IA, AUTOSCALE_ENABLED, AUTOSCALE_STEP, AUTOSCALE_MAX_FACTOR
from botfinal import worker_ia_loop  # type: ignore (circular import safe at runtime)

async def autoscale_workers(fila_entrada: asyncio.Queue, fila_saida: asyncio.Queue, worker_tasks: list[asyncio.Task]):
    """Dynamically start new IA workers based on queue backlog.
    The function never reduces the number of workers – it only adds up to
    ``NUM_WORKERS_IA * AUTOSCALE_MAX_FACTOR``.
    """
    if not AUTOSCALE_ENABLED:
        return
    base = NUM_WORKERS_IA
    max_workers = base * AUTOSCALE_MAX_FACTOR

    while True:
        # Desired workers = base + (queued tasks // AUTOSCALE_STEP)
        desired = min(max_workers, base + fila_entrada.qsize() // AUTOSCALE_STEP)
        current = len(worker_tasks)
        if desired > current:
            for i in range(current, desired):
                # Worker IDs are 1‑based for logging consistency
                task = asyncio.create_task(worker_ia_loop(i + 1, fila_entrada, fila_saida))
                worker_tasks.append(task)
                print(f"⚙️ [Autoscale] Spawned worker IA #{i + 1} (total {len(worker_tasks)})")
        await asyncio.sleep(2)

