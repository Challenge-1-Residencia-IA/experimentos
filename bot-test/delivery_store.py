"""Diário de deduplicação. Não armazena texto ou nome de contato."""
import hashlib
import os
import sqlite3
from pathlib import Path


def message_key(chat, message_id):
    return hashlib.sha256((chat + "\0" + message_id).encode("utf-8")).hexdigest()


class DeliveryStore:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.instance_lock = open(str(path) + ".lock", "a+b")
        self.instance_lock.seek(0, 2)
        if self.instance_lock.tell() == 0:
            self.instance_lock.write(b"0")
            self.instance_lock.flush()
        self.instance_lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.instance_lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.instance_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.instance_lock.close()
            raise RuntimeError("Outra instância do bot está usando este banco") from None
        self.db = sqlite3.connect(path, timeout=10)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("""CREATE TABLE IF NOT EXISTS deliveries (
            key TEXT PRIMARY KEY, state TEXT NOT NULL,
            updated TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
        self.db.execute("UPDATE deliveries SET state='interrupted', updated=CURRENT_TIMESTAMP WHERE state IN ('queued','analyzing','ready','sending')")
        self.db.commit()

    def known(self, key):
        return self.db.execute("SELECT 1 FROM deliveries WHERE key=?", (key,)).fetchone() is not None

    def claim(self, key, state="queued"):
        with self.db:
            cursor = self.db.execute("INSERT OR IGNORE INTO deliveries(key,state) VALUES (?,?)", (key, state))
        return cursor.rowcount == 1

    def state(self, key, state):
        with self.db:
            self.db.execute("UPDATE deliveries SET state=?, updated=CURRENT_TIMESTAMP WHERE key=?", (state, key))

    def reserve_send(self, key):
        with self.db:
            cursor = self.db.execute("UPDATE deliveries SET state='sending', updated=CURRENT_TIMESTAMP WHERE key=? AND state='ready'", (key,))
        return cursor.rowcount == 1

    def close(self):
        self.db.close()
        self.instance_lock.close()
