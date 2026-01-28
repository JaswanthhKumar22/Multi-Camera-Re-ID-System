import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity

class GlobalIdentityManager:
    def __init__(
        self,
        similarity_threshold=0.75,
        lost_timeout=5,
        exit_timeout=30
    ):
        self.similarity_threshold = similarity_threshold
        self.lost_timeout = lost_timeout
        self.exit_timeout = exit_timeout

        self.global_db = {}
        self.next_gid = 0

    def _new_gid(self):
        gid = self.next_gid
        self.next_gid += 1
        return gid

    def match(self, embedding):
        now = time.time()

        if len(self.global_db) == 0:
            gid = self._new_gid()
            self.global_db[gid] = {
                "embedding": embedding,
                "last_seen": now,
                "state": "ACTIVE"
            }
            return gid

        gids = list(self.global_db.keys())
        embs = np.array([self.global_db[g]["embedding"] for g in gids])

        sims = cosine_similarity(embedding.reshape(1, -1), embs)[0]
        best_idx = np.argmax(sims)
        best_score = sims[best_idx]

        if best_score >= self.similarity_threshold:
            gid = gids[best_idx]

            # Update embedding (EMA)
            self.global_db[gid]["embedding"] = (
                0.7 * self.global_db[gid]["embedding"] + 0.3 * embedding
            )
            self.global_db[gid]["last_seen"] = now

            # Handle re-appearance
            if self.global_db[gid]["state"] != "ACTIVE":
                print(f"🔁 Global ID {gid} RE-APPEARED")
                self.global_db[gid]["state"] = "ACTIVE"

            return gid

        gid = self._new_gid()
        self.global_db[gid] = {
            "embedding": embedding,
            "last_seen": now,
            "state": "ACTIVE"
        }
        print(f"➕ New Global ID {gid}")
        return gid

    def update_states(self):
        now = time.time()

        for gid, data in self.global_db.items():
            elapsed = now - data["last_seen"]

            if elapsed > self.exit_timeout:
                if data["state"] != "EXITED":
                    data["state"] = "EXITED"
                    print(f"🚪 Global ID {gid} EXITED")
            elif elapsed > self.lost_timeout:
                if data["state"] != "LOST":
                    data["state"] = "LOST"
                    print(f"❓ Global ID {gid} LOST")
