import queue
import threading

class EventQueue:
    def __init__(self, db):
        self.queue = queue.Queue()
        self.db = db
        self.running = True

        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

    def _worker_loop(self):
    # 🔥 Create DB connection INSIDE worker thread
        self.db.connect()

        while self.running:
            try:
                event = self.queue.get(timeout=1)
                self.db.log_event(**event)
            except queue.Empty:
                continue


    def push(self, global_id, camera_id, is_suspect, similarity):
        self.queue.put({
            "global_id": global_id,
            "camera_id": camera_id,
            "is_suspect": is_suspect,
            "similarity": similarity
        })

    def stop(self):
        self.running = False
