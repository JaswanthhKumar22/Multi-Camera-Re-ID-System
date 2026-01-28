import threading
from detector import PersonDetector
from tracker import PersonTracker
from reid import ReIDExtractor
from global_identity_manager import GlobalIdentityManager
from camera_worker import run_camera
from suspect_query import SuspectQuery
from database import ReIDDatabase
from event_queue import EventQueue


# Shared components
detector = PersonDetector()
reid = ReIDExtractor()
global_id_manager = GlobalIdentityManager(similarity_threshold=0.75)

# Per-camera trackers
tracker_cam1 = PersonTracker()
tracker_cam2 = PersonTracker()
tracker_cam3 = PersonTracker()
tracker_cam4 = PersonTracker()

suspect = SuspectQuery(
    image_path="../suspect.jpeg",
    reid_model=reid
)
db = ReIDDatabase()
event_queue = EventQueue(db)
camera_configs = [
    (1, "../videos/cam1.mp4"),
    (2, "../videos/cam2.mp4"),
    (3, "../videos/cam3.mp4"),
    (4, "../videos/cam4.mp4"),
]

t1 = threading.Thread(
    target=run_camera,
    args=(1, "../videos/cam1.mp4", detector, tracker_cam1, reid, global_id_manager, suspect,0.55,event_queue)
)

t2 = threading.Thread(
    target=run_camera,
    args=(2, "../videos/cam2.mp4", detector, tracker_cam2, reid, global_id_manager, suspect,0.55,event_queue)
)

t3 = threading.Thread(
    target=run_camera,
    args=(3, "../videos/cam3.mp4", detector, tracker_cam3, reid, global_id_manager, suspect,0.55,event_queue)
)

t4 = threading.Thread(
    target=run_camera,
    args=(4, "../videos/cam4.mp4", detector, tracker_cam4, reid, global_id_manager, suspect,0.55,event_queue)
)


from suspect_query import SuspectQuery

suspect = SuspectQuery(
    image_path="../suspect.jpeg",
    reid_model=reid
)

if __name__ == "__main__":
    t1.start()
    t2.start()
    t3.start()
    t4.start()

    t1.join()
    t2.join()
    t3.join()
    t4.join()
    event_queue.stop()

