import cv2
from detector import PersonDetector
from tracker import PersonTracker
from reid import ReIDExtractor
from global_identity_manager import GlobalIdentityManager

reid = ReIDExtractor()
global_id_manager = GlobalIdentityManager(similarity_threshold=0.75)

# Maps local track_id → global_id
trackid_to_globalid = {}

detector = PersonDetector()
tracker = PersonTracker()

cap = cv2.VideoCapture("videos/cam1.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (960, 540))

    detections = detector.detect(frame)
    tracks = tracker.update(detections, frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())

        person_crop = frame[t:b, l:r]
        if person_crop.size == 0:
            continue

        embedding = reid.extract(person_crop)

        # 🔥 Assign Global ID
        if track_id not in trackid_to_globalid:
            global_id = global_id_manager.match(embedding)
            trackid_to_globalid[track_id] = global_id
            print(
                f"Track {track_id} → Global ID {global_id} | shape {embedding.shape}"
            )
        else:
            global_id = trackid_to_globalid[track_id]

        # Visualization
        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"GID {global_id}",
            (l, t - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imshow("Tracking + Global ReID", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
