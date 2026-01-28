import cv2
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_color_for_id(gid):
    random.seed(gid)
    return (
        random.randint(50, 255),
        random.randint(50, 255),
        random.randint(50, 255)
    )

def run_camera(
    cam_id,
    video_path,
    detector,
    tracker,
    reid,
    global_id_manager,
    suspect_query=None,
    suspect_threshold=0.55,
    event_queue=None
):
    cap = cv2.VideoCapture(video_path)

    trackid_to_globalid = {}

    # 🔥 CONFIRMATION STATE (PER CAMERA)
    confirmed_suspects = {}
    CONFIRM_FRAMES = 10

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

            # ---------- GLOBAL ID ----------
            if track_id not in trackid_to_globalid:
                global_id = global_id_manager.match(embedding)
                trackid_to_globalid[track_id] = global_id
                print(f"[Cam {cam_id}] Track {track_id} → Global {global_id}")
            else:
                global_id = trackid_to_globalid[track_id]

            # ---------- SUSPECT CONFIRMATION ----------
            score = 0.0
            is_suspect = False

            if suspect_query is not None:
                score = cosine_similarity(
                    embedding.reshape(1, -1),
                    suspect_query.embedding.reshape(1, -1)
                )[0][0]

                if score >= suspect_threshold:
                    confirmed_suspects[global_id] = (
                        confirmed_suspects.get(global_id, 0) + 1
                    )
                else:
                    confirmed_suspects[global_id] = max(
                        0, confirmed_suspects.get(global_id, 0) - 1
                    )

                if confirmed_suspects.get(global_id, 0) >= CONFIRM_FRAMES:
                    is_suspect = True

            # ---------- VISUALIZATION ----------
            if is_suspect:
                color = (0, 0, 255)
                label = f"SUSPECT ({score:.2f})"
            else:
                color = get_color_for_id(global_id)
                label = f"GID {global_id}"

            cv2.rectangle(frame, (l, t), (r, b), color, 2)
            cv2.putText(
                frame,
                label,
                (l, t - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            # ---------- EVENT LOGGING ----------
            if event_queue is not None:
                event_queue.push(
                    global_id=global_id,
                    camera_id=cam_id,
                    is_suspect=is_suspect,
                    similarity=score
                )

        global_id_manager.update_states()

        cv2.imshow(f"Camera {cam_id}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(f"Camera {cam_id}")
