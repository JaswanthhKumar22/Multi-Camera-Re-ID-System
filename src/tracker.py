from deep_sort_realtime.deepsort_tracker import DeepSort

class PersonTracker:
    def __init__(self):
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            max_iou_distance=0.7,
            embedder="mobilenet",
            half=True
        )

    def update(self, detections, frame):
        """
        detections: [[x1, y1, x2, y2, conf], ...]
        """
        dets = []
        for x1, y1, x2, y2, conf in detections:
            w = x2 - x1
            h = y2 - y1
            dets.append(([x1, y1, w, h], conf, "person"))

        tracks = self.tracker.update_tracks(dets, frame=frame)
        return tracks
