"""
Store Intelligence System — Staff Classifier.

A heuristic-based classifier that determines if a tracked person is staff
based on duration, zone coverage, and consistency.
"""
from typing import Dict, List, Optional
from config.settings import get_settings

class StaffClassifier:
    def __init__(self):
        self._settings = get_settings()
        self._staff_dwell_threshold = getattr(self._settings, "STAFF_DWELL_THRESHOLD_SECONDS", 1800)
        self._staff_zone_threshold = getattr(self._settings, "STAFF_ZONE_VISIT_THRESHOLD", 5)
        self._classified: Dict[int, bool] = {}

    def classify(self, track_id: int, zones_visited: List[str], session_duration_s: float) -> bool:
        """Classify whether a track is staff based on heuristics."""
        if track_id in self._classified and self._classified[track_id]:
            return True
            
        unique_zones = len(set(z for z in zones_visited if z))
        is_staff = False
        
        if session_duration_s >= self._staff_dwell_threshold:
            is_staff = True
        elif unique_zones >= self._staff_zone_threshold:
            is_staff = True
            
        self._classified[track_id] = is_staff
        return is_staff
        

    def update(self, track_id: int, zone_id: str, timestamp: float) -> None:
        pass

    def get_staff_flags(self) -> Dict[int, bool]:
        return self._classified

    def remove_track(self, track_id: int) -> None:
        if track_id in self._classified:
            del self._classified[track_id]
