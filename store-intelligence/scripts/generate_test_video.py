#!/usr/bin/env python3
"""
Store Intelligence System — Synthetic Test Video Generator.

Creates a 30-second 640×480 synthetic video at 15 FPS with colored
rectangles simulating people moving through the store.  Zone boundaries
are drawn as semi-transparent overlays matching config/zones.json.

Usage:
    python scripts/generate_test_video.py
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# ── Constants ───────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "test_data"
OUTPUT_PATH = OUTPUT_DIR / "sample_store.mp4"
ZONES_PATH = Path(__file__).resolve().parent.parent / "config" / "zones.json"

WIDTH, HEIGHT = 640, 480
FPS = 15
DURATION_SECONDS = 30
TOTAL_FRAMES = FPS * DURATION_SECONDS

# Person rectangle dimensions
PERSON_W, PERSON_H = 30, 60

# Colour palette for people (BGR)
PERSON_COLORS: List[Tuple[int, int, int]] = [
    (0, 0, 255),      # Red
    (0, 200, 0),      # Green
    (255, 100, 0),    # Blue
    (0, 200, 255),    # Yellow
    (200, 0, 200),    # Magenta
]


# ── Zone Loader ─────────────────────────────────────────────
def load_zones(path: Path) -> list[dict]:
    """Load zone definitions from zones.json."""
    with open(path, "r") as f:
        data = json.load(f)
    return data["zones"]


# ── Simulated Person ────────────────────────────────────────
@dataclass
class SimPerson:
    """A simulated person represented as a moving rectangle."""

    person_id: int
    color: Tuple[int, int, int]
    # Current position (top-left corner)
    x: float
    y: float
    # Velocity
    vx: float
    vy: float
    # Lifecycle
    enter_frame: int
    exit_frame: int
    # Waypoints: list of (x, y) targets the person walks toward
    waypoints: List[Tuple[float, float]] = field(default_factory=list)
    _wp_idx: int = 0
    speed: float = 1.5

    def update(self, frame_idx: int) -> bool:
        """Move the person toward its next waypoint. Returns False after exit_frame."""
        if frame_idx < self.enter_frame or frame_idx > self.exit_frame:
            return False

        # Navigate toward current waypoint
        if self._wp_idx < len(self.waypoints):
            tx, ty = self.waypoints[self._wp_idx]
            dx = tx - self.x
            dy = ty - self.y
            dist = math.hypot(dx, dy)
            if dist < self.speed * 2:
                self._wp_idx += 1
            else:
                self.vx = self.speed * dx / dist
                self.vy = self.speed * dy / dist

        self.x += self.vx
        self.y += self.vy

        # Clamp to frame
        self.x = max(0, min(WIDTH - PERSON_W, self.x))
        self.y = max(0, min(HEIGHT - PERSON_H, self.y))
        return True

    def draw(self, frame: np.ndarray) -> None:
        """Draw this person on the frame."""
        x1, y1 = int(self.x), int(self.y)
        x2, y2 = x1 + PERSON_W, y1 + PERSON_H
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.color, -1)
        # Head circle
        head_cx = x1 + PERSON_W // 2
        head_cy = y1 - 8
        cv2.circle(frame, (head_cx, head_cy), 8, self.color, -1)
        # ID label
        cv2.putText(
            frame, f"P{self.person_id}",
            (x1, y1 - 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1,
        )


# ── Create People with Realistic Paths ─────────────────────
def create_people() -> List[SimPerson]:
    """Create 5 simulated people with different movement paths."""
    people: List[SimPerson] = []

    # Zone centres (approximate, matching zones.json)
    entrance_center = (100, 520)
    lipstick_center = (320, 175)
    skincare_center = (540, 175)
    trial_center = (320, 420)
    checkout_center = (540, 520)

    # Person 0: Full shopper journey — entrance → lipstick → trial → checkout → exit
    people.append(SimPerson(
        person_id=0,
        color=PERSON_COLORS[0],
        x=100, y=470,
        vx=0, vy=0,
        enter_frame=0,
        exit_frame=TOTAL_FRAMES - 1,
        waypoints=[
            entrance_center,
            lipstick_center,
            trial_center,
            checkout_center,
            (620, 470),  # exit right
        ],
        speed=1.8,
    ))

    # Person 1: Browses skincare only, enters late
    people.append(SimPerson(
        person_id=1,
        color=PERSON_COLORS[1],
        x=100, y=470,
        vx=0, vy=0,
        enter_frame=FPS * 3,
        exit_frame=TOTAL_FRAMES - FPS * 2,
        waypoints=[
            entrance_center,
            skincare_center,
            (540, 250),   # linger
            skincare_center,
            entrance_center,
        ],
        speed=1.4,
    ))

    # Person 2: Loiterer — stays near entrance for a long time
    people.append(SimPerson(
        person_id=2,
        color=PERSON_COLORS[2],
        x=50, y=440,
        vx=0, vy=0,
        enter_frame=FPS * 1,
        exit_frame=TOTAL_FRAMES - 1,
        waypoints=[
            (80, 500),
            (150, 520),
            (80, 480),
            (120, 540),
            (80, 500),
            (150, 520),
        ],
        speed=0.6,   # slow — loitering behaviour
    ))

    # Person 3: Quick visitor — enters mid-video, goes to checkout fast
    people.append(SimPerson(
        person_id=3,
        color=PERSON_COLORS[3],
        x=100, y=470,
        vx=0, vy=0,
        enter_frame=FPS * 10,
        exit_frame=FPS * 25,
        waypoints=[
            entrance_center,
            checkout_center,
            (620, 470),
        ],
        speed=2.5,
    ))

    # Person 4: Explorer — visits every zone
    people.append(SimPerson(
        person_id=4,
        color=PERSON_COLORS[4],
        x=100, y=470,
        vx=0, vy=0,
        enter_frame=FPS * 5,
        exit_frame=TOTAL_FRAMES - 1,
        waypoints=[
            entrance_center,
            lipstick_center,
            skincare_center,
            trial_center,
            checkout_center,
            entrance_center,
        ],
        speed=2.0,
    ))

    return people


# ── Drawing Helpers ─────────────────────────────────────────
def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    """Convert '#RRGGBB' to (B, G, R)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)


def draw_zones(frame: np.ndarray, zones: list[dict]) -> None:
    """Draw semi-transparent zone overlays with labels."""
    overlay = frame.copy()
    for zone in zones:
        polygon = zone["polygon"]
        # Scale polygons from 640×640 config to 640×480 viewport
        # The zones.json uses a 640×640 coordinate space; scale y axis
        scaled = [(int(p[0]), int(p[1] * HEIGHT / 640)) for p in polygon]
        pts = np.array(scaled, dtype=np.int32)

        color = hex_to_bgr(zone["color"])
        cv2.fillPoly(overlay, [pts], color)

        # Label
        cx = int(np.mean([p[0] for p in scaled]))
        cy = int(np.mean([p[1] for p in scaled]))
        label = zone["name"]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(
            frame, label,
            (cx - tw // 2, cy + th // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
        )
        # Zone border
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

    # Blend overlay at 25 % opacity
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)


def draw_hud(frame: np.ndarray, frame_idx: int, active_count: int) -> None:
    """Draw heads-up-display info."""
    seconds = frame_idx / FPS
    text = f"Frame {frame_idx}/{TOTAL_FRAMES}  |  Time {seconds:.1f}s  |  People: {active_count}"
    cv2.putText(
        frame, text,
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA,
    )


# ── Main ────────────────────────────────────────────────────
def main() -> None:
    """Generate the synthetic test video."""
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load zones
    if not ZONES_PATH.exists():
        print(f"ERROR: Zone config not found at {ZONES_PATH}", file=sys.stderr)
        sys.exit(1)
    zones = load_zones(ZONES_PATH)

    # Create people
    people = create_people()

    # Set up VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        print("ERROR: Could not open VideoWriter", file=sys.stderr)
        sys.exit(1)

    print(f"Generating {DURATION_SECONDS}s video at {FPS} FPS ({TOTAL_FRAMES} frames)…")
    print(f"  Resolution : {WIDTH}×{HEIGHT}")
    print(f"  Output     : {OUTPUT_PATH}")
    print(f"  People     : {len(people)}")

    for frame_idx in range(TOTAL_FRAMES):
        # Background — dark store floor
        frame = np.full((HEIGHT, WIDTH, 3), (40, 40, 40), dtype=np.uint8)

        # Draw zone overlays
        draw_zones(frame, zones)

        # Update & draw people
        active_count = 0
        for person in people:
            if person.update(frame_idx):
                person.draw(frame)
                active_count += 1

        # HUD
        draw_hud(frame, frame_idx, active_count)

        writer.write(frame)

        # Progress
        if frame_idx % (FPS * 5) == 0:
            pct = frame_idx / TOTAL_FRAMES * 100
            print(f"  Progress: {pct:5.1f}%")

    writer.release()
    file_size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"✓ Video saved to {OUTPUT_PATH} ({file_size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
