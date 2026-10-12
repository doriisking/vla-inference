"""Best-effort CPU snapshot recorder: bounded queue, background disk writer.

The inference thread prepares CPU arrays before submit(); a CUDA -> CPU
snapshot may synchronize and delay the next prediction. JPEG, NPZ and JSONL
file writes are done only in the background writer thread.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from queue import Empty, Full, Queue
import threading
import time
from typing import Any, Mapping

import numpy as np

LOG = logging.getLogger(__name__)


def snapshot_numpy(value: Any) -> np.ndarray:
    """Return an owned CPU copy; a CUDA tensor copy may block inference."""
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    return np.array(value, copy=True)


@dataclass(frozen=True)
class EpisodeSample:
    episode_id: str
    sample_id: int
    observation_monotonic_ns: int
    inference_monotonic_ns: int
    progress: float | None
    progress_state: str | None
    latent: np.ndarray
    action_chunk: Mapping[str, np.ndarray]
    rgb: Mapping[str, np.ndarray] = field(default_factory=dict)
    robot_state: Mapping[str, np.ndarray] | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)


class EpisodeRecorder:
    """Nonblocking enqueue of stable CPU samples; full queues drop samples.

    submit() is not a zero-cost call: snapshot the tensors in the caller
    before invoking it. It never alters robot control or action generation.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        max_queue: int = 16,
        rgb_format: str = "jpeg",
        jpeg_quality: int = 80,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if max_queue < 1:
            raise ValueError("max_queue must be >= 1")
        if rgb_format not in {"jpeg", "npy"}:
            raise ValueError("rgb_format must be jpeg or npy")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.rgb_format = rgb_format
        self.jpeg_quality = jpeg_quality
        self._queue: Queue[EpisodeSample] = Queue(maxsize=max_queue)
        self._closing = threading.Event()
        self._lock = threading.Lock()
        self._accepted = 0
        self._dropped = 0
        self._written = 0
        self._errors = 0
        self._error_messages: list[str] = []
        self._write_json(self.output_dir / "metadata.json", dict(metadata or {}))
        self._thread = threading.Thread(
            target=self._write_loop, name="EpisodeRecorderWriter", daemon=False
        )
        self._thread.start()

    @staticmethod
    def _write_json(path: Path, value: Mapping[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, default=str)

    def submit(self, sample: EpisodeSample) -> bool:
        """Nonblocking; False if recording is closed or backlog is full."""
        with self._lock:
            if self._closing.is_set():
                return False
            try:
                self._queue.put_nowait(sample)
            except Full:
                self._dropped += 1
                return False
            self._accepted += 1
            return True

    def _write_loop(self) -> None:
        with (self.output_dir / "samples.jsonl").open("a", encoding="utf-8") as index:
            while not self._closing.is_set() or not self._queue.empty():
                try:
                    sample = self._queue.get(timeout=0.05)
                except Empty:
                    continue
                try:
                    entry = self._write_sample(sample)
                    index.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
                    index.flush()
                    with self._lock:
                        self._written += 1
                except Exception as exc:
                    LOG.exception("Could not write sample %s/%s", sample.episode_id, sample.sample_id)
                    with self._lock:
                        self._errors += 1
                        if len(self._error_messages) < 10:
                            self._error_messages.append(repr(exc))
                finally:
                    self._queue.task_done()

    def _write_sample(self, sample: EpisodeSample) -> dict[str, Any]:
        episode_id = str(sample.episode_id)
        allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        if not episode_id or any(char not in allowed for char in episode_id):
            raise ValueError("invalid episode_id")
        if sample.sample_id < 0:
            raise ValueError("sample_id must be nonnegative")
        episode_dir = self.output_dir / episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{sample.sample_id:07d}"
        arrays_path = episode_dir / f"{stem}.npz"
        arrays: dict[str, np.ndarray] = {"latent": np.asarray(sample.latent)}
        for name, value in sample.action_chunk.items():
            if not name.isidentifier():
                raise ValueError("invalid action name")
            arrays[f"action_{name}"] = np.asarray(value)
        if sample.robot_state is not None:
            for name, value in sample.robot_state.items():
                if not name.isidentifier():
                    raise ValueError("invalid state name")
                arrays[f"state_{name}"] = np.asarray(value)
        np.savez(arrays_path, **arrays)

        rgb_paths: dict[str, str] = {}
        for name, value in sample.rgb.items():
            if not name or any(char not in allowed for char in name):
                raise ValueError("invalid RGB view name")
            frame = np.asarray(value)
            if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
                raise ValueError("RGB frame must be HxWx3 uint8")
            if self.rgb_format == "jpeg":
                from PIL import Image
                frame_path = episode_dir / f"{stem}_{name}.jpg"
                Image.fromarray(frame, "RGB").save(frame_path, quality=self.jpeg_quality)
            else:
                frame_path = episode_dir / f"{stem}_{name}.npy"
                np.save(frame_path, frame)
            rgb_paths[name] = str(frame_path.relative_to(self.output_dir))
        return {
            "episode_id": episode_id,
            "sample_id": sample.sample_id,
            "observation_monotonic_ns": sample.observation_monotonic_ns,
            "inference_monotonic_ns": sample.inference_monotonic_ns,
            "progress": sample.progress,
            "progress_state": sample.progress_state,
            "npz": str(arrays_path.relative_to(self.output_dir)),
            "rgb": rgb_paths,
            "extras": dict(sample.extras),
        }

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "accepted": self._accepted,
                "dropped": self._dropped,
                "written": self._written,
                "write_errors": self._errors,
                "queue_depth": self._queue.qsize(),
            }

    def close(self, timeout: float | None = 15.0) -> bool:
        """Flush best effort. False means writer still active after timeout."""
        self._closing.set()
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            return False
        stats = self.stats()
        stats["errors"] = list(self._error_messages)
        stats["closed_wall_time_ns"] = time.time_ns()
        self._write_json(self.output_dir / "recorder_stats.json", stats)
        return True
