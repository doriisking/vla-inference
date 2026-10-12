import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from vla.recording import EpisodeRecorder, EpisodeSample, snapshot_numpy


class RecorderTest(unittest.TestCase):
    def test_saved_npz_and_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "session"
            recorder = EpisodeRecorder(folder, max_queue=4, rgb_format="npy")
            sample = EpisodeSample(
                episode_id="run_0001", sample_id=5,
                observation_monotonic_ns=111, inference_monotonic_ns=222,
                progress=0.4, progress_state="running",
                latent=snapshot_numpy(np.ones(2048, dtype=np.float16)),
                action_chunk={"motion_token": np.ones((40, 64), dtype=np.float32)},
                rgb={"ego_view": np.zeros((16, 16, 3), dtype=np.uint8)},
                robot_state={"projected_gravity": np.ones(3, dtype=np.float32)},
            )
            self.assertTrue(recorder.submit(sample))
            self.assertTrue(recorder.close(timeout=5))
            self.assertEqual(recorder.stats()["written"], 1)
            with np.load(folder / "run_0001" / "0000005.npz") as data:
                self.assertEqual(data["latent"].shape, (2048,))
                self.assertEqual(data["action_motion_token"].shape, (40, 64))
            row = json.loads((folder / "samples.jsonl").read_text().splitlines()[0])
            self.assertEqual(row["progress"], 0.4)
            self.assertTrue((folder / row["rgb"]["ego_view"]).is_file())

    def test_reject_after_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = EpisodeRecorder(Path(tmp) / "closed", rgb_format="npy")
            self.assertTrue(recorder.close(timeout=5))
            sample = EpisodeSample("ep1", 1, 1, 2, None, None, np.zeros(1), {})
            self.assertFalse(recorder.submit(sample))

    def test_jpeg_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = EpisodeRecorder(Path(tmp) / "jpeg", max_queue=2)
            sample = EpisodeSample(
                episode_id="run_0001", sample_id=1,
                observation_monotonic_ns=1, inference_monotonic_ns=2,
                progress=0.3, progress_state=None,
                latent=np.ones(2048, dtype=np.float16),
                action_chunk={},
                rgb={"ego_view": np.zeros((16, 16, 3), dtype=np.uint8)},
            )
            self.assertTrue(recorder.submit(sample))
            self.assertTrue(recorder.close(timeout=5))
            self.assertTrue((Path(tmp) / "jpeg" / "run_0001" / "0000001_ego_view.jpg").exists())


if __name__ == "__main__":
    unittest.main()
