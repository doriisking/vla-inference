# Read-only Episode Recorder (experimental)

This branch records GR00T N1.7 inference inputs and outputs for offline research. It does not intentionally alter the policy, FSM, or 50 Hz streamer. Real-robot end-to-end testing has NOT been performed.

## Run

Use your normal run_vla.py arguments, with a compatible trained progress probe, adding:

    --record --record-dir shared/episodes --record-every-n 5 --record-max-queue 16

Recording is off by default. Each fifth prediction is snapshotted into a bounded queue and written on a background writer thread. The progress probe is required. CPU snapshots (including GPU-to-CPU transfers) happen in the inference thread and CAN delay subsequent predictions. JPEG/NPZ/JSONL writing is performed in the writer thread. A full queue drops samples.

Before running on hardware, compare inference-loop p50/p95, output stream timing and dropped samples with recording disabled/enabled. run_vla.py can physically move the robot; apply normal robot safety precautions.

## Saved files

    shared/episodes/session_YYYYMMDD_HHMMSS_NNNNNNNNN/
      metadata.json
      samples.jsonl
      recorder_stats.json
      run_0001/
        0000005.npz
        0000005_ego_view.jpg

Each NPZ contains the 2048-D Action Head hidden state, predicted ActionChunk, and observed joint-state input groups. Predicted ActionChunk is NOT executed-action feedback. RGB comes from the exact Observation sent to the model. samples.jsonl records timestamps, camera stamps, progress, prompts, and relative paths.

The existing view_progress.py S/F/R marks are separate; this initial version does not automatically split the Recorder's run_0001 into viewer episodes. Match outcome marks to samples by timestamps and validate results with video before treating them as ground truth.

The shared/ directory is gitignored. Do not publish recorded research imagery, checkpoints, or other sensitive data to your public fork.

## CPU-only tests

    PYTHONPATH=src python -m unittest discover -s tests -p test_episode_recorder_queue.py -v

Tests cover NPZ/JPEG serialization, queue drain and closed-recorder behavior. They do NOT test CUDA or real-robot timing.
