"""The VLA runner: cameras + robot state -> GR00T inference -> 50 Hz stream.

The end-to-end assembly of the stage pipeline — everything else in vla/ is
a stage, this is the wiring:

    io          ColorSubscriber per checkpoint view + UnitreeStateReader,
                all on ONE DomainParticipant
    observation ObservationBuilder — fresh snapshots or nothing
    policy      SonicPolicy — Observation -> 40-step ActionChunk (~135 ms)
    chunking    ChunkCursor — the seam between the two clocks
    publisher   LatentActionStreamer — 50 Hz onto rt/kist/latent_action

Two loops: the streamer's 50 Hz Tx thread (its own), and THIS main thread
running inference back-to-back — every prediction is pushed with the ticks
it spent in flight already skipped, so the cursor splices it in at "now".
No inference throttle: a faster GPU simply replaces chunks more often, and
the horizon (0.78 s) rides out the slow moments. Ctrl+C stops the stream;
gearsonic's LOST recovery brings the robot to the safe standing pose.

Which cameras exist is not configured here: the checkpoint's modality
config names its views, and `cameras` only maps view names to
ext-sensor-io camera names — a 1-view checkpoint runs one subscriber, the
3-view one runs three.
"""

import os
import time
from dataclasses import dataclass, field

import tyro

from common.cyclonedds.config import apply_cyclonedds_xml, load_dds_config

from .chunking import ChunkCursor
from .io.realsense import ColorSubscriber, color_topic_for
from .io.unitree import UnitreeStateReader
from .observation import ObservationBuilder
from .policy import SonicPolicy
from .publisher import LatentActionStreamer
from .publisher.latent_action_streamer import CONTROL_DT_NS

# The training prompt — a single-task checkpoint conditions on this exact
# string (UNITREE_G1_SONIC_3VIEWS.md §5); a different checkpoint needs its
# own training prompt passed via --prompt.
DEFAULT_PROMPT = (
    "Open the right door of the refrigerator. Hook the yellow tip attached "
    "to your right hand under the door handle and pull."
)


@dataclass
class Config:
    checkpoint: str = "/workspace/checkpoints/checkpoint-4500"
    """Finetuned checkpoint directory (mounted by docker/run.sh)."""

    prompt: str = DEFAULT_PROMPT
    """Task instruction — MUST match the checkpoint's training string.
    With --cortex this is only the warmup prompt; live instructions come
    from the orchestrator."""

    cortex: bool = False
    """Take instructions live from the cortex orchestrator
    (rt/cortex/vla/cmd -> instruction, rt/cortex/vla/state <- 10 Hz status)
    instead of the fixed --prompt. IDLE holds the last commanded posture."""

    step_timeout_s: float = 0.0
    """--cortex only: fail a RUNNING subtask after this many seconds
    (detail "timeout <s>s", posture held). <= 0 disables (v0.1 default —
    tune after real-robot measurement)."""

    embodiment_tag: str = "unitree_g1_sonic_3views"
    """Which of the checkpoint's embodiments to run."""

    device: str = "cuda:0"
    """CUDA device for the model."""

    cameras: dict[str, str] = field(
        default_factory=lambda: {
            "ego_view": "head",
            "left_wrist": "left_wrist",
            "right_wrist": "right_wrist",
        }
    )
    """Checkpoint view name -> ext-sensor-io camera name. Only the views the
    checkpoint's modality config asks for are subscribed."""

    config: str = "config/config.yaml"
    """Network settings (dds: domain_id, cyclonedds_xml) — gearsonic-style.
    A missing file at this default path falls back to built-in defaults."""

    domain: int | None = None
    """DDS domain id override (must match gearsonic). Default: the config
    file's dds.domain_id."""

    report_s: float = 2.0
    """Seconds between status lines (inference latency, cursor health)."""

    probe: str | None = None
    """Progress probe: one .pt, or a JSON index {"<fit prompt>": "<.pt>", ...}
    (shared/probe/probes.json). With the index the .pt whose prompt equals the
    live instruction is selected per subtask (--cortex) and loaded on first
    use; a cmd may force one with args=["probe=<name>"]. Only valid with the
    checkpoint the probes were fitted on. None disables the probe."""

    probe_dir: str = "shared/probe"
    """Where the per-rollout progress JSONL goes (host-mounted via shared/)."""

    record: bool = False
    """Enable read-only episode recording; requires --probe."""

    record_dir: str = "shared/episodes"
    """Host-mounted directory containing per-run recording sessions."""

    record_every_n: int = 5
    """Record every Nth prediction (CPU snapshot overhead remains)."""

    record_max_queue: int = 16
    """Bounded writer queue; drop samples when full."""

    # ProgressMonitor thresholds (--probe with --cortex). Tune on real-robot
    # logs; the viewer replays a log under different values (view_progress.py).
    done_threshold: float = 0.70
    """progress >= this -> DONE at once."""

    stuck_value_threshold: float = 0.55
    """Flat below this -> STALLED candidate; flat above -> stays RUNNING."""

    slope_stuck_threshold: float = 0.027
    """progress slope (per s) over --slope-window-s under which it is 'flat'."""

    slope_window_s: float = 5.0
    """Sliding window for the slope."""

    stall_min_elapsed_s: float = 10.0
    """No STALLED before this many seconds into a subtask (approach phase)."""

    stall_hold_s: float = 3.0
    """Low-and-flat must persist this long before STALLED."""

    task: str | None = None
    """Short task name for the progress log file name
    (<task>_<checkpoint>_<probe>_<NN>_<time>.jsonl). Default: the first words
    of --prompt, or "cortex" with --cortex (live instructions)."""


def main(config: Config) -> None:
    dds_cfg = load_dds_config(config.config)
    apply_cyclonedds_xml(dds_cfg.cyclonedds_xml)
    domain = config.domain if config.domain is not None else dds_cfg.domain_id

    # Model first: 25+ s of loading during which DDS would just sit idle.
    print(f"Loading checkpoint {config.checkpoint} ({config.embodiment_tag})...")
    policy = SonicPolicy(
        config.checkpoint, device=config.device, embodiment_tag=config.embodiment_tag
    )
    views = policy.video_views
    missing = [v for v in views if v not in config.cameras]
    if missing:
        raise SystemExit(
            f"checkpoint wants views {views} but --cameras has no mapping for "
            f"{missing} (ext-sensor-io camera name needed)"
        )
    print(f"Checkpoint views: {views}")

    # Progress probe (optional): hooks the DiT latent inside predict() —
    # the score is a free byproduct of each inference, logged per rollout.
    probe = None
    progress_log = None
    if config.probe is not None:
        from .progress_probe import ProgressLog, ProgressProbe
        from .progress_probe.progress_log import run_stem

        probe = ProgressProbe(config.probe)
        probe.attach(policy.torch_model)
        if not config.cortex:
            # Fixed prompt: pick its head once (a single .pt stays active and
            # only warns on mismatch; a bank may yield none -> no score).
            probe.select(config.prompt)
        task = config.task or ("cortex" if config.cortex else config.prompt)
        probe_label = os.path.splitext(os.path.basename(config.probe))[0]  # "probes" for the index
        stem = run_stem(
            config.probe_dir,
            task,
            os.path.basename(os.path.normpath(config.checkpoint)),
            probe_label,
        )
        progress_log = ProgressLog(config.probe_dir, stem)

    # One participant for every Rx source (ChannelFactory convention); the
    # streamer's writer owns its Tx side separately.
    from cyclonedds.domain import DomainParticipant

    participant = DomainParticipant(domain)
    cameras = {}
    for view in views:
        camera = ColorSubscriber(view, topic=color_topic_for(config.cameras[view]))
        camera.start(participant=participant)
        cameras[view] = camera
    state_reader = UnitreeStateReader()
    state_reader.start(participant=participant)
    builder = ObservationBuilder(cameras, state_reader)

    cursor = ChunkCursor()
    streamer = LatentActionStreamer()
    tick_s = CONTROL_DT_NS / 1e9

    # Cortex bridge (optional): DDS reader/writer + 10 Hz publisher around
    # the SubtaskMachine. We hold the machine separately so the runner can
    # feed it progress verdicts from the probe (see below).
    machine = None
    bridge = None
    if config.cortex:
        from .cortex import CortexBridge, Effect, SubtaskMachine

        machine = SubtaskMachine(step_timeout_s=config.step_timeout_s)
        bridge = CortexBridge(machine, cursor)

    # Progress monitor (only when probe AND cortex are both on): turns the
    # probe raw stream into a DONE/STALLED verdict for the machine. Reset
    # on every new subtask so the running max doesn't carry over.
    monitor = None
    last_subtask_id: tuple[str, int] = ("", 0)
    if probe is not None and machine is not None:
        from .progress_probe import ProgressMonitor

        monitor = ProgressMonitor(
            done_threshold=config.done_threshold,
            stuck_value_threshold=config.stuck_value_threshold,
            slope_stuck_threshold=config.slope_stuck_threshold,
            window_s=config.slope_window_s,
            stall_min_elapsed_s=config.stall_min_elapsed_s,
            stall_hold_s=config.stall_hold_s,
        )

    recorder = None
    record_capture_errors = 0
    if config.record:
        if probe is None:
            raise SystemExit("--record requires --probe to capture the DiT feature")
        if config.record_every_n < 1 or config.record_max_queue < 1:
            raise SystemExit("--record-every-n and --record-max-queue must be >= 1")
        from .recording import EpisodeRecorder, EpisodeSample, snapshot_numpy

        session = f"session_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1000000000:09d}"
        recorder = EpisodeRecorder(
            os.path.join(config.record_dir, session),
            max_queue=config.record_max_queue,
            metadata={
                "checkpoint": config.checkpoint,
                "embodiment_tag": config.embodiment_tag,
                "probe": config.probe,
                "prompt_initial": config.prompt,
                "cortex": config.cortex,
                "views": views,
                "record_every_n": config.record_every_n,
                "progress_log": None if progress_log is None else progress_log.path,
                "action_note": "predicted ActionChunk; not actual executed feedback",
            },
        )
        print(f"[EpisodeRecorder] recording to {recorder.output_dir}")

    try:
        # Wait for the sensors, then warm up the model off-stream: the first
        # prediction pays CUDA init (~1 s) — better spent before gearsonic
        # can see us than as a stale first chunk.
        print("Waiting for fresh sensor streams...")
        observation = None
        while observation is None:
            observation = builder.build(config.prompt)
            if observation is None:
                time.sleep(0.05)
        print("Warmup inference...")
        policy.predict(observation)

        streamer.start(cursor, domain_id=domain)
        if bridge is not None:
            bridge.start(domain_id=domain)
            print("Streaming — cortex mode: idle until the first SubtaskCmd")
        else:
            print(f"Streaming — prompt: {config.prompt!r}")

        last_report = time.monotonic()
        latency_sum = 0.0
        predictions = 0
        record_seq = 0
        while True:
            t0 = time.monotonic()
            prompt = bridge.instruction() if bridge is not None else config.prompt
            if prompt is None:
                # No RUNNING subtask: the frozen cursor keeps the posture (or
                # stays silent before the first task) — no inference to run.
                time.sleep(tick_s)
                continue
            observation = builder.build(prompt)
            if observation is None:
                # The builder already reported which stream is stale; the
                # cursor plays out its horizon, then hands over to recovery.
                time.sleep(tick_s)
                continue

            observation_monotonic_ns = time.monotonic_ns() if recorder is not None else 0
            chunk = policy.predict(observation)
            if bridge is not None and bridge.instruction() != prompt:
                # The subtask was cancelled while this
                # prediction was in flight: pushing it would unpin the
                # frozen posture (or act on the old instruction) — drop it.
                continue
            elapsed = time.monotonic() - t0
            cursor.push(chunk, skip_ticks=round(elapsed / tick_s))

            progress = None
            reading = None
            if probe is not None:
                # New subtask? (cortex) Select the head fitted on its prompt —
                # or the cmd's probe= override — and reset the running max,
                # BEFORE reading this prediction's score.
                if machine is not None:
                    current_subtask_id = machine.subtask_id()
                    if current_subtask_id != last_subtask_id:
                        probe.select(prompt, override=machine.probe_override())
                        if monitor is not None:
                            monitor.reset()
                        last_subtask_id = current_subtask_id
                # The hook fired inside predict(); this is that observation's
                # score. One dot product on GPU + .item() sync — negligible.
                # None = no head for this prompt: no score, no verdict.
                raw = probe.read()
                t_probe = time.monotonic()
                # Perception -> probe latency budget, per prediction:
                #   frame_age_ms  oldest camera frame's age at observation build
                #   probe_lag_ms  that frame's DDS arrival -> probe score
                #                 (= frame_age + build/predict/read, our clock)
                #   sensor_lag_ms sensor capture stamp -> probe score
                #                 (ext-sensor-io's clock; needs synced clocks)
                timing = {"probe": probe.active_name}
                if observation.camera_age_s:
                    frame_age_s = max(observation.camera_age_s.values())
                    oldest_stamp_ns = min(observation.camera_stamp_ns.values())
                    timing.update({
                        "frame_age_ms": frame_age_s * 1e3,
                        "probe_lag_ms": (frame_age_s + (t_probe - t0)) * 1e3,
                        "sensor_lag_ms": time.time() * 1e3 - oldest_stamp_ns / 1e6,
                    })
                if raw is not None and monitor is not None:
                    now = time.monotonic()
                    reading = monitor.update(raw, now)
                    # Feed the verdict to the machine — DONE/STALLED transition.
                    # Apply its cursor effect exactly as the bridge does for
                    # cancel/timeout: FREEZE pins the last posture, otherwise
                    # the chunk runs out and gearsonic blends to safe standing.
                    effect = machine.on_progress(reading.state, reading.progress, now)
                    if effect is Effect.FREEZE:
                        cursor.freeze()
                    progress = reading.progress
                else:
                    progress = raw
                progress_log.append(progress, elapsed * 1e3, reading=reading, timing=timing)

            # Research recorder never modifies robot control or FSM decisions.
            # Snapshots (especially CUDA -> CPU) CAN delay the next inference.
            record_seq += 1
            if recorder is not None and record_seq % config.record_every_n == 0:
                try:
                    feature = probe.feature_for_recording()
                    if feature is not None:
                        sample = EpisodeSample(
                            episode_id="run_0001",  # viewer marks require offline alignment
                            sample_id=record_seq,
                            observation_monotonic_ns=observation_monotonic_ns,
                            inference_monotonic_ns=time.monotonic_ns(),
                            progress=None if progress is None else float(progress),
                            progress_state=None if reading is None else reading.state.value,
                            latent=snapshot_numpy(feature),
                            action_chunk={
                                "motion_token": snapshot_numpy(chunk.motion_token),
                                "left_hand_joints": snapshot_numpy(chunk.left_hand_joints),
                                "right_hand_joints": snapshot_numpy(chunk.right_hand_joints),
                            },
                            rgb={name: snapshot_numpy(frame) for name, frame in observation.video.items()},
                            robot_state={name: snapshot_numpy(state) for name, state in observation.state.items()},
                            extras={
                                "wall_time_ns": time.time_ns(),
                                "prompt": prompt,
                                "probe_name": probe.active_name,
                                "inference_ms": elapsed * 1000.0,
                                "camera_stamp_ns": observation.camera_stamp_ns,
                                "camera_age_s": observation.camera_age_s,
                                "subtask_id": None if machine is None else str(machine.subtask_id()),
                            },
                        )
                        recorder.submit(sample)  # drops if queue full, never blocks
                except Exception as exc:
                    record_capture_errors += 1
                    if record_capture_errors <= 3:
                        print(f"[EpisodeRecorder] capture skipped: {exc!r}", flush=True)

            latency_sum += elapsed
            predictions += 1
            if t0 - last_report >= config.report_s:
                stats = cursor.stats()
                progress_part = (
                    "" if probe is None
                    else f"progress {'-' if progress is None else f'{progress:.2f}'} | "
                )
                print(
                    f"inference {latency_sum / predictions * 1e3:.0f}ms avg "
                    f"({predictions / (t0 - last_report):.1f}/s) | "
                    f"{progress_part}"
                    f"published {streamer.published} | held {stats['held']} "
                    f"starved {stats['starved']} stale {stats['stale_pushes']} "
                    f"late {streamer.late}",
                    flush=True,
                )
                last_report = t0
                latency_sum = 0.0
                predictions = 0
    except KeyboardInterrupt:
        print("\nStopping (gearsonic recovers to safe standing on stream loss)")
    finally:
        if bridge is not None:
            bridge.stop()
        streamer.stop()
        for camera in cameras.values():
            camera.stop()
        state_reader.stop()
        if probe is not None:
            probe.detach()
            progress_log.close()
        if recorder is not None:
            flushed = recorder.close(timeout=15)
            print(f"[EpisodeRecorder] closed={flushed} stats={recorder.stats()} "
                  f"capture_errors={record_capture_errors}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Config))
