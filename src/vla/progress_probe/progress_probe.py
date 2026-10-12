"""ProgressProbe: read task progress [0, 1] out of the VLA's own DiT latent.

One hook, one or many heads: a probe .pt is fitted per (checkpoint, prompt),
so a run that switches prompts (--cortex) keeps an index prompt -> .pt and
selects (and lazily loads) the one fitted on the live instruction
(see probe_bank.pick).

A linear probe (ridge regression, nn.Linear(2048, 1) + feature statistics)
fitted on the 2048-dim mean-pooled output of `action_head.vl_self_attention`
— see vla_data/README.md §2. The heavy part is the VLA itself: a forward
hook copies the latent out of the `get_action` forward that runs anyway,
so `read()` after `predict()` costs one dot product on the GPU (fused
normalize + linear) plus one .item() sync.

The probe is NOT standalone: its weights live in the latent space of the
one checkpoint it was fitted against (`meta["extractor_checkpoint"]`, the
published groot_n17_0820succ_6k_step4500) under that checkpoint's training
prompt. A different extractor or prompt gives meaningless scores — attach()
and check_prompt() print what was recorded so the operator can verify the
pairing.

This module imports torch — inference-container only, like sonic_policy.
"""

import json
import os

import torch

from .probe_bank import pick

FEATURE_DIM = 2048


class ProbeHead:
    """One probe .pt folded into a single dot product (+ its fit metadata)."""

    def __init__(self, probe_path: str):
        payload = torch.load(probe_path, map_location="cpu", weights_only=False)
        linear = torch.nn.Linear(FEATURE_DIM, 1)
        linear.load_state_dict(payload["w"])
        linear.eval()
        w = linear.weight.detach().squeeze(0).float()   # (2048,)
        b = float(linear.bias.detach().item())
        mu = payload["mu"].float()                       # (2048,)
        sd = payload["sd"].float()                       # (2048,)
        # Fuse (x - mu)/sd then Linear into ONE dot product:
        #   y = w·((x-mu)/sd) + b = (w/sd)·x + (b - (w/sd)·mu)
        # Saves 2048 subs + 2048 divs per read().
        self.w_eff = (w / sd).contiguous()               # (2048,)
        self.b_eff = b - float((self.w_eff * mu).sum().item())
        self.meta: dict = payload.get("meta", {})
        self.path = probe_path
        self.name = os.path.splitext(os.path.basename(probe_path))[0]
        self.prompt: str | None = self.meta.get("prompt")
        self.extractor: str = self.meta.get("extractor_checkpoint", "<unrecorded>")

    def score(self, feature: "torch.Tensor") -> float:
        if self.w_eff.device != feature.device:
            self.w_eff = self.w_eff.to(feature.device)
        with torch.no_grad():
            # One dot product + bias. .item() drains the sync predict() left.
            return float((self.w_eff @ feature).item()) + self.b_eff


class ProgressProbe:
    """Hook a Gr00tN1d7 model once; score each prediction with the active head.

    `path` is one .pt (that head is active from the start, as before) or a
    JSON index {"<fit prompt>": "<relative .pt path>", ...} — a bank, one head
    per fitted prompt. With a bank nothing is active until select(): the
    runner calls it with the live instruction whenever the subtask changes;
    the matching .pt is loaded on first use (25 KB, milliseconds) and cached,
    and read() returns None while no head matches (no score, no verdict).

    Lifecycle:

        probe = ProgressProbe("shared/probe/probes.json")   # or ".../probe_pick.pt"
        probe.attach(policy.torch_model)                    # forward hook, model untouched
        probe.select(instruction, override=None)            # per subtask (bank) / once (file)
        ...
        chunk = policy.predict(observation)                 # hook fires inside this call
        progress = probe.read()                             # score for THAT observation, or None
    """

    def __init__(self, path: str):
        self._cache: dict[str, ProbeHead] = {}       # .pt path -> loaded head
        self._feature: torch.Tensor | None = None
        self._handle = None
        if path.endswith(".json"):
            with open(path) as f:
                index: dict[str, str] = json.load(f)
            base = os.path.dirname(os.path.abspath(path))
            # prompt -> absolute .pt path; nothing loaded yet
            self.index = {prompt: os.path.join(base, rel) for prompt, rel in index.items()}
            self.active: ProbeHead | None = None
            self.strict = True
            print(f"[ProgressProbe] index {path}: {len(self.index)} prompts")
            for prompt, pt in self.index.items():
                print(f"[ProgressProbe]   {os.path.basename(pt)}  <-  {prompt!r}")
        else:
            head = self._load(path)
            self.index = {head.prompt or "": path}
            self.active = head
            self.strict = False
        print("[ProgressProbe] the running checkpoint must be the model the probes were fitted on")

    def _load(self, pt_path: str) -> ProbeHead:
        head = self._cache.get(pt_path)
        if head is None:
            head = ProbeHead(pt_path)
            self._cache[pt_path] = head
            print(f"[ProgressProbe] loaded {head.name}  prompt={head.prompt!r}  extractor={head.extractor}")
        return head

    # ── head selection ────────────────────────────────────────────────────

    @property
    def active_name(self) -> str | None:
        return None if self.active is None else self.active.name

    def select(self, instruction: str, override: str | None = None) -> str | None:
        """Pick the head fitted on `instruction` (or the override, by file
        stem). Returns its name, or None when nothing matches — then read()
        yields None. A single-file probe stays active and only warns."""
        if not self.strict and override is None:
            self.check_prompt(instruction)
            return self.active_name
        chosen = pick(list(self.index.items()), instruction, override)
        if chosen.path is None:
            self.active = None
        else:
            self.active = self._load(chosen.path)
            if self.active.prompt is not None and self.active.prompt != instruction and override is None:
                print(f"[ProgressProbe] WARNING: index says {self.active.name} for this prompt "
                      f"but its .pt was fitted on {self.active.prompt!r}")
        print(f"[ProgressProbe] select: {chosen.reason}")
        return self.active_name

    # keep the old single-probe accessor
    @property
    def meta(self) -> dict:
        return {} if self.active is None else self.active.meta

    def check_prompt(self, prompt: str) -> None:
        """Warn when the runner's prompt differs from the probe's fit prompt."""
        fit_prompt = None if self.active is None else self.active.prompt
        if fit_prompt is not None and fit_prompt != prompt:
            print(
                "[ProgressProbe] WARNING: prompt differs from the probe's fit "
                f"prompt — scores may not be meaningful.\n"
                f"  fit:     {fit_prompt!r}\n"
                f"  running: {prompt!r}"
            )

    def attach(self, model: torch.nn.Module) -> None:
        """Hook `model.action_head.vl_self_attention` (fires once per predict)."""
        module = model.action_head.vl_self_attention

        def grab(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            # (B, seq, 2048) -> (2048,): stay on GPU (no .cpu() mid-forward).
            # read() forces the sync at the end via .item() — this preserves
            # CUDA overlap during predict().
            self._feature = tensor.detach().float().mean(dim=1).squeeze(0)

        self._handle = module.register_forward_hook(grab)

    def read(self) -> float | None:
        """Progress of the latest prediction — None before the first one, and
        None while no head is active (bank with no match for this prompt)."""
        if self._feature is None or self.active is None:
            return None
        return self.active.score(self._feature)

    def feature_for_recording(self) -> "torch.Tensor | None":
        """Latest pooled 2048-D feature; caller snapshots before next inference.

        Returned tensor may be on CUDA; CPU copy can synchronize inference.
        """
        return self._feature

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
