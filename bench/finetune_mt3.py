"""Fine-tune a YourMT3+ checkpoint on the synthetic electronic dataset
(bench/edm_dataset.py), using the authors' own training loop.

    python bench/finetune_mt3.py --data data/edm --variant moe-nops --steps 3000

The released checkpoint is copied into a new experiment folder; YourMT3's
train.py then loads its weights and starts a fresh run (new optimiser and
schedule). The result is registered as variant "<variant>-edm" in
<model dir>/YourMT3/amt/logs/2024/<exp>/checkpoints/last.ckpt.
"""

import argparse
import contextlib
import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")  # torch.compile needs Triton, which Windows lacks
import copy
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from musictool.mt3 import VARIANTS, ckpt_path, ymt3_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/edm")
    ap.add_argument("--variant", default="moe-nops", choices=list(VARIANTS))
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bsz", type=int, nargs=2, default=[2, 4], help="sub-batch and local batch size")
    ap.add_argument("--exp", default=None)
    ap.add_argument("--freeze-encoder", action="store_true",
                    help="train only the decoder (no encoder activations are kept)")
    ap.add_argument("--checkpointing", action="store_true",
                    help="gradient checkpointing in the T5 decoder: recompute activations in the backward pass")
    a = ap.parse_args()

    root = ymt3_dir(a.variant)
    src = root / "amt" / "src"
    data_home = Path(a.data).resolve()
    checkpoint, loader = VARIANTS[a.variant]
    exp = a.exp or f"{checkpoint.split('@')[0]}_edm"

    # new experiment folder seeded with the released weights
    exp_dir = root / "amt" / "logs" / "2024" / exp / "checkpoints"
    exp_dir.mkdir(parents=True, exist_ok=True)
    seed = exp_dir / "last.ckpt"
    if not seed.exists():
        shutil.copy(root / ckpt_path(a.variant), seed)

    sys.path.insert(0, str(src))
    os.chdir(root)  # the authors' paths (amt/logs, ...) are relative to the repo root
    import config.config as C
    import config.data_presets as DP

    C.shared_cfg["PATH"]["data_home"] = str(data_home)
    # Windows starts DataLoader workers by re-importing modules, which the exec'd train module
    # below can't survive; load data in the main process.
    C.shared_cfg["DATAIO"]["num_workers"] = 0
    C.shared_cfg["DATAIO"]["prefetch_factor"] = None
    C.shared_cfg["DATAIO"]["persistent_workers"] = False
    DP.data_preset_single_cfg["edm"] = {**copy.deepcopy(DP.data_preset_single_cfg["slakh"]), "dataset_name": "edm"}

    loader = [x for x in loader]  # model architecture args as the inference loader uses
    sys.argv = ["train.py", f"{exp}@last.ckpt", "-p", "2024", "-d", "edm", *loader, "-pr", "16",
                "-bsz", str(a.bsz[0]), str(a.bsz[1]), "-it", str(a.steps), "-lr", str(a.lr), "-o", "AdamW",
                "-s", "cosine", "-wb", "offline", "-se", "40000", "-vit", str(max(500, a.steps // 4))]
    print(" ".join(sys.argv))
    # This release's initialize_trainer returns no wandb logger, but train.main() still
    # logs to it; run train.py with those logging lines dropped.
    import re
    import types

    code = (src / "train.py").read_text(encoding="utf-8")
    code = re.sub(r"\n\s*if trainer\.global_rank == 0:\n(\s*wandb_logger\.experiment[^\n]*\n)+", "\n", code)
    code = re.sub(r"\n\s*wandb_logger\.watch\([^\n]*\)", "\n", code)
    # the seed is the same pinned release checkpoint mt3.py already loads this way
    code = code.replace('torch.load(dir_info["last_ckpt_path"])', 'torch.load(dir_info["last_ckpt_path"], weights_only=False)')
    if a.freeze_encoder:
        code = code.replace(
            "        model.load_state_dict(state_dict, strict=False)\n",
            "        model.load_state_dict(state_dict, strict=False)\n"
            "        for _name, _p in model.named_parameters():\n"
            "            if _name.split('.')[0] in ('encoder', 'pre_encoder'):\n"
            "                _p.requires_grad = False\n", 1)
        assert "_p.requires_grad = False" in code, "train.py changed; can't freeze the encoder"
    if a.checkpointing:
        code = code.replace(
            "        model.load_state_dict(state_dict, strict=False)\n",
            "        model.load_state_dict(state_dict, strict=False)\n"
            "        for _m in model.modules():\n"
            "            if hasattr(_m, 'gradient_checkpointing'):\n"
            "                _m.gradient_checkpointing = True\n"
            # with checkpointing the T5 stack returns no cache; ymt3.forward unpacks two values
            "        _dec_fwd = model.decoder.forward\n"
            "        def _dec_two(*a, **k):\n"
            "            out = _dec_fwd(*a, **k)\n"
            "            return (out[0], None) if isinstance(out, tuple) and len(out) == 1 else out\n"
            "        model.decoder.forward = _dec_two\n", 1)
    train = types.ModuleType("train")
    train.__file__ = str(src / "train.py")
    exec(compile(code, str(src / "train.py"), "exec"), train.__dict__)
    started = __import__("time").time()
    train.main()

    # Lightning saves under lightning_logs/version_N (this release has no wandb logger to
    # redirect it); move the result into the experiment folder that musictool.mt3 loads.
    runs = [p for p in (root / "lightning_logs").glob("version_*/checkpoints/last.ckpt") if p.stat().st_mtime > started]
    if not runs:
        raise SystemExit("training finished but no checkpoint was written")
    shutil.copy(max(runs, key=lambda p: p.stat().st_mtime), seed)
    print("fine-tuned checkpoint:", seed)


if __name__ == "__main__":
    main()
