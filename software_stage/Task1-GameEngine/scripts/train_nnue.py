#!/usr/bin/env python3
"""
Train a sparse NNUE-style evaluator for chess6x6 and export runtime weights.

Input (supervised): tuning text with one position per line:
  36 ints board + side_to_move + result
where result is in {0.0, 0.5, 1.0}.

Optional TD-leaf fine-tune:
  JSONL file, one game per line:
  {"result": 1.0, "positions": [{"board":[36 ints], "side":0}, ...]}

Exports:
  - PyTorch checkpoints (.pt), resumable
  - Engine runtime model (.nnue) binary
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import struct
import subprocess
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import logging
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, Dataset

# Suppress verbose torch.compile/inductor "failed to eagerly compile backwards for dynamic"
# warnings. These are harmless — inductor falls back to lazy backward compilation.
# We use a root-logger filter so it survives torch._dynamo.reset() which reinitialises
# torch's internal logging state.
_TORCH_NOISY_PATTERNS = (
    "failed to eagerly compile backwards",
    "CUDAGraph supports dynamic shapes",
)

class _TorchCompileFilter(logging.Filter):
    def filter(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            return True
        return not any(p in msg for p in _TORCH_NOISY_PATTERNS)

def _suppress_torch_compile_warnings():
    root = logging.getLogger()
    for f in list(root.filters):
        if isinstance(f, _TorchCompileFilter):
            root.removeFilter(f)
    root.addFilter(_TorchCompileFilter())
    for name in ("torch._functorch._aot_autograd.graph_compile", "torch._inductor"):
        logging.getLogger(name).setLevel(logging.ERROR)

_suppress_torch_compile_warnings()

FEATURE_COUNT = 36 * 10  # piece_id(1..10) x square(0..35)
NNUE_MAGIC = 0x3145554E
NNUE_VERSION_V1 = 1
NNUE_VERSION_V2 = 2  # adds optional second hidden layer


def board_to_features(board36: Sequence[int]) -> List[int]:
    feats: List[int] = []
    for sq, pid in enumerate(board36):
        if pid:
            feats.append((pid - 1) * 36 + sq)
    return feats


def mirror_board(board36: Sequence[int]) -> List[int]:
    """Horizontal mirror of a 6x6 board (flip files a-f ↔ f-a)."""
    mirrored = [0] * 36
    for sq in range(36):
        rank, file = divmod(sq, 6)
        mirrored[rank * 6 + (5 - file)] = board36[sq]
    return mirrored


def texel_prob(scores_cp: torch.Tensor, k: float) -> torch.Tensor:
    scale = k * math.log(10.0) / 200.0
    return torch.sigmoid(scores_cp * scale)


class PositionDataset(Dataset):
    def __init__(self, rows: Sequence[Tuple[List[int], int, float, float]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        return self.rows[idx]


def collate_positions(batch):
    feat_flat: List[int] = []
    offsets = [0]
    sides = []
    score_cps = []
    targets = []
    for feats, side, score_cp, result in batch:
        feat_flat.extend(feats)
        offsets.append(len(feat_flat))
        sides.append(side)
        score_cps.append(score_cp)
        targets.append(result)
    return (
        torch.tensor(feat_flat, dtype=torch.long),
        torch.tensor(offsets, dtype=torch.long),
        torch.tensor(sides, dtype=torch.long),
        torch.tensor(score_cps, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
    )


class SparseNNUE(nn.Module):
    def __init__(self, hidden_size: int, hidden2_size: int = 0, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.hidden2_size = hidden2_size
        self.feat = nn.EmbeddingBag(
            FEATURE_COUNT, hidden_size, mode="sum", include_last_offset=True
        )
        self.stm = nn.Embedding(2, hidden_size)
        self.hidden_bias = nn.Parameter(torch.zeros(hidden_size))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if hidden2_size > 0:
            self.hidden2 = nn.Linear(hidden_size, hidden2_size)
            self.out = nn.Linear(hidden2_size, 1)
            nn.init.normal_(self.hidden2.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.hidden2.bias)
        else:
            self.hidden2 = None
            self.out = nn.Linear(hidden_size, 1)

        nn.init.normal_(self.feat.weight, mean=0.0, std=0.03)
        nn.init.normal_(self.stm.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.out.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.out.bias)

    def forward(self, feat_idx: torch.Tensor, offsets: torch.Tensor, stm: torch.Tensor):
        x = self.feat(feat_idx, offsets) + self.stm(stm) + self.hidden_bias
        x = torch.clamp(x, 0.0, 1.0)  # clipped ReLU
        x = self.dropout(x)
        if self.hidden2 is not None:
            x = self.hidden2(x)
            x = torch.clamp(x, 0.0, 1.0)  # clipped ReLU
        return self.out(x).squeeze(-1)  # centipawn-scale scalar


@dataclass
class TdGame:
    features: List[List[int]]
    sides: List[int]
    result: float


def load_supervised_rows(path: Path, augment: bool = False) -> List[Tuple[List[int], int, float, float]]:
    rows: List[Tuple[List[int], int, float, float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 38:
                continue
            board = list(map(int, parts[:36]))
            side = int(parts[36])
            if len(parts) >= 39:
                # New format: board[36] side score_cp result
                score_cp = float(parts[37])
                score_cp = max(-3000.0, min(3000.0, score_cp))  # clamp extremes
                result = float(parts[38])
            else:
                # Old format: board[36] side result (no search score)
                score_cp = float('nan')
                result = float(parts[37])
            rows.append((board_to_features(board), side, score_cp, result))
            if augment:
                rows.append((board_to_features(mirror_board(board)), side, score_cp, result))
    return rows


def load_td_games(path: Path) -> List[TdGame]:
    games: List[TdGame] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            result = float(obj["result"])
            positions = obj["positions"]
            feats: List[List[int]] = []
            sides: List[int] = []
            for pos in positions:
                board = pos["board"]
                side = int(pos["side"])
                feats.append(board_to_features(board))
                sides.append(side)
            if feats:
                games.append(TdGame(features=feats, sides=sides, result=result))
    return games


def maybe_generate_supervised_data(args) -> Path:
    data_path = Path(args.data)
    if args.auto_datagen_games <= 0:
        return data_path

    project_dir = Path(__file__).resolve().parents[1]
    out_path = Path(args.auto_datagen_out) if args.auto_datagen_out else data_path

    if getattr(args, 'auto_datagen_use_fsf', False):
        return _run_fsf_datagen(project_dir, out_path, args.auto_datagen_games,
                                getattr(args, 'fsf_time', 100), args, jsonl_path=None)

    datagen_bin = project_dir / "datagen"

    if not datagen_bin.exists():
        subprocess.run(["make", "datagen"], cwd=project_dir, check=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    out_dir = Path(args.out_dir)
    for candidate in [out_dir / "nnue_latest.nnue",
                       project_dir / "models" / "nnue" / "nnue_latest.nnue"]:
        if candidate.exists():
            env["CHESS_NNUE_PATH"] = str(candidate)
            env["CHESS_USE_NNUE"] = "1"
            print(f"datagen: using NNUE {candidate}")
            break
    subprocess.run(
        [
            str(datagen_bin),
            str(args.auto_datagen_games),
            str(args.auto_datagen_depth),
            str(out_path),
        ],
        cwd=project_dir,
        check=True,
        env=env,
    )
    return out_path


def _run_fsf_datagen(
    project_dir: Path, out_path: Path, games: int, time_ms: int,
    args, jsonl_path: Path | None = None,
) -> Path:
    """Run datagen_fsf in --fsf-both mode with mixed skill levels."""
    datagen_bin = project_dir / "datagen_fsf"
    if not datagen_bin.exists():
        subprocess.run(["make", "datagen_fsf"], cwd=project_dir, check=True)

    skill = getattr(args, 'fsf_skill_level', 20)
    skill_min = getattr(args, 'fsf_skill_min', 5)
    skill_max = getattr(args, 'fsf_skill_max', 19)
    mix_pct = getattr(args, 'fsf_mix_pct', 30)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(datagen_bin), str(games), str(time_ms), str(out_path),
        "--fsf-both",
        "--skill-level", str(skill),
        "--skill-min", str(skill_min),
        "--skill-max", str(skill_max),
        "--mix-pct", str(mix_pct),
    ]
    if jsonl_path:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--jsonl", str(jsonl_path)])
        print(f"datagen_fsf: generating JSONL to {jsonl_path}")

    print(f"datagen_fsf: {games} games, {time_ms}ms/move, skill {skill} (weak {skill_min}-{skill_max}, {mix_pct}% 20v20)")
    subprocess.run(cmd, cwd=project_dir, check=True)
    return out_path


def run_datagen(project_dir: Path, out_path: Path, games: int, depth: int,
                jsonl_path: Path | None = None, use_fsf: bool = False, fsf_args=None,
                nnue_out_dir: Path | None = None):
    if use_fsf and fsf_args is not None:
        return _run_fsf_datagen(project_dir, out_path, games,
                                getattr(fsf_args, 'fsf_time', 100),
                                fsf_args, jsonl_path)

    datagen_bin = project_dir / "datagen"
    if not datagen_bin.exists():
        subprocess.run(["make", "datagen"], cwd=project_dir, check=True)

    # Use the current best/latest model for self-play data generation
    env = os.environ.copy()
    nnue_candidates = []
    if nnue_out_dir is not None:
        nnue_candidates.append(Path(nnue_out_dir) / "nnue_latest.nnue")
    nnue_candidates.append(project_dir / "models" / "nnue" / "nnue_latest.nnue")
    for candidate in nnue_candidates:
        if candidate.exists():
            env["CHESS_NNUE_PATH"] = str(candidate)
            env["CHESS_USE_NNUE"] = "1"
            print(f"datagen: using NNUE {candidate}")
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(datagen_bin), str(games), str(depth), str(out_path)]
    if jsonl_path:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        cmd.append(str(jsonl_path))
        print(f"datagen: generating JSONL to {jsonl_path}")

    subprocess.run(
        cmd,
        cwd=project_dir,
        check=True,
        env=env
    )


def regenerate_data(
    project_dir: Path,
    base_data_path: Path,
    games: int,
    depth: int,
    mode: str,
    max_rows: int = 0,
    jsonl_path: Path | None = None,
    use_fsf: bool = False,
    fsf_args=None,
    nnue_out_dir: Path | None = None,
) -> Path:
    if mode == "replace":
        run_datagen(project_dir, base_data_path, games, depth, jsonl_path,
                    use_fsf=use_fsf, fsf_args=fsf_args, nnue_out_dir=nnue_out_dir)
        return base_data_path

    # append mode: generate to temp then append lines
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", prefix="nnue_regen_", delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
    
    tmp_jsonl = None
    if jsonl_path:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", prefix="nnue_regen_", delete=False
        ) as tmpj:
            tmp_jsonl = Path(tmpj.name)

    run_datagen(project_dir, tmp_path, games, depth, tmp_jsonl,
                use_fsf=use_fsf, fsf_args=fsf_args, nnue_out_dir=nnue_out_dir)
    
    # Append text data
    with base_data_path.open("a", encoding="utf-8") as out_f, tmp_path.open(
        "r", encoding="utf-8"
    ) as in_f:
        for line in in_f:
            out_f.write(line)
    tmp_path.unlink(missing_ok=True)

    # Append JSONL data
    if jsonl_path and tmp_jsonl and tmp_jsonl.exists():
        with jsonl_path.open("a", encoding="utf-8") as out_j, tmp_jsonl.open(
            "r", encoding="utf-8"
        ) as in_j:
            for line in in_j:
                out_j.write(line)
        tmp_jsonl.unlink(missing_ok=True)

    # Apply sliding window (trim to max_rows)
    if max_rows > 0:
        with base_data_path.open("r", encoding="utf-8") as f:
            lines = deque(f, maxlen=max_rows)
        with base_data_path.open("w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"regen: trimmed {base_data_path} to {max_rows} rows")
        
        # Also trim JSONL if it exists (approximate game count based on positions)
        # Note: Trimming JSONL is harder because 1 row != 1 game. 
        # For now, we'll keep the full JSONL as it's used for TD-leaf trajectories.

    return base_data_path


def build_loaders(rows, args, amp):
    random.shuffle(rows)
    n_val = int(len(rows) * args.val_split)
    val_rows = rows[:n_val] if n_val > 0 else rows[:1]
    train_rows = rows[n_val:] if n_val > 0 else rows

    train_ds = PositionDataset(train_rows)
    val_ds = PositionDataset(val_rows)

    use_persistent = args.workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=amp,
        persistent_workers=use_persistent,
        prefetch_factor=4 if args.workers > 0 else None,
        collate_fn=collate_positions,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.workers // 2),
        pin_memory=amp,
        persistent_workers=use_persistent,
        prefetch_factor=4 if args.workers > 0 else None,
        collate_fn=collate_positions,
        drop_last=False,
    )
    return train_loader, val_loader, len(train_ds), len(val_ds)


def export_nnue_binary(model: SparseNNUE, path: Path):
    hidden = model.hidden_size
    hidden2 = model.hidden2_size
    feat_w = model.feat.weight.detach().cpu().contiguous().float().numpy()
    stm_w = model.stm.weight.detach().cpu().contiguous().float().numpy()
    h_bias = model.hidden_bias.detach().cpu().contiguous().float().numpy()
    out_w = model.out.weight.detach().cpu().contiguous().float().numpy().reshape(-1)
    out_b = float(model.out.bias.detach().cpu().item())

    version = NNUE_VERSION_V2 if hidden2 > 0 else NNUE_VERSION_V1

    with path.open("wb") as f:
        f.write(struct.pack("<IIII", NNUE_MAGIC, version, FEATURE_COUNT, hidden))
        if version == NNUE_VERSION_V2:
            f.write(struct.pack("<I", hidden2))
        f.write(h_bias.tobytes(order="C"))
        f.write(stm_w.reshape(-1).tobytes(order="C"))
        f.write(feat_w.reshape(-1).tobytes(order="C"))
        if hidden2 > 0:
            h2_w = model.hidden2.weight.detach().cpu().contiguous().float().numpy()
            h2_b = model.hidden2.bias.detach().cpu().contiguous().float().numpy()
            # Store as [H x H2] (transpose from PyTorch's [H2 x H])
            f.write(h2_w.T.tobytes(order="C"))
            f.write(h2_b.tobytes(order="C"))
        f.write(out_w.tobytes(order="C"))
        f.write(struct.pack("<f", out_b))


def save_checkpoint(
    ckpt_path: Path,
    model: SparseNNUE,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    epoch: int,
    step: int,
    best_val: float,
    args,
):
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "best_val": best_val,
        "args": vars(args),
    }
    if scheduler:
        state["scheduler_state"] = scheduler.state_dict()
    torch.save(state, ckpt_path)


def load_checkpoint(
    ckpt_path: Path,
    model: SparseNNUE,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
):
    d = torch.load(ckpt_path, map_location="cpu")
    saved_state = d["model_state"]
    current_state = model.state_dict()
    # Filter out keys with shape mismatch or that are missing in checkpoint
    compatible = {}
    skipped = []
    for k, v in saved_state.items():
        if k in current_state and current_state[k].shape == v.shape:
            compatible[k] = v
        else:
            skipped.append(k)
    missing_keys = [k for k in current_state if k not in compatible]
    arch_changed = bool(skipped or missing_keys)
    if arch_changed:
        print(f"[CKPT] Architecture change detected — loading compatible weights only")
        if skipped:
            print(f"[CKPT]   Skipped (shape mismatch): {skipped}")
        if missing_keys:
            print(f"[CKPT]   New layers (random init): {missing_keys}")
        model.load_state_dict(compatible, strict=False)
        print("[CKPT] Optimizer state reset due to architecture change")
        return int(d.get("epoch", 0)), int(d.get("step", 0)), float("inf")
    model.load_state_dict(saved_state)
    optimizer.load_state_dict(d["optimizer_state"])
    scaler_state = d.get("scaler_state", {})
    if scaler_state:
        scaler.load_state_dict(scaler_state)
    if scheduler and "scheduler_state" in d:
        scheduler.load_state_dict(d["scheduler_state"])
    return int(d.get("epoch", 0)), int(d.get("step", 0)), float(d.get("best_val", float("inf")))

def run_tdleaf_epoch(
    model: SparseNNUE,
    td_games: Sequence[TdGame],
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    k: float,
    lam: float,
    amp: bool,
    batch_size: int = 64,
) -> float:
    if not td_games:
        return 0.0
    model.train()
    random.shuffle(td_games)  # in-place order shuffle
    total = 0.0
    n_batches = 0

    for bi in range(0, len(td_games), batch_size):
        batch = td_games[bi : bi + batch_size]

        # Flatten all positions from all games in this batch into one forward pass
        feat_flat: List[int] = []
        offsets = [0]
        sides: List[int] = []
        game_lengths: List[int] = []    # number of positions per game
        results: List[float] = []

        for g in batch:
            for feats in g.features:
                feat_flat.extend(feats)
                offsets.append(len(feat_flat))
            sides.extend(g.sides)
            game_lengths.append(len(g.features))
            results.append(g.result)

        feat_t = torch.tensor(feat_flat, dtype=torch.long, device=device)
        off_t = torch.tensor(offsets, dtype=torch.long, device=device)
        side_t = torch.tensor(sides, dtype=torch.long, device=device)

        with torch.amp.autocast(device_type=device.type, enabled=amp):
            scores = model(feat_t, off_t, side_t)
            probs = texel_prob(scores, k)

            # Build TD-leaf targets per game (no_grad, backward through each game)
            with torch.no_grad():
                targets = torch.empty_like(probs)
                pos = 0
                for gi, glen in enumerate(game_lengths):
                    boot = torch.tensor(results[gi], dtype=probs.dtype, device=device)
                    for i in range(glen - 1, -1, -1):
                        idx = pos + i
                        if i < glen - 1:
                            boot = (1.0 - lam) * probs[idx + 1].detach() + lam * boot
                        targets[idx] = boot
                    pos += glen

            loss = torch.mean((probs - targets) ** 2)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.item())
        n_batches += 1

    return total / max(1, n_batches)


def evaluate_loader(
    model: SparseNNUE,
    loader: DataLoader,
    device: torch.device,
    k: float,
    amp: bool,
    blend_lambda: float = 0.0,
) -> float:
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for feat_idx, offsets, side, score_cp, target in loader:
            feat_idx = feat_idx.to(device, non_blocking=True)
            offsets = offsets.to(device, non_blocking=True)
            side = side.to(device, non_blocking=True)
            score_cp = score_cp.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                scores = model(feat_idx, offsets, side)
                pred = texel_prob(scores, k)
                if blend_lambda > 0.0:
                    score_prob = texel_prob(score_cp, k)
                    blended = torch.where(
                        torch.isfinite(score_cp),
                        (1.0 - blend_lambda) * target + blend_lambda * score_prob,
                        target,
                    )
                else:
                    blended = target
                loss = torch.mean((pred - blended) ** 2)
            total += float(loss.item()) * target.numel()
            n += target.numel()
    return total / max(1, n)


def main():
    ap = argparse.ArgumentParser(description="Train sparse NNUE for chess6x6")
    ap.add_argument("--data", required=True, help="supervised tuning data text file")
    ap.add_argument("--out-dir", default="models/nnue", help="checkpoint/export directory")
    ap.add_argument("--epochs", type=int, default=None, help="target epoch to stop at; if omitted, train indefinitely")
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--hidden-size", type=int, default=128)
    ap.add_argument("--hidden2-size", type=int, default=0, help="second hidden layer size (0=single layer)")
    ap.add_argument("--dropout", type=float, default=0.0, help="dropout rate on hidden layer (0=disabled)")
    ap.add_argument("--grad-clip", type=float, default=0.0, help="gradient norm clipping (0=disabled)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-scheduler", choices=["none", "step", "plateau"], default="none")
    ap.add_argument("--lr-step-size", type=int, default=10, help="epochs before decay for step scheduler")
    ap.add_argument("--lr-gamma", type=float, default=0.5, help="decay factor for scheduler")
    ap.add_argument("--lr-patience", type=int, default=5, help="patience for plateau scheduler")
    ap.add_argument("--lr-threshold", type=float, default=1e-5, help="threshold for measuring the new optimum")
    ap.add_argument("--lr-cooldown", type=int, default=0, help="epochs to wait before resuming normal operation after lr has been reduced")
    ap.add_argument("--weight-decay", type=float, default=1e-6)
    ap.add_argument("--k", type=float, default=1.2, help="Texel sigmoid K")
    ap.add_argument("--blend-lambda", type=float, default=0.5,
                    help="blend search score into training target: "
                         "target = (1-lam)*result + lam*sigmoid(score_cp*K); "
                         "0=pure WDL, 1=pure score; NaN scores fall back to WDL")
    ap.add_argument("--val-split", type=float, default=0.05)
    ap.add_argument("--augment", action="store_true", help="enable horizontal mirror augmentation (doubles data)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", default="", help="checkpoint path to resume")
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--tdleaf-jsonl", default="", help="optional trajectory jsonl")
    ap.add_argument("--tdleaf-lambda", type=float, default=0.0)
    ap.add_argument("--tdleaf-batch-size", type=int, default=64, help="games per TD-leaf batch (higher=faster)")
    ap.add_argument("--auto-datagen-games", type=int, default=0, help="if >0, generate supervised data before training")
    ap.add_argument("--auto-datagen-depth", type=int, default=3, help="search depth for auto datagen")
    ap.add_argument("--auto-datagen-out", default="", help="output path for auto-generated data (default: --data path)")
    ap.add_argument("--regen-every-epochs", type=int, default=0, help="if >0, regenerate supervised data every N epochs")
    ap.add_argument("--regen-games", type=int, default=0, help="number of games for each periodic regeneration")
    ap.add_argument("--regen-depth", type=int, default=3, help="search depth for periodic regeneration")
    ap.add_argument("--regen-mode", choices=["replace", "append"], default="replace", help="replace data file or append newly generated data")
    ap.add_argument("--regen-max-rows", type=int, default=0, help="maximum number of rows to keep in data file (sliding window)")
    ap.add_argument("--regen-out", default="", help="data path for periodic regeneration (default: training data path)")
    ap.add_argument("--lr-reset-on-regen", action="store_true", help="reset learning rate to initial value after data regeneration")
    # FSF data generation options
    ap.add_argument("--auto-datagen-use-fsf", action="store_true", help="use fairy-stockfish for initial data generation")
    ap.add_argument("--regen-use-fsf", action="store_true", help="use fairy-stockfish for periodic data regeneration")
    ap.add_argument("--fsf-time", type=int, default=100, help="movetime in ms for FSF data generation (default: 100)")
    ap.add_argument("--fsf-skill-level", type=int, default=20, help="FSF strong side skill level (default: 20)")
    ap.add_argument("--fsf-skill-min", type=int, default=5, help="FSF weak side min skill (default: 5)")
    ap.add_argument("--fsf-skill-max", type=int, default=19, help="FSF weak side max skill (default: 19)")
    ap.add_argument("--fsf-mix-pct", type=int, default=30, help="percentage of 20v20 games in FSF mode (default: 30)")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"

    # RTX 4060 / Ada Lovelace optimizations
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("medium")

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    export_path = out_dir / "nnue_latest.nnue"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = maybe_generate_supervised_data(args)
    if not data_path.exists():
        raise RuntimeError(f"training data not found: {data_path}")

    rows = load_supervised_rows(data_path, augment=args.augment)
    if not rows:
        raise RuntimeError(f"no training rows loaded from {data_path}")

    train_loader, val_loader, train_size, val_size = build_loaders(rows, args, amp)

    td_games: List[TdGame] = []
    if args.tdleaf_jsonl:
        td_games = load_td_games(Path(args.tdleaf_jsonl))
    if args.tdleaf_lambda > 0.0 and not args.tdleaf_jsonl:
        raise RuntimeError("tdleaf_lambda > 0 requires --tdleaf-jsonl")
    if args.tdleaf_lambda > 0.0 and len(td_games) == 0:
        raise RuntimeError("TD-leaf enabled but no trajectory games were loaded")

    model = SparseNNUE(hidden_size=args.hidden_size, hidden2_size=args.hidden2_size, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    scheduler = None
    if args.lr_scheduler == "step":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=args.lr_patience,
            factor=args.lr_gamma,
            threshold=args.lr_threshold,
            cooldown=args.lr_cooldown,
        )

    start_epoch = 1
    step = 0
    best_val = float("inf")
    resume_path = Path(args.resume) if args.resume else (ckpt_dir / "latest.pt")
    if resume_path.exists():
        start_epoch, step, best_val = load_checkpoint(resume_path, model, optimizer, scaler, scheduler)
        start_epoch += 1
        print(
            f"resumed from {resume_path} at epoch={start_epoch-1}, step={step}, best_val={best_val:.7f}"
        )

    print(
        f"device={device}, amp={amp}, train={train_size}, val={val_size}, "
        f"td_games={len(td_games)}, hidden={args.hidden_size}"
    )

    # torch.compile for fused GPU kernels (Ada Lovelace benefits significantly)
    # feat_idx has variable length per batch → skip CUDA graphs for dynamic shapes
    compiled_model = None
    if device.type == "cuda" and hasattr(torch, "compile"):
        try:
            torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
            compiled_model = torch.compile(model, mode="max-autotune", dynamic=True)
            print("torch.compile enabled (max-autotune, dynamic)")
        except Exception as e:
            print(f"torch.compile unavailable: {e}")
            compiled_model = None
    train_model = compiled_model if compiled_model is not None else model

    epoch = start_epoch
    epochs_since_regen = 0
    while True:
        if args.epochs is not None and epoch > args.epochs:
            break
        model.train()
        running = 0.0
        seen = 0

        for feat_idx, offsets, side, score_cp, target in train_loader:
            feat_idx = feat_idx.to(device, non_blocking=True)
            offsets = offsets.to(device, non_blocking=True)
            side = side.to(device, non_blocking=True)
            score_cp = score_cp.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=amp):
                scores = train_model(feat_idx, offsets, side)
                pred = texel_prob(scores, args.k)
                if args.blend_lambda > 0.0:
                    score_prob = texel_prob(score_cp, args.k)
                    blended_target = torch.where(
                        torch.isfinite(score_cp),
                        (1.0 - args.blend_lambda) * target + args.blend_lambda * score_prob,
                        target,
                    )
                else:
                    blended_target = target
                loss = torch.mean((pred - blended_target) ** 2)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            bsz = target.numel()
            running += float(loss.item()) * bsz
            seen += bsz
            step += 1

        train_loss = running / max(1, seen)
        val_loss = evaluate_loader(train_model, val_loader, device, args.k, amp, blend_lambda=args.blend_lambda)

        if scheduler:
            if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        curr_lr = optimizer.param_groups[0]["lr"]
        bad_epochs = ""
        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            bad_epochs = f" bad={scheduler.num_bad_epochs}/{args.lr_patience}"

        td_loss = 0.0
        td_msg = "off"
        if td_games and args.tdleaf_lambda > 0.0:
            td_loss = run_tdleaf_epoch(
                train_model,
                td_games,
                optimizer,
                scaler,
                device,
                args.k,
                args.tdleaf_lambda,
                amp,
                batch_size=args.tdleaf_batch_size,
            )
            val_loss = evaluate_loader(train_model, val_loader, device, args.k, amp, blend_lambda=args.blend_lambda)
            td_msg = f"{td_loss:.7f}"
        elif args.tdleaf_jsonl and args.tdleaf_lambda <= 0.0:
            td_msg = "disabled(lambda<=0)"

        print(
            f"epoch={epoch} lr={curr_lr:.2e}{bad_epochs} train_mse={train_loss:.7f} val_mse={val_loss:.7f} "
            f"tdleaf_mse={td_msg}"
        )

        latest = ckpt_dir / "latest.pt"
        save_checkpoint(latest, model, optimizer, scaler, scheduler, epoch, step, best_val, args)
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                ckpt_dir / f"epoch_{epoch:04d}.pt",
                model,
                optimizer,
                scaler,
                scheduler,
                epoch,
                step,
                best_val,
                args,
            )

        export_nnue_binary(model, export_path)
        if val_loss < best_val:
            best_val = val_loss
            export_nnue_binary(model, out_dir / "nnue_best.nnue")

        if (
            args.regen_every_epochs > 0
            and args.regen_games > 0
            and (args.epochs is None or epoch < args.epochs)
            and epochs_since_regen >= args.regen_every_epochs
        ):
            epochs_since_regen = 0
            regen_path = Path(args.regen_out) if args.regen_out else data_path
            regen_jsonl = Path(args.tdleaf_jsonl) if args.tdleaf_jsonl else None
            
            print(
                f"regen: epoch={epoch} games={args.regen_games} depth={args.regen_depth} "
                f"mode={args.regen_mode} out={regen_path}"
            )
            data_path = regenerate_data(
                project_dir=Path(__file__).resolve().parents[1],
                base_data_path=regen_path,
                games=args.regen_games,
                depth=args.regen_depth,
                mode=args.regen_mode,
                max_rows=args.regen_max_rows,
                jsonl_path=regen_jsonl,
                use_fsf=args.regen_use_fsf,
                fsf_args=args,
                nnue_out_dir=out_dir,
            )
            rows = load_supervised_rows(data_path, augment=args.augment)
            if not rows:
                raise RuntimeError(f"after regeneration, no rows loaded from {data_path}")
            train_loader, val_loader, train_size, val_size = build_loaders(rows, args, amp)

            # Reset torch.compile caches — CUDA graphs are shape-specific
            if compiled_model is not None:
                torch._dynamo.reset()
                _suppress_torch_compile_warnings()  # dynamo.reset() can reinit torch logging
                compiled_model = torch.compile(model, mode="max-autotune", dynamic=True)
                train_model = compiled_model
                print("regen: torch.compile reset and recompiled")

            if args.tdleaf_jsonl:
                td_games = load_td_games(Path(args.tdleaf_jsonl))
                print(f"regen: reloaded {len(td_games)} TD-leaf games")
            
            msg_lr = ""
            if args.lr_reset_on_regen:
                # Reset LR to initial value
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr
                # Reset best_val so scheduler starts fresh on new data distribution
                best_val = float('inf')
                # Re-initialize scheduler to clear internal bad_epochs counter
                if args.lr_scheduler == "step":
                    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
                elif args.lr_scheduler == "plateau":
                    scheduler = lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        patience=args.lr_patience,
                        factor=args.lr_gamma,
                        threshold=args.lr_threshold,
                        cooldown=args.lr_cooldown,
                    )
                msg_lr = f" lr reset to {args.lr:.2e}"

            print(
                f"regen complete: rows={len(rows)} train={train_size} val={val_size}{msg_lr}"
            )
        epoch += 1
        epochs_since_regen += 1

    print(f"done. latest={export_path} best={out_dir / 'nnue_best.nnue'}")


if __name__ == "__main__":
    main()
