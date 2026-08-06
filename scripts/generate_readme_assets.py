"""Generate deterministic README assets from real KOVARA-9 evidence.

The demo frame and animation use the bundled frontier walkthrough seed. The policy
comparison reads the frozen v0.1.0 result JSON without modifying it. The social preview
is explanatory project branding and contains no empirical claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from kovara9.agents.frontier import FrontierPolicy
from kovara9.config.loader import load_bundled_demo_config
from kovara9.config.models import DemoConfig, DemoEpisodeConfig
from kovara9.core.types import Position, WorldSnapshot
from kovara9.evaluation.records import EpisodeRecord
from kovara9.evaluation.runner import run_episode

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR: Final = REPOSITORY_ROOT / "docs" / "assets"
FINAL_RESULTS_PATH: Final = REPOSITORY_ROOT / "docs" / "day8-final-heldout-results.json"
DEMO_CONFIG_PATH: Final = (
    REPOSITORY_ROOT / "src" / "kovara9" / "resources" / "open_source_demo.yaml"
)

Color = tuple[int, int, int]

PALETTE: Final[tuple[Color, ...]] = (
    (7, 17, 31),  # 0 background
    (13, 27, 42),  # 1 panel
    (41, 68, 95),  # 2 grid
    (244, 247, 251),  # 3 foreground
    (159, 179, 200),  # 4 muted
    (77, 208, 225),  # 5 cyan
    (79, 140, 255),  # 6 blue
    (255, 183, 77),  # 7 amber
    (239, 108, 117),  # 8 coral
    (103, 211, 145),  # 9 green
    (217, 226, 232),  # 10 floor
    (38, 55, 70),  # 11 obstacle
    (242, 95, 92),  # 12 target
    (85, 199, 122),  # 13 recovered
    (50, 92, 132),  # 14 blue shadow
    (132, 91, 34),  # 15 amber shadow
)

# A compact, original bitmap alphabet keeps generation dependency-free and deterministic.
FONT: Final[dict[str, tuple[str, ...]]] = {
    " ": ("00000",) * 7,
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "J": ("00111", "00010", "00010", "00010", "10010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    "/": ("00001", "00010", "00010", "00100", "01000", "01000", "10000"),
    ":": ("00000", "00100", "00100", "00000", "00100", "00100", "00000"),
    ".": ("00000", "00000", "00000", "00000", "00000", "00110", "00110"),
    "%": ("11001", "11010", "00100", "01000", "10110", "00110", "00000"),
    "(": ("00010", "00100", "01000", "01000", "01000", "00100", "00010"),
    ")": ("01000", "00100", "00010", "00010", "00010", "00100", "01000"),
    "=": ("00000", "11111", "00000", "11111", "00000", "00000", "00000"),
}


@dataclass
class IndexedCanvas:
    """Small indexed-color raster canvas."""

    width: int
    height: int
    pixels: bytearray

    @classmethod
    def create(cls, width: int, height: int, color: int = 0) -> IndexedCanvas:
        return cls(width=width, height=height, pixels=bytearray([color]) * (width * height))

    def set_pixel(self, x: int, y: int, color: int) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            self.pixels[y * self.width + x] = color

    def rectangle(self, x: int, y: int, width: int, height: int, color: int) -> None:
        left = max(0, x)
        top = max(0, y)
        right = min(self.width, x + width)
        bottom = min(self.height, y + height)
        if left >= right or top >= bottom:
            return
        row = bytes([color]) * (right - left)
        for row_y in range(top, bottom):
            offset = row_y * self.width + left
            self.pixels[offset : offset + len(row)] = row

    def line(self, x0: int, y0: int, x1: int, y1: int, color: int) -> None:
        delta_x = abs(x1 - x0)
        step_x = 1 if x0 < x1 else -1
        delta_y = -abs(y1 - y0)
        step_y = 1 if y0 < y1 else -1
        error = delta_x + delta_y
        while True:
            self.set_pixel(x0, y0, color)
            if x0 == x1 and y0 == y1:
                break
            twice_error = 2 * error
            if twice_error >= delta_y:
                error += delta_y
                x0 += step_x
            if twice_error <= delta_x:
                error += delta_x
                y0 += step_y

    def text(self, value: str, x: int, y: int, color: int, scale: int = 1) -> None:
        cursor_x = x
        for character in value.upper():
            glyph = FONT.get(character, FONT[" "])
            for row_index, row in enumerate(glyph):
                for column_index, enabled in enumerate(row):
                    if enabled == "1":
                        self.rectangle(
                            cursor_x + column_index * scale,
                            y + row_index * scale,
                            scale,
                            scale,
                            color,
                        )
            cursor_x += 6 * scale


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload))
    )


def write_png(path: Path, canvas: IndexedCanvas) -> None:
    """Write an indexed canvas as a deterministic RGB PNG."""

    raw_rows = bytearray()
    for y in range(canvas.height):
        raw_rows.append(0)
        row_offset = y * canvas.width
        for x in range(canvas.width):
            red, green, blue = PALETTE[canvas.pixels[row_offset + x]]
            raw_rows.extend((red, green, blue))
    header = struct.pack(">IIBBBBB", canvas.width, canvas.height, 8, 2, 0, 0, 0)
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(bytes(raw_rows), level=9))
        + _chunk(b"IEND", b"")
    )
    path.write_bytes(payload)


def _pack_codes(codes: list[tuple[int, int]]) -> bytes:
    output = bytearray()
    accumulator = 0
    bit_count = 0
    for code, width in codes:
        accumulator |= code << bit_count
        bit_count += width
        while bit_count >= 8:
            output.append(accumulator & 0xFF)
            accumulator >>= 8
            bit_count -= 8
    if bit_count:
        output.append(accumulator & 0xFF)
    return bytes(output)


def _lzw_encode(indices: bytes, minimum_code_size: int = 4) -> bytes:
    """Encode GIF image data with a deterministic LZW dictionary."""

    clear_code = 1 << minimum_code_size
    end_code = clear_code + 1
    dictionary = {bytes([index]): index for index in range(clear_code)}
    next_code = end_code + 1
    code_width = minimum_code_size + 1
    codes: list[tuple[int, int]] = [(clear_code, code_width)]
    if not indices:
        codes.append((end_code, code_width))
        return _pack_codes(codes)

    sequence = bytes([indices[0]])
    for index in indices[1:]:
        candidate = sequence + bytes([index])
        if candidate in dictionary:
            sequence = candidate
            continue
        codes.append((dictionary[sequence], code_width))
        if next_code < 4096:
            dictionary[candidate] = next_code
            next_code += 1
            if next_code == 1 << code_width and code_width < 12:
                code_width += 1
        else:
            codes.append((clear_code, code_width))
            dictionary = {bytes([value]): value for value in range(clear_code)}
            next_code = end_code + 1
            code_width = minimum_code_size + 1
        sequence = bytes([index])
    codes.append((dictionary[sequence], code_width))
    codes.append((end_code, code_width))
    return _pack_codes(codes)


def write_gif(path: Path, frames: list[IndexedCanvas], delay_cs: int = 35) -> None:
    """Write indexed canvases as a looping GIF89a animation."""

    if not frames:
        raise ValueError("At least one GIF frame is required.")
    width = frames[0].width
    height = frames[0].height
    if any(frame.width != width or frame.height != height for frame in frames):
        raise ValueError("All GIF frames must use the same dimensions.")

    palette_bytes = bytearray()
    for red, green, blue in PALETTE:
        palette_bytes.extend((red, green, blue))
    data = bytearray(b"GIF89a")
    data.extend(struct.pack("<HH", width, height))
    data.extend((0xF3, 0x00, 0x00))  # Global 16-color table, background index 0.
    data.extend(palette_bytes)
    data.extend(b"\x21\xff\x0bNETSCAPE2.0\x03\x01\x00\x00\x00")

    for frame in frames:
        data.extend(b"\x21\xf9\x04\x00")
        data.extend(struct.pack("<H", delay_cs))
        data.extend(b"\x00\x00")
        data.extend(b"\x2c\x00\x00\x00\x00")
        data.extend(struct.pack("<HH", width, height))
        data.append(0x00)
        data.append(0x04)
        compressed = _lzw_encode(bytes(frame.pixels))
        for offset in range(0, len(compressed), 255):
            block = compressed[offset : offset + 255]
            data.append(len(block))
            data.extend(block)
        data.append(0x00)
    data.append(0x3B)
    path.write_bytes(bytes(data))


def _load_frontier_demo() -> tuple[
    DemoConfig, DemoEpisodeConfig, EpisodeRecord, list[WorldSnapshot]
]:
    config = load_bundled_demo_config()
    policy_config = next(episode for episode in config.episodes if episode.policy == "frontier")
    if policy_config.seed not in config.seed_partitions.train.resolved_seeds:
        raise ValueError("README demo seed must belong to the declared training partition.")
    if policy_config.seed in config.seed_partitions.test.resolved_seeds:
        raise ValueError("README assets may not use a locked final-test seed.")

    snapshots: list[WorldSnapshot] = []
    record = run_episode(
        env_config=config.environment,
        seed=policy_config.seed,
        policy_factory=FrontierPolicy,
        snapshot_observer=snapshots.append,
    )
    return config, policy_config, record, snapshots


def _draw_demo_snapshot(
    snapshot: WorldSnapshot,
    *,
    policy_config: DemoEpisodeConfig,
    frame_number: int,
) -> IndexedCanvas:
    canvas = IndexedCanvas.create(480, 352)
    canvas.rectangle(16, 16, 448, 320, 1)
    canvas.text("REAL SIMULATOR WALKTHROUGH", 32, 30, 3, 2)
    canvas.text(f"FRONTIER  SEED {policy_config.seed}", 32, 50, 5, 1)

    grid_x = 32
    grid_y = 76
    cell_size = 30
    for y in range(snapshot.height):
        for x in range(snapshot.width):
            position = Position(row=y, col=x)
            color = 11 if snapshot.obstacles[y, x] else 10
            if position in snapshot.recovered_targets:
                color = 13
            elif position in snapshot.targets:
                color = 12
            left = grid_x + x * cell_size
            top = grid_y + y * cell_size
            canvas.rectangle(left, top, cell_size - 1, cell_size - 1, color)

    for agent_index, (_agent_id, position) in enumerate(sorted(snapshot.agent_positions.items())):
        left = grid_x + position.col * cell_size
        top = grid_y + position.row * cell_size
        color = 6 if agent_index % 2 == 0 else 7
        shadow = 14 if agent_index % 2 == 0 else 15
        canvas.rectangle(left + 5, top + 5, cell_size - 10, cell_size - 10, shadow)
        canvas.rectangle(left + 8, top + 8, cell_size - 16, cell_size - 16, color)
        canvas.text(str(agent_index + 1), left + 12, top + 11, 3, 1)

    panel_x = 292
    canvas.text("EXAMPLE RUN", panel_x, 82, 3, 1)
    canvas.text("NOT A BENCHMARK", panel_x, 98, 8, 1)
    canvas.text(f"STEP {snapshot.step_count}", panel_x, 128, 4, 1)
    recovered = len(snapshot.recovered_targets)
    total = len(snapshot.targets)
    canvas.text(f"RECOVERED {recovered}/{total}", panel_x, 144, 4, 1)
    canvas.text(f"FRAME {frame_number + 1}", panel_x, 160, 4, 1)

    canvas.rectangle(panel_x, 194, 10, 10, 6)
    canvas.text("AGENT 1", panel_x + 18, 196, 4, 1)
    canvas.rectangle(panel_x, 214, 10, 10, 7)
    canvas.text("AGENT 2", panel_x + 18, 216, 4, 1)
    canvas.rectangle(panel_x, 234, 10, 10, 12)
    canvas.text("TARGET", panel_x + 18, 236, 4, 1)
    canvas.rectangle(panel_x, 254, 10, 10, 13)
    canvas.text("RECOVERED", panel_x + 18, 256, 4, 1)
    canvas.text("SAFE DEMO SEED", panel_x, 290, 5, 1)
    canvas.text("NO TRAINING", panel_x, 306, 5, 1)
    return canvas


def _draw_policy_comparison(results: dict[str, object]) -> tuple[IndexedCanvas, dict[str, float]]:
    classification = str(results["classification"])
    if classification != "exploration_transfer_without_task_completion":
        raise ValueError(f"Unexpected frozen-result classification: {classification}")

    policy_results = results["policy_results"]
    metrics = {
        "random": _nested_float(policy_results, "random", "pooled", "success_rate"),
        "frontier": _nested_float(policy_results, "frontier", "pooled", "success_rate"),
        "untrained_mean": sum(
            _nested_float(policy_results, f"untrained_seed_{seed}", "pooled", "success_rate")
            for seed in range(3)
        )
        / 3,
        "trained_mean": sum(
            _nested_float(policy_results, f"trained_seed_{seed}", "pooled", "success_rate")
            for seed in range(3)
        )
        / 3,
    }
    canvas = IndexedCanvas.create(960, 540)
    canvas.rectangle(32, 28, 896, 484, 1)
    canvas.text("POLICY SUCCESS RATE", 64, 58, 3, 4)
    canvas.text("FROZEN V0.1 HELD-OUT EVALUATION", 64, 96, 5, 2)
    canvas.text("PUBLISHED RESULTS - NOT DEMO OUTPUT", 64, 118, 8, 2)

    rows = (
        ("RANDOM (200)", metrics["random"], 4),
        ("FRONTIER (200)", metrics["frontier"], 9),
        ("UNTRAINED MEAN (3 X 200)", metrics["untrained_mean"], 7),
        ("TRAINED MEAN (3 X 200)", metrics["trained_mean"], 6),
    )
    chart_x = 330
    chart_width = 520
    for row_index, (label, value, color) in enumerate(rows):
        y = 170 + row_index * 68
        canvas.text(label, 64, y + 8, 3, 2)
        canvas.rectangle(chart_x, y, chart_width, 30, 2)
        bar_width = round(chart_width * value)
        if bar_width > 0:
            canvas.rectangle(chart_x, y, bar_width, 30, color)
        else:
            canvas.rectangle(chart_x, y, 3, 30, color)
        canvas.text(f"{value * 100:.1f}%", 868, y + 8, 3, 2)

    canvas.text("EXPLORATION TRANSFER WITHOUT FULL TASK COMPLETION", 64, 448, 3, 2)
    canvas.text("SOURCE: DOCS/DAY8-FINAL-HELDOUT-RESULTS.JSON", 64, 480, 4, 1)
    return canvas, metrics


def _nested_float(root: object, *keys: str) -> float:
    current = root
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise TypeError(f"Frozen result path is missing or invalid: {'/'.join(keys)}")
        current = current[key]
    if not isinstance(current, int | float):
        raise TypeError(f"Frozen result value is not numeric: {'/'.join(keys)}")
    return float(current)


def _draw_social_preview() -> IndexedCanvas:
    canvas = IndexedCanvas.create(1280, 640)
    canvas.rectangle(52, 52, 1176, 536, 1)

    # Explanatory multi-agent grid/network motif; it is branding, not result data.
    grid_x = 760
    grid_y = 126
    cell = 54
    for offset in range(8):
        canvas.line(grid_x, grid_y + offset * cell, grid_x + 7 * cell, grid_y + offset * cell, 2)
        canvas.line(grid_x + offset * cell, grid_y, grid_x + offset * cell, grid_y + 7 * cell, 2)
    nodes = ((1, 1, 6), (5, 2, 7), (3, 5, 5), (6, 6, 9))
    centers: list[tuple[int, int]] = []
    for x, y, color in nodes:
        center_x = grid_x + x * cell
        center_y = grid_y + y * cell
        centers.append((center_x, center_y))
        canvas.rectangle(center_x - 12, center_y - 12, 25, 25, color)
        canvas.rectangle(center_x - 5, center_y - 5, 11, 11, 3)
    canvas.line(*centers[0], *centers[2], 5)
    canvas.line(*centers[1], *centers[2], 5)
    canvas.line(*centers[2], *centers[3], 5)

    canvas.text("KOVARA-9", 104, 148, 3, 10)
    canvas.rectangle(106, 246, 460, 8, 5)
    canvas.text("REPRODUCIBLE MULTI-AGENT RL", 108, 292, 5, 3)
    canvas.text("FOR COOPERATIVE EXPLORATION", 108, 330, 3, 3)
    canvas.text("IN PROCEDURAL ENVIRONMENTS", 108, 368, 3, 3)
    canvas.text("RESEARCH ENGINEERING / OPEN SOURCE", 108, 446, 4, 2)
    return canvas


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_assets(output_dir: Path) -> dict[str, object]:
    """Generate all raster README assets and return their provenance manifest."""

    output_dir.mkdir(parents=True, exist_ok=True)
    config, policy_config, record, snapshots = _load_frontier_demo()
    if not snapshots:
        raise RuntimeError("The frontier walkthrough produced no simulator snapshots.")

    demo_frames = [
        _draw_demo_snapshot(snapshot, policy_config=policy_config, frame_number=index)
        for index, snapshot in enumerate(snapshots)
    ]
    frame_path = output_dir / "demo-frame.png"
    gif_path = output_dir / "demo-frontier.gif"
    write_png(frame_path, demo_frames[-1])
    write_gif(gif_path, demo_frames)

    frozen_results = json.loads(FINAL_RESULTS_PATH.read_text(encoding="utf-8"))
    if not isinstance(frozen_results, dict):
        raise TypeError("Frozen final results must be a JSON object.")
    comparison, metrics = _draw_policy_comparison(frozen_results)
    comparison_path = output_dir / "policy-comparison.png"
    write_png(comparison_path, comparison)

    social_path = output_dir / "kovara9-social-preview.png"
    write_png(social_path, _draw_social_preview())

    generated = (frame_path, gif_path, comparison_path, social_path)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "generator": "scripts/generate_readme_assets.py",
        "sources": {
            "demo_config": str(DEMO_CONFIG_PATH.relative_to(REPOSITORY_ROOT)).replace("\\", "/"),
            "demo_config_sha256": _sha256(DEMO_CONFIG_PATH),
            "frozen_results": str(FINAL_RESULTS_PATH.relative_to(REPOSITORY_ROOT)).replace(
                "\\", "/"
            ),
            "frozen_results_sha256": _sha256(FINAL_RESULTS_PATH),
        },
        "demo": {
            "policy": policy_config.policy,
            "label": policy_config.name,
            "seed": policy_config.seed,
            "seed_partition": "train/development",
            "final_test_seed_used": False,
            "steps": record.episode_length,
            "targets_recovered": record.targets_recovered,
            "target_count": config.environment.num_targets,
            "success": record.success,
            "frames": len(snapshots),
            "purpose": "behavioral walkthrough; not a benchmark estimate",
        },
        "comparison": {
            "classification": frozen_results["classification"],
            "success_rates": metrics,
            "purpose": "derived visualization of frozen v0.1.0 held-out results",
        },
        "assets": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)} for path in generated
        },
    }
    manifest_path = output_dir / "readme-assets-manifest.json"
   manifest_bytes = (
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
).encode("utf-8")
manifest_path.write_bytes(manifest_bytes)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate deterministic README visuals from real KOVARA-9 evidence."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination directory (default: docs/assets).",
    )
    arguments = parser.parse_args()
    manifest = generate_assets(arguments.output_dir.resolve())
    sys.stdout.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
