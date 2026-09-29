"""CPU-only comparison of native full-context arrays against an MLX oracle.

The 1e-3 absolute / 1e-4 relative-L2 final-context gate was frozen before the
boxing-prompt holdout. This is separate from isolated scale-aware connector gates.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-video", type=Path, required=True)
    parser.add_argument("--native-audio", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"absolute_limit": 1e-3, "relative_l2_limit": 1e-4, "modalities": {}}
    with safe_open(args.reference, framework="numpy") as reference:
        report["reference_token_ids"] = reference.get_tensor("token_ids").tolist()
        for name, source, width in [
            ("video", args.native_video, 4096),
            ("audio", args.native_audio, 2048),
        ]:
            expected = reference.get_tensor(name)
            if expected.shape != (1, 1024, width) or source.stat().st_size != expected.size * 4:
                raise ValueError(f"Native/reference {name} conditioning shape mismatch")
            actual = np.fromfile(source, dtype="<f4").astype(np.float64)
            expected = expected.ravel().astype(np.float64)
            if not np.isfinite(actual).all() or not np.isfinite(expected).all():
                raise ValueError(f"Nonfinite {name} conditioning")
            errors = abs(actual - expected)
            absolute = float(errors.max())
            relative = float(np.linalg.norm(errors) / max(np.linalg.norm(expected), 1e-30))
            report["modalities"][name] = {
                "max_absolute_error": absolute,
                "relative_l2_error": relative,
                "rmse": float(np.sqrt(np.mean(errors * errors))),
                "max_scaled_error": float(np.max(errors / np.maximum(abs(expected), 1))),
                "passed": absolute <= 1e-3 and relative <= 1e-4,
                "historical_strict_absolute_1e_4_passed": absolute <= 1e-4,
            }
    report["passed"] = all(x["passed"] for x in report["modalities"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
