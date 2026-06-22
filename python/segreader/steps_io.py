"""Save pipeline debug steps (PNGs + steps.json) to disk.

Shared by main.py and evaluate.py so neither needs to spawn a subprocess
just to write processed-step output.
"""
import base64
import json
import os
from pathlib import Path

import cv2


def image_to_b64(img) -> str:
    """Encode a BGR/grayscale NumPy image as a base64 PNG string."""
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode()


def save_processed_steps(
    steps: list,
    image_path: str,
    result: str,
    params: dict,
    base_dir: str = "data/processed",
) -> str:
    """Save each pipeline step as a PNG and write a steps.json summary.

    Returns the relative path to the written steps.json file.
    """
    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = os.path.join(base_dir, stem)
    os.makedirs(out_dir, exist_ok=True)

    json_steps = []
    for i, step in enumerate(steps):
        if "image" in step and step["image"] is not None:
            cv2.imwrite(os.path.join(out_dir, f"{i:02d}_{step['id']}.png"), step["image"])

        s = {k: v for k, v in step.items() if k not in ("image", "digits")}
        s["image_b64"] = image_to_b64(step["image"]) if step.get("image") is not None else None

        if "digits" in step:
            serialised_digits = []
            for d in step["digits"]:
                cv2.imwrite(os.path.join(out_dir, f"{i:02d}_digit{d['index']}_patch.png"), d["patch"])
                cv2.imwrite(os.path.join(out_dir, f"{i:02d}_digit{d['index']}_zones.png"), d["zone_vis"])
                sd = {k: v for k, v in d.items() if k not in ("patch", "zone_vis")}
                sd["patch_b64"] = image_to_b64(d["patch"])
                sd["zone_vis_b64"] = image_to_b64(d["zone_vis"])
                serialised_digits.append(sd)
            s["digits"] = serialised_digits

        json_steps.append(s)

    json_path = os.path.join(out_dir, "steps.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"image_path": image_path, "result": result, "params": params, "steps": json_steps},
            f,
            indent=2,
        )

    return Path(json_path).as_posix()
