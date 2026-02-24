from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any

from PIL import Image, ImageDraw, ImageFont


# -------------------------
# Fixed object library
# -------------------------

@dataclass
class ObjSpec:
    name: str
    color: str
    shape: str  # square | rectangle | circle | triangle


@dataclass
class ObjInstance:
    spec: ObjSpec
    bbox: Tuple[int, int, int, int]  # x1,y1,x2,y2
    center: Tuple[int, int]


OBJ_SPECS: Dict[str, ObjSpec] = {
    "book": ObjSpec(name="book", color="red", shape="square"),
    "bottle": ObjSpec(name="bottle", color="blue", shape="rectangle"),
    "cat": ObjSpec(name="cat", color="white", shape="circle"),
    "arrow": ObjSpec(name="arrow", color="black", shape="triangle"),
}

COLORS = {
    "red": (220, 40, 40),
    "blue": (60, 120, 220),
    "white": (240, 240, 240),
    "black": (30, 30, 30),
}

CANVAS = (256, 256)


def _nonoverlap_bbox(
    rng: random.Random,
    existing: List[Tuple[int, int, int, int]],
    w_range=(46, 74),
    h_range=(46, 74),
    tries=250,
):
    W, H = CANVAS
    for _ in range(tries):
        w = rng.randint(*w_range)
        h = rng.randint(*h_range)
        x1 = rng.randint(12, W - w - 12)
        y1 = rng.randint(12, H - h - 12)
        x2, y2 = x1 + w, y1 + h
        ok = True
        for (a1, b1, a2, b2) in existing:
            pad = 10
            if not (x2 + pad < a1 or a2 + pad < x1 or y2 + pad < b1 or b2 + pad < y1):
                ok = False
                break
        if ok:
            return (x1, y1, x2, y2)

    # fallback even if overlaps
    w = rng.randint(*w_range)
    h = rng.randint(*h_range)
    x1 = rng.randint(12, W - w - 12)
    y1 = rng.randint(12, H - h - 12)
    return (x1, y1, x1 + w, y1 + h)


def render_scene(objs: List[ObjInstance], seed: int) -> Image.Image:
    # Slightly off-white background so white cat still has visible edges.
    img = Image.new("RGB", CANVAS, (245, 245, 245))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    # Draw each object
    for inst in objs:
        o = inst.spec
        x1, y1, x2, y2 = inst.bbox
        fill = COLORS[o.color]
        outline = (15, 15, 15)

        if o.shape == "circle":
            draw.ellipse([x1, y1, x2, y2], fill=fill, outline=outline, width=3)
        elif o.shape == "square":
            # force square by using min side
            side = min(x2 - x1, y2 - y1)
            x2s, y2s = x1 + side, y1 + side
            draw.rectangle([x1, y1, x2s, y2s], fill=fill, outline=outline, width=3)
        elif o.shape == "rectangle":
            draw.rectangle([x1, y1, x2, y2], fill=fill, outline=outline, width=3)
        elif o.shape == "triangle":
            cx, cy = inst.center
            pts = [(cx, y1), (x1, y2), (x2, y2)]
            draw.polygon(pts, fill=fill, outline=outline)
        else:
            raise ValueError(f"Unknown shape: {o.shape}")

        # tiny label (helps VLM bind object identity)
        draw.text((x1 + 3, y1 + 3), o.name[:4], fill=(10, 10, 10), font=font)

    return img


def make_episode(seed: int) -> Dict[str, Any]:
    """Create one episode with ONLY these 4 possible objects.

    Scene contains a random subset of {book,bottle,cat,arrow}.
    Question is: "Is there a {object}?" where {object} is one of the 4.
    """
    rng = random.Random(seed)

    # Random subset (at least 1 object, up to all 4)
    names = list(OBJ_SPECS.keys())
    k = rng.randint(1, 4)
    present = rng.sample(names, k=k)

    # Create placements
    bboxes: List[Tuple[int, int, int, int]] = []
    objs: List[ObjInstance] = []

    for nm in present:
        spec = OBJ_SPECS[nm]

        if spec.shape == "rectangle":
            bbox = _nonoverlap_bbox(rng, bboxes, w_range=(54, 86), h_range=(38, 62))
        else:
            bbox = _nonoverlap_bbox(rng, bboxes, w_range=(46, 74), h_range=(46, 74))

        bboxes.append(bbox)
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        objs.append(ObjInstance(spec=spec, bbox=bbox, center=(cx, cy)))

    img = render_scene(objs, seed=seed)

    # Target query object (can be present or absent)
    target = rng.choice(names)
    gt_exists = target in present

    question = f"Is there a {target}?"  # as requested

    return {
        "seed": seed,
        "image": img,
        "present": present,
        "target": target,
        "gt_exists": gt_exists,
        "target_color": OBJ_SPECS[target].color,
        "question": question,
    }
