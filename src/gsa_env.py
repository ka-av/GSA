from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any

from PIL import Image, ImageDraw, ImageFont, ImageFilter


# -------------------------
# Scene + Question schema
# -------------------------

@dataclass
class Obj:
    name: str           # e.g. "mug"
    color: str          # e.g. "red"
    shape: str          # "circle"|"square"|"triangle"
    bbox: Tuple[int,int,int,int]  # x1,y1,x2,y2
    center: Tuple[int,int]
    depth: int          # smaller = closer (for our synthetic convention)


@dataclass
class Question:
    qtype: str          # "exist"|"attr"|"spatial"|"geom"
    prompt: str
    gt: Dict[str, Any]  # ground-truth labels
    ambiguous: bool     # whether we *expect* abstain is acceptable


# -------------------------
# Environment generator
# -------------------------

COLORS = {
    "red": (220, 40, 40),
    "green": (40, 170, 70),
    "blue": (60, 120, 220),
    "yellow": (230, 210, 60),
    "black": (30, 30, 30),
    "white": (240, 240, 240),
    "purple": (140, 70, 170),
}

SHAPES = ["circle", "square", "triangle"]
OBJ_NAMES = ["mug", "book", "ball", "phone", "key", "bottle", "shoe", "chair", "cat", "arrow"]

CANVAS = (256, 256)

def _rand_color(rng: random.Random) -> str:
    return rng.choice(list(COLORS.keys()))

def _rand_shape(rng: random.Random) -> str:
    return rng.choice(SHAPES)

def _rand_name(rng: random.Random) -> str:
    return rng.choice(OBJ_NAMES)

def _nonoverlap_bbox(rng: random.Random, existing: List[Tuple[int,int,int,int]], size_range=(40,70), tries=200):
    W, H = CANVAS
    for _ in range(tries):
        w = rng.randint(*size_range)
        h = rng.randint(*size_range)
        x1 = rng.randint(10, W - w - 10)
        y1 = rng.randint(10, H - h - 10)
        x2, y2 = x1 + w, y1 + h
        ok = True
        for (a1,b1,a2,b2) in existing:
            # simple overlap check with padding
            pad = 8
            if not (x2+pad < a1 or a2+pad < x1 or y2+pad < b1 or b2+pad < y1):
                ok = False
                break
        if ok:
            return (x1,y1,x2,y2)
    # fallback even if overlaps
    w = rng.randint(*size_range)
    h = rng.randint(*size_range)
    x1 = rng.randint(10, W - w - 10)
    y1 = rng.randint(10, H - h - 10)
    return (x1,y1,x1+w,y1+h)

def render_scene(objs: List[Obj], add_text: bool, text: str, blur: bool, occlude: bool, seed: int) -> Image.Image:
    img = Image.new("RGB", CANVAS, (245, 245, 245))
    draw = ImageDraw.Draw(img)

    # optional font
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()

    # draw objects from far to near (depth large -> far)
    for o in sorted(objs, key=lambda x: x.depth, reverse=True):
        x1,y1,x2,y2 = o.bbox
        fill = COLORS[o.color]
        outline = (20, 20, 20)

        if o.shape == "circle":
            draw.ellipse([x1,y1,x2,y2], fill=fill, outline=outline, width=3)
        elif o.shape == "square":
            draw.rectangle([x1,y1,x2,y2], fill=fill, outline=outline, width=3)
        else:
            # triangle
            cx, cy = o.center
            pts = [(cx, y1), (x1, y2), (x2, y2)]
            draw.polygon(pts, fill=fill, outline=outline)

        # small label to make attribute binding easier
        draw.text((x1+3, y1+3), o.name[:4], fill=(10,10,10), font=font)

    if add_text:
        draw.text((12, 226), text, fill=(10,10,10), font=font)

    if occlude:
        # cover a random patch (simulate occlusion)
        rng = random.Random(seed + 999)
        ox1 = rng.randint(80, 160)
        oy1 = rng.randint(60, 140)
        ox2 = ox1 + rng.randint(40, 80)
        oy2 = oy1 + rng.randint(30, 70)
        draw.rectangle([ox1,oy1,ox2,oy2], fill=(245,245,245), outline=None)

    if blur:
        img = img.filter(ImageFilter.GaussianBlur(radius=2.0))

    return img

def make_episode(seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)

    # scene params
    n_objs = rng.randint(4, 6)
    add_text = rng.random() < 0.35
    blur = rng.random() < 0.25
    occlude = rng.random() < 0.25

    # create objects
    bboxes = []
    used = set()
    objs: List[Obj] = []
    for i in range(n_objs):
        name = _rand_name(rng)
        while name in used:
            name = _rand_name(rng)
        used.add(name)

        color = _rand_color(rng)
        shape = _rand_shape(rng)
        bbox = _nonoverlap_bbox(rng, bboxes)
        bboxes.append(bbox)
        x1,y1,x2,y2 = bbox
        cx, cy = (x1+x2)//2, (y1+y2)//2
        depth = rng.randint(1, 3)
        objs.append(Obj(name=name, color=color, shape=shape, bbox=bbox, center=(cx,cy), depth=depth))

    # optional text content
    text = rng.choice(["HELLO", "EXIT", "OPEN", "STOP", "GSA"])
    img = render_scene(objs, add_text=add_text, text=text, blur=blur, occlude=occlude, seed=seed)

    # build questions (exist + attr + spatial + geom)
    # Q1: existence of a target object/color pair (sometimes absent)
    present = rng.random() < 0.65
    if present:
        target = rng.choice(objs)
        exist_name = target.name
        exist_color = target.color
        gt_exist = True
        gt_bbox = list(target.bbox)
    else:
        exist_name = rng.choice(OBJ_NAMES)
        exist_color = rng.choice(list(COLORS.keys()))
        # ensure absent by checking
        if any(o.name == exist_name and o.color == exist_color for o in objs):
            # force absent: tweak color
            exist_color = "purple" if exist_color != "purple" else "yellow"
        gt_exist = False
        gt_bbox = None

    q_exist = Question(
        qtype="exist",
        prompt=f"Existence+Box: Is there a {exist_color} {exist_name}? Answer yes/no. If yes, provide bbox [x1,y1,x2,y2].",
        gt={"exists": gt_exist, "name": exist_name, "color": exist_color, "bbox": gt_bbox},
        ambiguous=blur or occlude
    )

    # Q2: attribute binding: color of a named object (guarantee present)
    target2 = rng.choice(objs)
    q_attr = Question(
        qtype="attr",
        prompt=f"Attribute: What is the color of the {target2.name}? (single word)",
        gt={"name": target2.name, "color": target2.color},
        ambiguous=blur or occlude
    )

    # Q3: spatial left/right between two objects (guarantee present)
    a, b = rng.sample(objs, 2)
    gt_left = a.center[0] < b.center[0]
    # Ask: "Is A left of B?"
    q_spatial = Question(
        qtype="spatial",
        prompt=f"Spatial: Is the {a.name} to the LEFT of the {b.name}? Answer yes/no.",
        gt={"a": a.name, "b": b.name, "relation": "left_of", "answer": gt_left},
        ambiguous=blur or occlude
    )

    # Q4: geometry/orientation: arrow direction OR depth ordering
    if any(o.name == "arrow" for o in objs):
        # we won't actually draw arrow direction; keep it simple -> depth ordering instead
        pass
    c, d = rng.sample(objs, 2)
    gt_closer = c.depth < d.depth  # smaller depth = closer
    q_geom = Question(
        qtype="geom",
        prompt=f"Geometry: Which is closer to the camera: {c.name} or {d.name}? Answer with one name.",
        gt={"c": c.name, "d": d.name, "closer": c.name if gt_closer else d.name},
        ambiguous=blur or occlude
    )

    questions = [q_exist, q_attr, q_spatial, q_geom]

    return {
        "seed": seed,
        "image": img,
        "objects": objs,
        "questions": questions,
        "flags": {"blur": blur, "occlude": occlude, "add_text": add_text, "text": text}
    }