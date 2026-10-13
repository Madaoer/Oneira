"""Single image -> condition video and prompt, ready for oneira.inference.

    python -m oneira.image_to_cond --image room.png --spec room.yaml --output my_samples/room

Perception (LingBot-Map points, GeoCalib gravity, SAM3 masks, MoGe-2 camera height) gives the ground plane and
the object boxes in metres. One of two fixed camera motions is planned over the 243 frames, the world-state table
is rendered by oneira.render, and the folder gets first_frame.png, cond.mp4 and prompt.json (plus world_state.json,
scene.json and camera.json for inspection). Perception results are cached in <output>/perception/.
"""

import argparse
import json
import os

import cv2
import numpy as np
import yaml
from PIL import Image

from . import hub
from .render import UP, box_corners, render

NF, FPS, W, H = 243, 24, 832, 480
CHUNK_STARTS = [0, 29, 60, 91, 121, 152, 183, 214]
GROUND_WORDS = ["ground", "road", "steps", "floor"]

LINGBOT_MAP = "hf://robbyant/lingbot-map/lingbot-map.pt"
GEOCALIB = "pinhole"
SAM3 = "facebook/sam3"
MOGE = "Ruicheng/moge-2-vitl-normal"

CONF_MIN = 1.0
ANCHOR_MIN_PTS = 2000
GROUND_INLIER_M = 0.10
GROUND_MAX_TILT_DEG = 15.0
WALKABLE_CELL_M = 0.20
DP_TOLERANCE_M = 0.15
WALKABLE_MIN_AREA_M2 = 1.0
MASK_MIN_PX = 400
MASK_EDGE_PAD = 4
SIZE_FIX_RANGE = (0.35, 3.0)
SIZE_FIX_ITERS = 3
V_MIN, V_MAX, MAX_TURN_DPS = 0.45, 2.0, 40.0
EX = np.array([1.0, 0.0, 0.0])
EY = np.cross(UP, EX)
STATE_RGB = {"default": (127, 119, 221), "active": (255, 255, 15), "third": (40, 120, 255)}

# action -> (chunk sentence, table entry kind, state before, state after)
ACTIONS = {
    "pick": ("A hand picks up {n}. {N} is lifted off the table and carried.", "vanish", "default", "default"),
    "smash": (
        "A hand smashes {n} with a hammer. {N} is shattered into pieces and the pieces stay scattered.",
        "vanish",
        "default",
        "default",
    ),
    "shoot": (
        "A hand raises a pistol and fires one shot at {n}. {N} is knocked off its place by a bullet and ends up "
        "lying a stride away, leaving the spot where it sat empty.",
        "vanish",
        "default",
        "default",
    ),
    "open": ("A hand opens {n}. {N} ends up fully open and stays open.", "gradient", "default", "active"),
    "close": ("A hand closes {n}. {N} ends up fully closed and stays closed.", "gradient_rev", "active", "default"),
    "fire": (
        "A hand casts a jet of fire at {n}. {N} catches fire and burns steadily from then on.",
        "gradient",
        "default",
        "active",
    ),
    "freeze": (
        "A hand casts a jet of frost at {n}. {N} is coated in white frost and ice and stays frozen over.",
        "gradient",
        "default",
        "active",
    ),
    "ice_out": (
        "A hand casts a jet of frost at {n}. The fire on {n} goes out and {n} is left blackened and frost-covered.",
        "gradient",
        "active",
        "third",
    ),
    "invisible": (
        "A hand casts an invisibility spell at {n}. {N} fades away until it is invisible, and stays invisible.",
        "gradient",
        "default",
        "active",
    ),
    "gild": (
        "A hand casts a gilding spell at {n}. {N} turns into solid gold and stays gold.",
        "gradient",
        "default",
        "active",
    ),
}

VIDEO1_DEF = (
    "<Video 1> is a schematic 3D annotation of the same place: a smooth colour-gradient backdrop encodes the camera's "
    "heading (its hue shifts as the camera turns), a cyan wireframe lattice of cubes marks 3D space, yellow-green grid "
    "lines mark the walkable ground, and together they encode the camera path through the scene. It carries no "
    "appearance information. Solid coloured blocks mark where the marked objects are and how big they are: people, "
    "vehicles, or the target object that will be acted on. A block's colour encodes which of its states the object is "
    "in: purple means its ordinary default state (a person or vehicle simply present, or a target object as it is "
    "normally found), yellow means the first non-default state the object takes on in this clip, and blue means a "
    "second, later non-default state. An object may already be in a non-default state when the clip begins, in which "
    "case its block is yellow from the start. The colours mark only the order of the states, not what the object "
    "looks like - what each state actually is is stated in the text of the time segment in which the change happens. "
    "A block moves with its object, a gradual colour transition marks the moment a change is under way, and a block "
    "may disappear when its object leaves the scene, is taken into the hand, or stops being a single object at all "
    "because it has been broken apart. The blocks are positional markers, not objects to be drawn."
)
VIDEO1_RETENTION = (
    "<Video 1> (camera movement and object placement structure): weak_reference - only the camera trajectory and the "
    "marked positions and states of the objects are followed. No wireframe lattice, no colour-gradient backdrop, no "
    "grid lines and no flat-shaded blocks appear anywhere in the target video. Each coloured block is replaced by the "
    "corresponding photorealistic object in the state its colour indicates, at the same position and occupying the "
    "same fraction of the frame, consistent with <Subject 1>."
)


# ---------------------------------------------------------------- perception (cached in <output>/perception)


def run_lingbot_map(frame, out, ckpt):
    """Point map, confidence and intrinsics of the single frame (LingBot-Map streaming mode)."""
    import torch
    from lingbot_map.models.gct_stream import GCTStream
    from lingbot_map.utils.load_fn import load_and_preprocess_images
    from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri

    model = GCTStream(
        img_size=518,
        patch_size=14,
        enable_3d_rope=True,
        max_frame_num=1024,
        kv_cache_sliding_window=64,
        kv_cache_scale_frames=8,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=True,
        camera_num_iterations=4,
    )
    sd = torch.load(hub.resolve(ckpt), map_location="cuda", weights_only=False)
    model.load_state_dict(sd.get("model", sd), strict=False)
    model = model.to("cuda").eval()
    images = load_and_preprocess_images([frame], mode="crop", image_size=518, patch_size=14).to("cuda")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        preds = model.inference_streaming(images, num_scale_frames=8, keyframe_interval=1, output_device=None)
    extri, intri = pose_encoding_to_extri_intri(preds["pose_enc"], images.shape[-2:])
    c2w = np.tile(np.eye(4), (1, 1, 1))
    c2w[:, :3, :4] = np.asarray(extri.squeeze(0).float().cpu())
    get = lambda k: np.asarray(preds[k].squeeze(0).float().cpu())
    K = np.asarray(intri.squeeze(0).float().cpu())
    depth = get("depth")
    if "world_points" in preds:
        pts = get("world_points")
    else:  # the released head predicts depth only: unproject with K and pose
        d = depth[..., 0] if depth.ndim == 4 else depth
        T, Hp, Wp = d.shape
        uu, vv = np.meshgrid(np.arange(Wp), np.arange(Hp))
        pix = np.stack([uu, vv, np.ones_like(uu)]).reshape(3, -1)
        pts = np.empty((T, Hp, Wp, 3), np.float32)
        for t in range(T):
            cam = (np.linalg.inv(K[t]) @ pix) * d[t].reshape(1, -1)
            pts[t] = ((c2w[t, :3, :3] @ cam).T + c2w[t, :3, 3]).reshape(Hp, Wp, 3)
    np.savez(out, c2w=c2w, K=K, world_points=pts, depth=depth, depth_conf=get("depth_conf"))
    del model
    torch.cuda.empty_cache()


def run_geocalib(frame, out, c2w0, weights):
    """Gravity rotation (camera 0 -> +y up), focal, roll and pitch."""
    import torch
    from geocalib import GeoCalib

    gc = GeoCalib(weights).to("cuda")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    res = gc.calibrate(gc.load_image(frame).to("cuda"))
    up = res["gravity"].vec3d.detach().cpu().numpy().ravel().astype(np.float64)
    up /= np.linalg.norm(up)
    w = c2w0[:3, :3] @ up
    w /= np.linalg.norm(w)
    json.dump(
        {
            "world_rot_to_gravity": gravity_rotation(w).tolist(),
            "geocalib": {
                "f": float(res["camera"].f.detach().cpu().numpy().ravel()[0]),
                "roll_deg": float(res["gravity"].roll),
                "pitch_deg": float(res["gravity"].pitch),
            },
        },
        open(out, "w"),
        indent=1,
    )
    del gc
    torch.cuda.empty_cache()


def run_sam3(frame, out_dir, concepts, model_id):
    """One SAM3 text-prompted session per concept; masks (n_obj, 1, H, W) and scores (n_obj, 1) per concept."""
    import torch
    from transformers import Sam3VideoModel, Sam3VideoProcessor

    model = Sam3VideoModel.from_pretrained(model_id).to("cuda", dtype=torch.bfloat16)
    processor = Sam3VideoProcessor.from_pretrained(model_id)
    video = np.array(Image.open(frame).convert("RGB"))[None]
    for concept in concepts:
        session = processor.init_video_session(
            video=video,
            inference_device="cuda",
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=torch.bfloat16,
        )
        session = processor.add_text_prompt(inference_session=session, text=concept)
        per_frame = {
            o.frame_idx: processor.postprocess_outputs(session, o)
            for o in model.propagate_in_video_iterator(inference_session=session, max_frame_num_to_track=1)
        }
        ids = sorted({int(i) for o in per_frame.values() for i in o["object_ids"].tolist()})
        row = {g: r for r, g in enumerate(ids)}
        masks = np.zeros((len(ids), 1, H, W), bool)
        scores = np.full((len(ids), 1), np.nan, np.float16)
        for f, o in per_frame.items():
            m = o["masks"].cpu().numpy() if hasattr(o["masks"], "cpu") else np.asarray(o["masks"])
            s = o["scores"].tolist() if hasattr(o["scores"], "tolist") else list(o["scores"])
            for k, g in enumerate(o["object_ids"].tolist()):
                mk = m[k][0] if m[k].ndim == 3 else m[k]
                masks[row[int(g)], f] = mk > 0.5
                scores[row[int(g)], f] = s[k]
        np.savez_compressed(
            f"{out_dir}/sam3_{concept.replace(' ', '_')}.npz", masks=masks, object_ids=np.array(ids), scores=scores
        )
    del model
    torch.cuda.empty_cache()


def run_moge(frame, out, f_px, model_id):
    """Metric depth with the LingBot-Map field of view, used only for the camera height."""
    import torch
    from moge.model import import_model_class_by_version

    model = import_model_class_by_version("v2").from_pretrained(model_id).to("cuda").eval()
    rgb = np.array(Image.open(frame).convert("RGB"))
    with torch.no_grad():
        o = model.infer(
            torch.tensor(rgb / 255.0, dtype=torch.float32, device="cuda").permute(2, 0, 1),
            fov_x=float(np.degrees(2 * np.arctan(W / 2 / f_px))),
            resolution_level=9,
            use_fp16=False,
        )
    np.savez(out, depth=o["depth"].cpu().numpy(), mask=o["mask"].cpu().numpy(), f=f_px)
    del model
    torch.cuda.empty_cache()


def perceive(frame, pdir, words, args):
    os.makedirs(pdir, exist_ok=True)
    if not os.path.isfile(f"{pdir}/lbm.npz"):
        run_lingbot_map(frame, f"{pdir}/lbm.npz", args.lingbot_map)
    L = np.load(f"{pdir}/lbm.npz")
    if not os.path.isfile(f"{pdir}/geocalib.json"):
        run_geocalib(frame, f"{pdir}/geocalib.json", L["c2w"][0].astype(np.float64), args.geocalib)
    concepts = list(GROUND_WORDS) + [w for w in dict.fromkeys(split_word(w)[0] for w in words) if w not in GROUND_WORDS]
    todo = [c for c in concepts if not os.path.isfile(f"{pdir}/sam3_{c.replace(' ', '_')}.npz")]
    if todo:
        run_sam3(frame, pdir, todo, args.sam3)
    f832 = float(L["K"][0][0, 0]) * W / L["world_points"].shape[2]
    if not os.path.isfile(f"{pdir}/moge.npz"):
        run_moge(frame, f"{pdir}/moge.npz", f832, args.moge)


# ---------------------------------------------------------------- geometry


def rodrigues(axis, th):
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def gravity_rotation(up_w):
    """Rotation taking the up vector onto +y."""
    up_w = up_w / np.linalg.norm(up_w)
    tgt = np.array([0.0, 1.0, 0.0])
    v = np.cross(up_w, tgt)
    s, c = np.linalg.norm(v), float(up_w @ tgt)
    if s < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)


def split_word(w):
    """'word#k' -> (word, k): the k-th instance from the left; plain 'word' -> (word, None)."""
    if "#" in w:
        b, r = w.rsplit("#", 1)
        return b, int(r)
    return w, None


def load_masks(pdir, word):
    z = np.load(f"{pdir}/sam3_{word.replace(' ', '_')}.npz")
    return z["masks"], z["scores"]


def rank_instances(masks):
    order = []
    for i in range(masks.shape[0]):
        m = masks[i, 0]
        if m.any():
            order.append((float(np.where(m)[1].mean()), i))
    return [i for _, i in sorted(order)]


def pick_instance(masks, scores):
    """Instance present in the most frames, mean score as tie-break."""
    if masks.shape[0] == 0:
        return None
    presence = masks.any(axis=(2, 3)).sum(axis=1).astype(np.float64)
    mean_sc = np.nan_to_num(np.nanmean(scores.astype(np.float32), axis=1), nan=0.0)
    return int((presence + 0.001 * mean_sc).argmax())


def ground_mask(pdir):
    m = np.zeros((H, W), bool)
    for w in GROUND_WORDS:
        masks, _ = load_masks(pdir, w)
        if masks.shape[0]:
            m |= masks[:, 0].any(0)
    return m


def fit_plane_ransac(P, thresh, iters=600, up=None, max_tilt=None, rng=None):
    rng = rng or np.random.default_rng(0)
    if len(P) < 3:
        return None
    best = (None, None, np.zeros(len(P), bool))
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n)
        if ln < 1e-9:
            continue
        n = n / ln
        if up is not None and max_tilt is not None and np.degrees(np.arccos(min(1.0, abs(n @ up)))) > max_tilt:
            continue
        d = -(n @ a)
        inl = np.abs(P @ n + d) < thresh
        if inl.sum() > best[2].sum():
            best = (n, d, inl)
    n, d, inl = best
    if n is None or inl.sum() < 3:
        return None
    Q = P[inl]
    c = Q.mean(0)
    X = Q - c
    n = np.linalg.eigh(X.T @ X)[1][:, 0]
    n /= np.linalg.norm(n)
    d = -(n @ c)
    return n, float(d), np.abs(P @ n + d) < thresh


def moge_camera_height(pdir, Rg):
    """Median height of the SAM3 ground pixels below the camera in MoGe-2's metric depth."""
    g = np.load(f"{pdir}/moge.npz")
    D, M, f = g["depth"], g["mask"].astype(bool), float(g["f"])
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    P = np.stack([(u - W / 2) / f * D, (v - H / 2) / f * D, D], -1)
    gm = ground_mask(pdir)
    return float(-np.median((P[gm & M & np.isfinite(D)] @ Rg.T)[:, 1]))


def fit_ground(pdir):
    """Metric scale from the predicted camera height, then the RANSAC ground plane in the gravity frame."""
    L = np.load(f"{pdir}/lbm.npz")
    g = json.load(open(f"{pdir}/geocalib.json"))
    Rg = np.asarray(g["world_rot_to_gravity"], float)
    Wp, Hp = L["world_points"].shape[2], L["world_points"].shape[1]
    conf = L["depth_conf"][0]
    mr = cv2.resize(ground_mask(pdir).astype(np.uint8), (Wp, Hp), interpolation=cv2.INTER_NEAREST).astype(bool)
    P = L["world_points"][0][mr & (conf > CONF_MIN)] @ Rg.T
    assert len(P) >= ANCHOR_MIN_PTS, f"too few ground points ({len(P)} < {ANCHOR_MIN_PTS})"
    h_lbm = float(-np.median(P[:, 1]))
    assert h_lbm > 0, "the ground is above the camera"
    cam_h = moge_camera_height(pdir, Rg)
    s_u = h_lbm / cam_h
    fit = fit_plane_ransac(P / s_u, GROUND_INLIER_M, up=UP, max_tilt=GROUND_MAX_TILT_DEG)
    assert fit is not None, "ground RANSAC failed"
    n, d, inl = fit
    if n[1] < 0:
        n, d = -n, -d
    K = L["K"][0].astype(float).copy()
    K[0] *= W / Wp
    K[1] *= H / Hp
    n_cam0 = Rg.T @ n
    return dict(
        Rg=Rg,
        s_u=float(s_u),
        n_g=np.asarray(n, float),
        d=float(d),
        n=n_cam0 / np.linalg.norm(n_cam0),
        K=K,
        f=float(K[0, 0]),
        W=W,
        H=H,
        moge_cam_h=cam_h,
        tilt_deg=float(np.degrees(np.arccos(min(1.0, n[1])))),
        inlier=float(inl.mean()),
        geom={"roll_deg": g["geocalib"]["roll_deg"], "pitch_deg": g["geocalib"]["pitch_deg"]},
    )


def proj_wh(center, size, yaw, Rw, tw, K):
    """Projected 2D box (width, height) in pixels; None if a corner is behind the camera."""
    fx, fy, cx, cy, sx, sy = K
    corners, _, _ = box_corners(center, size, float(yaw), UP, EX, EY)
    uv = []
    for w in corners:
        p = Rw.T @ (np.asarray(w, float) - tw)
        if p[2] <= 1e-6:
            return None
        uv.append(((fx * p[0] / p[2] + cx) * sx, (fy * p[1] / p[2] + cy) * sy))
    uv = np.array(uv)
    return uv.max(0)[0] - uv.min(0)[0], uv.max(0)[1] - uv.min(0)[1]


def fit_box(pdir, m, G):
    """Mask x point map -> gravity-aligned box in metres (PCA yaw, 5-95 percentile extents)."""
    L = np.load(f"{pdir}/lbm.npz")
    pts, conf = L["world_points"][0], L["depth_conf"][0]
    Hp, Wp = pts.shape[:2]
    mr = cv2.resize(m.astype(np.uint8), (Wp, Hp), interpolation=cv2.INTER_NEAREST).astype(bool)
    sel = mr & (conf > np.percentile(conf, 30))
    if sel.sum() < 40:
        return None
    pm = (pts[sel] @ G["Rg"].T) / G["s_u"]
    keep = np.linalg.norm(pm - np.median(pm, axis=0), axis=1) < 1.0
    if keep.sum() < 30:
        return None
    pm = pm[keep]
    xz = pm[:, [0, 2]] - np.median(pm[:, [0, 2]], axis=0)
    ax = np.linalg.eigh(np.cov(xz.T))[1][:, -1]
    yaw = float(np.arctan2(-ax[1], ax[0]))
    c, si = np.cos(yaw), np.sin(yaw)
    u, t = xz @ np.array([c, -si]), xz @ np.array([si, c])
    ys = pm[:, 1]
    size = [
        float(np.percentile(u, 95) - np.percentile(u, 5)),
        float(np.percentile(t, 95) - np.percentile(t, 5)),
        float(np.percentile(ys, 95) - np.percentile(ys, 5)),
    ]
    center = [float(np.median(pm[:, 0])), float(np.median(ys)), float(np.median(pm[:, 2]))]
    return {"center": center, "size": [max(s, 0.06) for s in size], "yaw": yaw}


def refine_size(box, m, G, pdir):
    """Scale the box so that its projection matches the mask (vertical scale may only grow)."""
    if m.sum() < MASK_MIN_PX:
        return box, False
    L = np.load(f"{pdir}/lbm.npz")
    K0 = L["K"][0]
    fx, fy, cx, cy = float(K0[0, 0]), float(K0[1, 1]), float(K0[0, 2]), float(K0[1, 2])
    K = (fx, fy, cx, cy, W / (cx * 2), H / (cy * 2))
    ys, xs = np.where(m)
    u0, u1, v0, v1 = xs.min(), xs.max(), ys.min(), ys.max()
    ok_w = u0 > MASK_EDGE_PAD and u1 < W - 1 - MASK_EDGE_PAD
    ok_h = v0 > MASK_EDGE_PAD and v1 < H - 1 - MASK_EDGE_PAD
    if not (ok_w or ok_h):
        return box, False
    Rw, tw = G["Rg"] @ L["c2w"][0][:3, :3], (G["Rg"] @ L["c2w"][0][:3, 3]) / G["s_u"]
    lo, hi = SIZE_FIX_RANGE
    box = dict(box, size=list(box["size"]))
    acc_w = acc_h = 1.0
    for _ in range(SIZE_FIX_ITERS):
        wh = proj_wh(box["center"], box["size"], box["yaw"], Rw, tw, K)
        ok = wh is not None and wh[0] >= 1 and wh[1] >= 1
        step_w = (u1 - u0) / wh[0] if ok and ok_w else 1.0
        step_h = (v1 - v0) / wh[1] if ok and ok_h else 1.0
        step_h = max(step_h, 1.0 / acc_h)
        step_w = min(max(acc_w * step_w, lo), hi) / acc_w
        step_h = min(max(acc_h * step_h, lo), hi) / acc_h
        acc_w, acc_h = acc_w * step_w, acc_h * step_h
        sz = box["size"]
        box["size"] = [sz[0] * step_w, sz[1] * step_w, sz[2] * step_h]
        if abs(step_w - 1) < 0.02 and abs(step_h - 1) < 0.02:
            break
    return box, True


def object_box(pdir, word, G):
    """SAM3 instance of `word` -> box (camera-0 centre, size, yaw) or None."""
    base, rank = split_word(word)
    masks, scores = load_masks(pdir, base)
    if rank is not None:
        order = rank_instances(masks) if masks.shape[0] else []
        if rank >= len(order):
            return None
        inst = order[rank]
    else:
        inst = pick_instance(masks, scores)
    if inst is None or not masks[inst, 0].any():
        return None
    m = masks[inst, 0]
    box = fit_box(pdir, m, G)
    if box is None:
        return None
    box, _ = refine_size(box, m, G, pdir)
    c_g = np.asarray(box["center"], float)
    return dict(
        center_cam0=(G["Rg"].T @ c_g),
        size=[float(x) for x in box["size"]],
        yaw=float(box["yaw"]),
        dist=float(np.linalg.norm(c_g)),
        height=float(G["n_g"] @ c_g + G["d"]),
    )


def place_objects(spec, pdir, G):
    """Boxes for every object in the spec; optional class size prior and heading along the walking direction."""
    pos, objs = {}, {}
    for k, o in spec["objects"].items():
        b = object_box(pdir, o["word"], G)
        assert b is not None, f"no reliable box for {k!r} ({o['word']!r}); try another word or image"
        p, size, yaw = b["center_cam0"], b["size"], b["yaw"]
        if o.get("along_path"):
            fwd_g = G["Rg"] @ np.array([0.0, 0.0, 1.0])
            yaw = float(np.arctan2(-fwd_g[2], fwd_g[0]))
        if o.get("size"):
            sp = [float(x) for x in o["size"]]
            ray = p / max(float(np.linalg.norm(p)), 1e-9)
            r_g = G["Rg"] @ ray
            r_h = np.array([r_g[0], -r_g[2]])
            r_h /= max(float(np.linalg.norm(r_h)), 1e-9)
            ax_ = np.array([np.cos(yaw), np.sin(yaw)])
            cph = abs(float(ax_ @ r_h))
            sph = float(np.sqrt(max(1.0 - cph * cph, 0.0)))
            ext = lambda sz: sz[0] * cph + sz[1] * sph
            p = p + ray * (ext(sp) - ext(size)) / 2
            size = sp
        pos[k] = p
        objs[k] = dict(cls=o.get("class", split_word(o["word"])[0]), size=size, yaw=yaw)
        print(
            f"[box] {k}: {b['dist']:.2f} m away, centre {b['height']:.2f} m above the ground, size "
            f"{np.round(size, 2).tolist()} m",
            flush=True,
        )
    return pos, objs


# ---------------------------------------------------------------- camera motion


def plane_basis(n):
    fwd3 = np.array([0, 0, 1.0]) - float(np.array([0, 0, 1.0]) @ n) * n
    fwd3 /= np.linalg.norm(fwd3)
    cross3 = np.cross(n, fwd3)
    cross3 /= np.linalg.norm(cross3)
    return fwd3, cross3


def look_rot(f_t, n):
    z = f_t / np.linalg.norm(f_t)
    x = np.cross(z, n)
    x /= max(np.linalg.norm(x), 1e-9)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def smooth_path(wps, n, corner_r=0.9, m=4000):
    P = np.stack([np.asarray(w, float) for w in wps])
    P = P - (P @ n)[:, None] * n[None, :]
    seg0 = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    ss = np.linspace(0, seg0[-1], m)
    pts = np.stack([np.interp(ss, seg0, P[:, k]) for k in range(3)], 1)
    w = max(3, int(round(corner_r / (seg0[-1] / len(ss)))) | 1)
    pad = np.vstack([np.repeat(pts[:1], w, 0), pts, np.repeat(pts[-1:], w, 0)])
    ker = np.ones(w) / w
    pts = np.stack([np.convolve(pad[:, k], ker, "same")[w:-w] for k in range(3)], 1)
    seg = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    return pts, seg


def walk(wps, n, speed, look_at, look_hold, max_turn_dps, yaw_amp_deg=4.0, yaw_period_s=5.5, look_win_s=2.0):
    """Walk along waypoints at per-frame speed while looking at a point (yaw and pitch rate-limited)."""
    pts, seg = smooth_path(wps, n, corner_r=0.2)
    L = float(seg[-1])
    arc = np.r_[0.0, np.cumsum(speed[:-1] / FPS)]
    u0 = pts[1] - pts[0]
    u0 /= np.linalg.norm(u0)
    f0v = np.array([0.0, 0.0, 1.0])
    ref0 = f0v - float(f0v @ n) * n
    ref0 /= max(np.linalg.norm(ref0), 1e-9)
    c2w = np.tile(np.eye(4), (NF, 1, 1))
    for i in range(NF):
        t_s, s_i = i / FPS, float(arc[i])
        if s_i <= L:
            j = int(np.searchsorted(seg, s_i).clip(1, len(pts) - 1))
            pos, tan = pts[j], pts[j] - pts[j - 1]
        else:
            tan = pts[-1] - pts[-2]
            pos = pts[-1] + (s_i - L) * tan / np.linalg.norm(tan)
        tan = tan - float(tan @ n) * n
        tan /= max(np.linalg.norm(tan), 1e-9)
        yaw = float(np.arctan2(float(np.cross(ref0, tan) @ n), float(tan @ ref0)))
        yaw += np.deg2rad(yaw_amp_deg) * np.sin(2 * np.pi * t_s / yaw_period_s)
        f0, f1 = look_hold
        ramp_f = max(look_win_s * FPS, 1.0)
        look_w = min(1.0, (i - f0) / ramp_f + 1.0) if i < f0 else 1.0 if i <= f1 else max(0.0, 1.0 - (i - f1) / ramp_f)
        look_w = max(0.0, min(1.0, look_w if i >= f0 - ramp_f else 0.0))
        lat = np.cross(n, u0)
        sw = 0.010 * np.sin(2 * np.pi * t_s / 6.5) * lat + 0.008 * np.sin(2 * np.pi * t_s / 4.7) * n
        ramp = min(1.0, i / 12.0)
        pitch = 0.0
        if look_w > 0:
            to = np.asarray(look_at, float) - pos
            to_h = to - float(to @ n) * n
            lh = float(np.linalg.norm(to_h))
            behind = float(to_h @ tan) < 0 and lh < 1.0
            if lh > 0.40 and not behind:
                yaw_t = float(np.arctan2(float(np.cross(ref0, to_h) @ n), float(to_h @ ref0)))
                yaw += look_w * ((yaw_t - yaw + np.pi) % (2 * np.pi) - np.pi)
            if lh > 1e-3:
                pitch = look_w * float(np.arctan2(-float(to @ n), max(lh, 0.25)))
        f0h = f0v - float(f0v @ n) * n
        f0h /= max(np.linalg.norm(f0h), 1e-9)
        pitch0 = float(np.arctan2(-float(f0v @ n), max(float(np.linalg.norm(f0v - float(f0v @ n) * n)), 1e-9)))
        f_path = rodrigues(n, yaw * ramp) @ f0h
        p_total = pitch0 * (1.0 - ramp) + pitch * ramp
        if abs(p_total) > 1e-6:
            right = np.cross(f_path, n)
            right /= max(np.linalg.norm(right), 1e-9)
            f_path = rodrigues(right, -p_total) @ f_path
        if i > 0:
            f_prev = c2w[i - 1, :3, :3] @ np.array([0.0, 0.0, 1.0])
            f_now = f_path / max(np.linalg.norm(f_path), 1e-9)
            lim = np.deg2rad(max_turn_dps) / FPS

            def yp(f):
                h = f - float(f @ n) * n
                lh_ = max(float(np.linalg.norm(h)), 1e-9)
                return (
                    float(np.arctan2(float(np.cross(u0, h / lh_) @ n), float((h / lh_) @ u0))),
                    float(np.arctan2(float(f @ n), lh_)),
                )

            y0_, p0_ = yp(f_prev)
            y1_, p1_ = yp(f_now)
            dy = (y1_ - y0_ + np.pi) % (2 * np.pi) - np.pi
            dp = p1_ - p0_
            if abs(dy) > lim or abs(dp) > lim:
                y_ = y0_ + float(np.clip(dy, -lim, lim))
                p_ = p0_ + float(np.clip(dp, -lim, lim))
                f_path = np.cos(p_) * (rodrigues(n, y_) @ u0) + np.sin(p_) * n
        c2w[i, :3, :3] = look_rot(f_path, n)
        c2w[i, :3, 3] = pos + sw * ramp
    return c2w


def interact_path(G, target, stop_distance, t_arrive):
    """Walk straight up to `stop_distance` (horizontal metres) from the target by frame t_arrive, then stay,
    looking at it throughout."""
    n = G["n"]
    to_h = target - float(target @ n) * n
    D = float(np.linalg.norm(to_h))
    assert D > stop_distance, f"the target is {D:.2f} m away, closer than stop_distance {stop_distance}"
    u = to_h / D
    goal = u * (D - stop_distance)
    wps = [np.zeros(3), u * (D - stop_distance) * 0.5, goal]
    v_app = float(np.clip((D - stop_distance) / (t_arrive / FPS), V_MIN, V_MAX))
    speed = np.zeros(NF)
    speed[:t_arrive] = v_app
    speed[t_arrive:] = 0.03
    print(
        f"[camera] walk {D - stop_distance:.2f} m at {v_app:.2f} m/s, stop {stop_distance} m from the target",
        flush=True,
    )
    return walk(wps, n, speed, target, (0, NF - 1), MAX_TURN_DPS)


def navigate_path(G, speed=1.0, left=1.0, turn=-12.0, ramp_s=1.0, drift_from=121, yaw_amp_deg=4.0, yaw_period_s=5.5):
    """Walk forward the whole clip at `speed` m/s; from `drift_from` also step `left` metres to the left and turn
    `turn` degrees (negative = right), both eased in and reaching their value at the last frame."""
    left_m, turn_deg = left, turn
    n = G["n"]
    f0v = np.array([0.0, 0.0, 1.0])
    f0h = f0v - float(f0v @ n) * n
    pitch0 = float(np.arctan2(-float(f0v @ n), max(float(np.linalg.norm(f0h)), 1e-9)))
    f0h /= max(np.linalg.norm(f0h), 1e-9)
    lat = np.cross(n, f0h)
    c2w = np.tile(np.eye(4), (NF, 1, 1))
    span = max(NF - 1 - drift_from, 1)
    for i in range(NF):
        t_s, ramp = i / FPS, min(1.0, i / 12.0)
        s = (
            speed * (t_s - ramp_s / 2)
            if t_s >= ramp_s
            else speed * (t_s / 2 - ramp_s / (2 * np.pi) * np.sin(np.pi * t_s / ramp_s))
        )
        e = 0.5 - 0.5 * np.cos(np.pi * min(max(i - drift_from, 0), span) / span)
        yaw = np.deg2rad(turn_deg) * e + np.deg2rad(yaw_amp_deg) * np.sin(2 * np.pi * t_s / yaw_period_s) * ramp
        f = rodrigues(n, yaw) @ f0h
        p = pitch0 * (1.0 - ramp)
        if abs(p) > 1e-6:
            right = np.cross(f, n)
            right /= max(np.linalg.norm(right), 1e-9)
            f = rodrigues(right, -p) @ f
        sw = 0.010 * np.sin(2 * np.pi * t_s / 6.5) * lat + 0.008 * np.sin(2 * np.pi * t_s / 4.7) * n
        c2w[i, :3, :3] = look_rot(f, n)
        c2w[i, :3, 3] = s * f0h + left_m * e * lat + sw * ramp
    print(
        f"[camera] walk forward at {speed} m/s; from frame {drift_from} step {left_m} m left and turn {turn_deg} deg",
        flush=True,
    )
    return c2w


# ---------------------------------------------------------------- world-state table and scene


def bev_raster(xz, cell, pad_m=1.0, max_side=4000):
    lo = xz.min(0) - pad_m
    hi = xz.max(0) + pad_m
    size = np.maximum(np.ceil((hi - lo) / cell).astype(int), 3)
    if size.max() > max_side:
        return None, None
    img = np.zeros((size[1], size[0]), np.uint8)
    ij = np.floor((xz - lo) / cell).astype(int)
    img[np.clip(ij[:, 1], 0, size[1] - 1), np.clip(ij[:, 0], 0, size[0] - 1)] = 255
    return img, lo


def walkable_footprint(seed_xz, cam_xz, corridor_m):
    """Ground footprint seen in the image, plus a corridor along the part of the camera path it does not cover."""
    img, lo = bev_raster(np.vstack([seed_xz, cam_xz]), WALKABLE_CELL_M)
    assert img is not None, "walkable area too large to rasterise"

    def cells(q):
        c = np.floor((np.atleast_2d(q) - lo) / WALKABLE_CELL_M).astype(int)
        c[:, 0] = np.clip(c[:, 0], 0, img.shape[1] - 1)
        c[:, 1] = np.clip(c[:, 1], 0, img.shape[0] - 1)
        return c

    seed_cells = cells(seed_xz)
    obs = np.zeros_like(img)
    oc = cells(seed_xz)
    obs[oc[:, 1], oc[:, 0]] = 255
    obs = cv2.morphologyEx(obs, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    cc = cells(cam_xz)
    gap = obs[cc[:, 1], cc[:, 0]] == 0
    if gap.any():
        track = np.zeros_like(img)
        idx = np.flatnonzero(gap)
        for grp in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            a, b = max(0, grp[0] - 1), min(len(cc), grp[-1] + 2)
            seg = cc[a:b]
            if len(seg) >= 2:
                cv2.polylines(track, [seg.reshape(-1, 1, 2)], False, 255, 1)
            else:
                cv2.circle(track, tuple(seg[0]), 1, 255, -1)
        r = max(1, int(round(corridor_m / WALKABLE_CELL_M)))
        track = cv2.dilate(track, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        img = np.maximum(img, track)
        seed_cells = np.vstack([seed_cells, cc[gap]])
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    _, lab = cv2.connectedComponents(img)
    keep = set(np.unique(lab[seed_cells[:, 1], seed_cells[:, 0]]).tolist()) - {0}
    mask = np.isin(lab, list(keep)).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polys = []
    for c in sorted(cnts, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(c) * WALKABLE_CELL_M**2 < WALKABLE_MIN_AREA_M2:
            continue
        p = cv2.approxPolyDP(c, DP_TOLERANCE_M / WALKABLE_CELL_M, True).reshape(-1, 2) * WALKABLE_CELL_M + lo
        if len(p) >= 3:
            polys.append(p.tolist())
    assert polys, "no walkable polygon"
    return polys[0]


def walkable_polygon(pdir, G, c2w_g, band_m=1.5, extend_m=8.0):
    """Observed floor footprint, with a corridor where the camera leaves it and extended along the last heading."""
    cam_xz = c2w_g[:, [0, 2], 3]
    u = (c2w_g[-1, :3, :3] @ np.array([0, 0, 1.0]))[[0, 2]]
    u = u / max(np.linalg.norm(u), 1e-9)
    cam_xz = np.vstack([cam_xz, cam_xz[-1] + u[None, :] * np.arange(0.25, extend_m + 1e-6, 0.25)[:, None]])
    L = np.load(f"{pdir}/lbm.npz")
    Wp, Hp = L["world_points"].shape[2], L["world_points"].shape[1]
    mr = cv2.resize(ground_mask(pdir).astype(np.uint8), (Wp, Hp), interpolation=cv2.INTER_NEAREST).astype(bool)
    Pg = (L["world_points"][0][mr & (L["depth_conf"][0] > CONF_MIN)] / G["s_u"]) @ G["Rg"].T
    n_g = G["Rg"] @ G["n"]
    n_g = n_g / np.linalg.norm(n_g)
    seed_xz = Pg[np.abs(Pg @ n_g + G["d"]) < GROUND_INLIER_M][:, [0, 2]]
    assert len(seed_xz) >= ANCHOR_MIN_PTS, f"too few floor inliers ({len(seed_xz)})"
    return walkable_footprint(seed_xz, cam_xz, band_m)


def color_at(spec, f):
    s0, s1 = spec["s0"], spec["s1"]
    if "t0" not in spec or s0 == s1 or f < spec["t0"]:
        return s0
    if f >= spec["t1"]:
        return s1
    w = (f - spec["t0"] + 1) / (spec["t1"] - spec["t0"])
    return tuple(int(round((1 - w) * x + w * y)) for x, y in zip(STATE_RGB[s0], STATE_RGB[s1]))


def world_state(objs, pos, Rg, action=None, t0=91, t1=121):
    """Table of object boxes and their state over time. The acted-on object changes over chunk 4 (frames t0..t1);
    every other object stays in its default state."""
    table = []
    for tid, (k, o) in enumerate(objs.items(), 1):
        ent = dict(
            track_id=tid,
            cls=o["cls"],
            size=[float(x) for x in o["size"]],
            center=[float(x) for x in (Rg @ pos[k])],
            yaw=float(o["yaw"]),
        )
        if action and k == action["object"]:
            _, kind, s0, s1 = ACTIONS[action["type"]]
            if kind == "vanish":
                ent.update(kind=kind, exist=[0, t0], color={"s0": "default", "s1": "default"})
            else:
                ent.update(kind=kind, exist=[0, NF], color={"s0": s0, "s1": s1, "t0": t0, "t1": t1})
        else:
            ent.update(kind="gradient", exist=[0, NF], color={"s0": "default", "s1": "default"})
        table.append(ent)
    return table


def boxes_dynamic(table, motions):
    """Per-frame boxes for the renderer; objects with a motion drive along their heading (eased-in speed)."""
    out = []
    for o in table:
        m = motions.get(o["cls"])
        frames = {}
        for f in range(int(o["exist"][0]), min(int(o["exist"][1]), NF)):
            c = list(o["center"])
            if m:
                v, a = float(m["speed"]), float(m.get("accel", 1.0))
                t, ta = f / FPS, v / a
                s = 0.5 * a * t * t if t < ta else 0.5 * a * ta * ta + v * (t - ta)
                c = [c[0] + s * np.cos(o["yaw"]), c[1], c[2] - s * np.sin(o["yaw"])]
            frames[str(f)] = {"center": c, "size": o["size"], "yaw": float(o["yaw"]), "state": color_at(o["color"], f)}
        if frames:
            out.append(
                {"track_id": o["track_id"], "class": o["cls"], "kind": o["kind"], "is_tool": False, "frames": frames}
            )
    return out


def scene_dict(G, poly, boxes, path_dir):
    fwd3 = np.asarray(path_dir, float)
    fwd3 = fwd3 - float(fwd3[1]) * np.array([0.0, 1.0, 0.0])
    fwd3 /= np.linalg.norm(fwd3)
    return {
        "version": "representation_v2",
        "units": "meters",
        "intrinsics": {"fx": G["f"], "fy": G["f"], "cx": W / 2, "cy": H / 2, "w": W, "h": H},
        "camera_height_m": G["d"],
        "ground": [
            {
                "normal": [0.0, 1.0, 0.0],
                "d": G["d"],
                "walkable_polygon_xz": poly,
                "walkable_polygons_xz": [poly],
                "path_dir_xz": [float(fwd3[0]), float(fwd3[2])],
                "is_ramp": False,
                "inlier_frac": 1.0,
                "tilt_deg": 0.0,
                "confident": True,
                "adjacent": [],
            }
        ],
        "walls": [],
        "boxes": [],
        "boxes_dynamic": boxes,
        "path_ribbon": None,
        "gravity": (-np.array([0.0, 1.0, 0.0])).tolist(),
        "world_rot_to_gravity": np.eye(3).tolist(),
    }


def check_in_view(c2w_g, K, center_g, frames):
    """The acted-on object must be in the middle of the frame when its state changes."""
    for fi in frames:
        w2c = np.linalg.inv(c2w_g[fi])
        p = w2c[:3, :3] @ center_g + w2c[:3, 3]
        u, v = (K @ p)[:2] / (K @ p)[2]
        assert p[2] > 0 and abs(u - W / 2) <= 0.30 * W and abs(v - H / 2) <= 0.35 * H, (
            f"the target is not in the middle of the frame at frame {fi} (pixel {u:.0f}, {v:.0f}); change stop_distance"
        )


# ---------------------------------------------------------------- prompt


def global_caption(scene, objs_text):
    def lc(t):
        t = str(t).strip().rstrip(".")
        if t[:2].isalpha() and t[:2].isupper():
            return t
        return t[0].lower() + t[1:] if t else t

    def sent(t):
        t = str(t).strip()
        if not t:
            return ""
        t = t[0].upper() + t[1:]
        return t if t.endswith((".", "!", "?")) else t + "."

    mats = (
        ", ".join(scene["materials"]) if isinstance(scene.get("materials"), list) else str(scene.get("materials", ""))
    )
    return "\n".join(
        [
            "subject_definitions:",
            f"<Subject 1> is the {lc(scene['type'])} in <Picture 1>: a real-world location under "
            f"{lc(scene['time_weather'])}, with its specific lighting, colours and materials ({mats}), and its overall "
            f"spatial layout. Resting in this location: {objs_text}.",
            VIDEO1_DEF,
            "",
            "summary:",
            f"[reference generation] The target video is photorealistic live-action footage of <Subject 1>, with the camera "
            f"moving through the scene along the trajectory encoded by <Video 1>, and {objs_text} at the positions marked "
            f"in <Video 1>.",
            "",
            "retention_analysis:",
            "<Subject 1> (appears in [Shot 1]): fully_preserved - the photographic look, lighting, colours, materials and "
            "spatial layout of the location are retained throughout.",
            VIDEO1_RETENTION,
            "",
            "detailed_description:",
            f"The target video is photorealistic live-action footage. {sent(scene['lighting'])}",
            f"[Shot 1] {sent(scene['description'])}",
            "",
            "overall_soundscape:",
            "N/A - the target video has no soundtrack.",
            "",
            "non_diegetic_music:",
            "N/A",
        ]
    )


def chunk_captions(action, objs):
    caps = ["Nothing happens."] * len(CHUNK_STARTS)
    if action:
        name = objs[action["object"]]["cls"]
        caps[2] = ACTIONS[action["type"]][0].format(n=f"the {name}", N=f"The {name}")
    return [{"k": k + 1, "caption": c} for k, c in enumerate(caps)]


# ---------------------------------------------------------------- entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument(
        "--spec", required=True, help="yaml: objects, motion, action, scene (see Image to Condition Video in README.md)"
    )
    ap.add_argument("--output", required=True, help="sample folder to write")
    ap.add_argument("--lingbot-map", default=LINGBOT_MAP)
    ap.add_argument("--geocalib", default=GEOCALIB, help="checkpoint path, or pinhole to download it")
    ap.add_argument("--sam3", default=SAM3)
    ap.add_argument("--moge", default=MOGE)
    args = ap.parse_args()
    spec = yaml.safe_load(open(args.spec))
    out = args.output
    os.makedirs(out, exist_ok=True)
    frame = f"{out}/first_frame.png"
    im = Image.open(args.image).convert("RGB")
    (im if im.size == (W, H) else im.resize((W, H), Image.LANCZOS)).save(frame)

    pdir = f"{out}/perception"
    perceive(frame, pdir, [o["word"] for o in spec["objects"].values()], args)
    G = fit_ground(pdir)
    print(
        f"[ground] camera height {G['moge_cam_h']:.2f} m (MoGe-2), plane fit {G['d']:.2f} m, tilt "
        f"{G['tilt_deg']:.1f} deg, field of view {np.degrees(2 * np.arctan(W / 2 / G['f'])):.1f} deg",
        flush=True,
    )
    pos, objs = place_objects(spec, pdir, G)

    action = spec.get("action")
    if spec.get("motion", "interact") == "interact":
        assert action and action["type"] in ACTIONS, f"interact needs action.type in {sorted(ACTIONS)}"
        stop = float(spec.get("stop_distance", 0.9))
        p = pos[action["object"]]
        D = float(np.linalg.norm(p - float(p @ G["n"]) * G["n"]))
        t_arrive = int(np.clip(round(FPS * (D - stop) / 1.1), 1, 80))
        c2w = interact_path(G, p, stop, t_arrive)
    else:
        action = None
        c2w = navigate_path(G, **spec.get("navigate", {}))
    T4 = np.eye(4)
    T4[:3, :3] = G["Rg"]
    c2w_g = np.einsum("ij,njk->nik", T4, c2w)

    table = world_state(objs, pos, G["Rg"], action)
    if action:
        check_in_view(c2w_g, G["K"], G["Rg"] @ pos[action["object"]], (CHUNK_STARTS[3] - 1, CHUNK_STARTS[3]))
    motions = {o.get("class", split_word(o["word"])[0]): o["drive"] for o in spec["objects"].values() if o.get("drive")}
    poly = walkable_polygon(pdir, G, c2w_g)
    scene = scene_dict(G, poly, boxes_dynamic(table, motions), c2w_g[0, :3, :3] @ np.array([0, 0, 1.0]))
    json.dump(table, open(f"{out}/world_state.json", "w"), indent=1)
    json.dump(scene, open(f"{out}/scene.json", "w"), indent=1)
    json.dump({"K": G["K"].tolist(), "c2w": c2w_g.tolist()}, open(f"{out}/camera.json", "w"))
    render(json.load(open(f"{out}/scene.json")), c2w_g, G["K"], f"{out}/cond.mp4")

    if spec.get("global_caption"):
        cap = spec["global_caption"]
    else:
        objs_text = spec["scene"].get("objects") or " and ".join(f"the {o['cls']}" for o in objs.values())
        cap = global_caption(spec["scene"], objs_text)
    json.dump(
        {"global_caption": cap, "chunks": chunk_captions(action, objs)},
        open(f"{out}/prompt.json", "w"),
        indent=1,
        ensure_ascii=False,
    )
    print(f"[done] {out}: first_frame.png, cond.mp4, prompt.json", flush=True)


if __name__ == "__main__":
    main()
