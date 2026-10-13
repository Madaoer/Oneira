"""Condition-video renderer (moderngl, headless EGL): walkable-ground grid, world lattice, heading-coloured
backdrop and state-coloured object boxes."""

from __future__ import annotations

import math
import os
import sys

import cv2
import numpy as np


STATE_RGB = {
    "default": "purple",
    "active": "yellow",
    "third": "state_blue",
}

UP = np.array([0.0, 1.0, 0.0])

RULER_LAT_PITCH_M = 1.5

RULER_LAT_HALF_M = 8.0

RULER_CROSS_SPACING_M = 2.0

RULER_DEPTH_MIN_M = 1.5

RULER_DEPTH_MAX_M = 20.0

RULER_DEPTH_HARD_M = 60.0

RULER_LAT_HARD_M = 30.0

GROUND_ALPHA = 0.12

RULER_FADE_START_M = 15.0

RULER_FADE_FLOOR = 0.35

LATTICE_V_M = 1.5

LATTICE_LEVELS = 2

LATTICE_RADIUS_M = 6.6

LATTICE_FADE_M = 4.6

LATTICE_NODE_K = 0.020

LATTICE_NODE_PX = (1.0, 5.0)

LATTICE_NEAR_M = 0.8

LATTICE_NEAR_FADE_M = 2.0

LATTICE_GROW_M = 4.0


def make_lift(plane):
    """xz (N,2) -> world (N,3) on the ground plane n·P + d = 0."""
    n = np.asarray(plane["normal"], float)
    d = float(plane["d"])
    ny = n[1] if abs(n[1]) > 1e-6 else 1e-6

    def lift(xz):
        q = np.atleast_2d(np.asarray(xz, float))
        y = -(d + n[0] * q[:, 0] + n[2] * q[:, 1]) / ny
        return np.stack([q[:, 0], y, q[:, 1]], 1)

    return lift


def ruler_window(poly, fwd, lat, s_cam, t_cam):
    """Depth/lateral window implied by the walkable polygon."""
    if poly is None or len(poly) < 3:
        return RULER_DEPTH_MAX_M, RULER_DEPTH_MAX_M, RULER_LAT_HALF_M
    P = np.asarray(poly, float)
    s = P @ fwd - s_cam
    t = P @ lat - t_cam
    d1 = float(min(max(np.percentile(s, 98), RULER_DEPTH_MIN_M + 2.0), RULER_DEPTH_HARD_M))
    d0 = float(min(max(np.percentile(-s, 98), RULER_DEPTH_MIN_M + 2.0), RULER_DEPTH_HARD_M))
    half = float(min(max(np.percentile(np.abs(t), 98), 2.0), RULER_LAT_HARD_M))
    return d0, d1, half


def ground_faces(scene):
    """Every (plane, region) worth drawing."""
    out = []
    for g in scene.get("ground", []):
        if g.get("confident") is False:
            continue
        regs = g.get("walkable_polygons_xz") or (
            [g["walkable_polygon_xz"]] if len(g.get("walkable_polygon_xz", [])) >= 3 else []
        )
        for r in regs:
            if len(r) >= 3:
                out.append({**g, "walkable_polygon_xz": r})
    return out


def lattice_rods_nodes(
    plane,
    cam_c,
    grow_m=LATTICE_GROW_M,
    levels=LATTICE_LEVELS,
    radius=LATTICE_RADIUS_M,
    fade=LATTICE_FADE_M,
    v_m=LATTICE_V_M,
    polys=None,
    up=None,
):
    """Ground-ruler crossings extended upward. Returns (rods, nodes)."""
    lift = make_lift(plane)
    n = np.asarray(plane["normal"], float)
    nn = max(np.linalg.norm(n), 1e-9)
    n, d = n / nn, float(plane["d"]) / nn

    if float(n @ np.asarray(cam_c, float) + d) < 0:
        n = -n
    if up is None:
        rod_up = n
    else:
        rod_up = np.asarray(up, float)
        rod_up = rod_up / max(np.linalg.norm(rod_up), 1e-9)
        if rod_up @ n < 0:
            rod_up = -rod_up

        p_anchor = lift(np.array([cam_c[0], cam_c[2]]))[0]
        n = rod_up
        d = float(-(rod_up @ p_anchor))
        lift = make_lift({"normal": rod_up.tolist(), "d": d})
    pd = plane.get("path_dir_xz") or [0.0, 1.0]
    fwd = np.asarray(pd, float)
    if np.linalg.norm(fwd) < 1e-9:
        fwd = np.array([0.0, 1.0])
    fwd = fwd / np.linalg.norm(fwd)
    lat = np.array([-fwd[1], fwd[0]])
    lat_pitch = float(plane.get("ruler_lat_pitch_m", RULER_LAT_PITCH_M))
    cross_sp = float(plane.get("ruler_cross_spacing_m", RULER_CROSS_SPACING_M))

    foot = np.array([cam_c[0], cam_c[2]])
    s_cam, t_cam = float(foot @ fwd), float(foot @ lat)
    src = polys if polys is not None else [plane.get("walkable_polygon_xz") or []]
    cnts = (
        [np.asarray(q, float).astype(np.float32).reshape(-1, 1, 2) for q in src if len(q) >= 3]
        if np.isfinite(grow_m)
        else []
    )
    cnts = cnts or None

    base, shd = {}, {}
    for j in range(int(np.floor((s_cam - radius) / cross_sp)), int(np.ceil((s_cam + radius) / cross_sp)) + 1):
        for k in range(int(np.floor((t_cam - radius) / lat_pitch)), int(np.ceil((t_cam + radius) / lat_pitch)) + 1):
            q = fwd * (j * cross_sp) + lat * (k * lat_pitch)
            d = float(np.hypot(*(q - foot)))
            if d > radius or d < LATTICE_NEAR_M:
                continue

            g = 1.0
            if cnts is not None:
                sd = max(float(cv2.pointPolygonTest(c, (float(q[0]), float(q[1])), True)) for c in cnts)
                if sd < -grow_m:
                    continue
                g = 1.0 if sd >= 0 else 1.0 + sd / max(grow_m, 1e-9)
            base[(j, k)] = lift(q)[0]

            shd[(j, k)] = float(
                np.clip(
                    min(
                        1.0 if d <= fade else 1.0 - (d - fade) / max(radius - fade, 1e-9),
                        (d - LATTICE_NEAR_M) / max(LATTICE_NEAR_FADE_M - LATTICE_NEAR_M, 1e-9),
                        g,
                    ),
                    0.0,
                    1.0,
                )
            )

    rods, nodes = [], []
    for jk, p0 in base.items():
        sh = shd[jk]
        for m in range(levels):
            rods.append((p0 + rod_up * (m * v_m), p0 + rod_up * ((m + 1) * v_m), sh))
        for m in range(1, levels + 1):
            nodes.append((p0 + rod_up * (m * v_m), sh))

    for (j, k), p0 in base.items():
        for dj, dk in ((1, 0), (0, 1)):
            q = base.get((j + dj, k + dk))
            if q is None:
                continue
            sh = min(shd[(j, k)], shd[(j + dj, k + dk)])
            for m in range(1, levels + 1):
                rods.append((p0 + rod_up * (m * v_m), q + rod_up * (m * v_m), sh))
    return rods, nodes


def instance_rgb(track_id):
    """Stable saturated colour per track id."""
    import colorsys

    h = (int(track_id) * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255))


def _env_az0(scene, c2w0):
    """Backdrop azimuth zero: heading of the anchor camera."""
    gv = scene.get("gravity")
    up = np.array([0.0, 1.0, 0.0]) if gv is None else -np.asarray(gv, float)
    up = up / max(np.linalg.norm(up), 1e-9)
    e0 = np.array([1.0, 0.0, 0.0])
    if abs(e0 @ up) > 0.9:
        e0 = np.array([0.0, 0.0, 1.0])
    ex = e0 - (e0 @ up) * up
    ex /= np.linalg.norm(ex)
    ez = np.cross(up, ex)
    f = np.asarray(c2w0, float)[:3, 2]

    return float(np.arctan2(f @ ex, f @ ez))


def box_corners(center, size, yaw, up, ex, ey):
    """8 corners of a gravity-aligned OBB."""
    c = np.asarray(center, float)
    sx_, sy_, sh = size
    ca, sa = np.cos(yaw), np.sin(yaw)
    ax = ca * ex + sa * ey
    az = -sa * ex + ca * ey
    out = []
    for a in (-1, 1):
        for b in (-1, 1):
            for e in (-1, 1):
                out.append(c + ax * (a * sx_ / 2) + up * (b * sh / 2) + az * (e * sy_ / 2))
    return np.stack(out), ax, az


BOX_FACES = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]


def gl_projection(K, W, H, near=0.05, far=500.0):
    """OpenCV pixel intrinsics -> GL projection matrix (row-major)."""
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    P = np.zeros((4, 4), np.float64)
    P[0, 0] = 2.0 * fx / W
    P[0, 2] = 1.0 - 2.0 * cx / W
    P[1, 1] = 2.0 * fy / H
    P[1, 2] = 2.0 * cy / H - 1.0
    P[2, 2] = -(far + near) / (far - near)
    P[2, 3] = -2.0 * far * near / (far - near)
    P[3, 2] = -1.0
    return P


def gl_view(c2w):
    """c2w -> GL view matrix, including the OpenCV -> GL y/z flip."""
    R, t = np.asarray(c2w, float)[:3, :3], np.asarray(c2w, float)[:3, 3]
    V = np.eye(4)
    V[:3, :3] = R.T
    V[:3, 3] = -R.T @ t
    return np.diag([1.0, -1.0, -1.0, 1.0]) @ V


RGB = {
    "grid": (99, 153, 34),
    "amber": (186, 117, 23),
    "red": (226, 75, 74),
    "purple": (127, 119, 221),
    "blue": (55, 138, 221),
    "yellow": (255, 255, 15),
    "state_blue": (30, 90, 255),
    "cyan": (24, 227, 242),
}

R = sys.modules[__name__]


_GROUND_VS = """
#version 330
uniform mat4 mvp;
in vec3 pos;
out vec3 w;
void main() { w = pos; gl_Position = mvp * vec4(pos, 1.0); }
"""


_GROUND_FS = """
#version 330
uniform sampler2D bev;
uniform vec4  bev_box;
uniform vec2  fwd, lat;
uniform vec2  pitch;
uniform vec2  cam_st;
uniform vec3  cam;
uniform vec3  grid_rgb;
uniform float fill_alpha, fade_start, fade_floor, line_px;
uniform float win_back, win_fwd, win_half;
in  vec3 w;
out vec4 f;

float line_mask(float s, float sp) {
    float g  = abs(fract(s / sp - 0.5) - 0.5) * sp;
    float aa = max(fwidth(s), 1e-6) * line_px;
    return 1.0 - smoothstep(0.0, aa, g);
}

void main() {
    vec2 uv = (w.xz - bev_box.xy) / bev_box.zw;
    if (texture(bev, uv).r < 0.5) discard;

    vec2 p = vec2(dot(w.xz, fwd), dot(w.xz, lat));
    vec2 rel = p - cam_st;



    bool in_win = !(rel.x > win_fwd || rel.x < -win_back
                    || abs(rel.y) > win_half);
    float m = in_win ? max(line_mask(p.y, pitch.x), line_mask(p.x, pitch.y)) : 0.0;
    float d = length(w - cam);
    float fade = (d <= fade_start) ? 1.0
               : max(fade_floor, 1.0 - (1.0 - fade_floor)
                     * (d - fade_start) / max(win_fwd - fade_start, 1e-3));



    f = vec4(grid_rgb, max(m * fade, fill_alpha));
}
"""

_FLAT_VS = """
#version 330
uniform mat4 mvp;
in vec3 pos;
in vec4 rgba;
out vec4 v;
void main() { v = rgba; gl_Position = mvp * vec4(pos, 1.0); }
"""
_FLAT_FS = """
#version 330
in vec4 v;
out vec4 f;
void main() { f = v; }
"""


_NODE_VS = """
#version 330
uniform mat4 mvp;
uniform mat4 view;
uniform float foc, k_world;
uniform vec2 px_clamp;
in vec3 pos;
in vec4 rgba;
out vec4 v;
void main() {
    v = rgba;
    vec4 vp = view * vec4(pos, 1.0);
    float z = max(-vp.z, 1e-3);
    gl_PointSize = clamp(foc * k_world / z, px_clamp.x, px_clamp.y) * 2.0;
    gl_Position = mvp * vec4(pos, 1.0);
}
"""

_NODE_FS = """
#version 330
in vec4 v;
out vec4 f;
void main() {
    if (length(gl_PointCoord - vec2(0.5)) > 0.5) discard;
    f = v;
}
"""


_ENV_VS = "#version 330\nin vec2 p;void main(){gl_Position=vec4(p,1,1);}"
_ENV_FS = """
#version 330
uniform mat3 Rc2w;
uniform vec4 Kpx;
uniform vec2 res;
uniform vec3 up;
uniform float az0, strength;
out vec4 f;
const float PI = 3.14159265359;
void main() {

    vec2 uv = vec2(gl_FragCoord.x, res.y - gl_FragCoord.y);
    vec3 d = normalize(vec3((uv.x - Kpx.z) / Kpx.x,
                            (uv.y - Kpx.w) / Kpx.y, 1.0));
    vec3 w = Rc2w * d;
    vec3 e0 = (abs(dot(vec3(1,0,0), up)) > 0.9) ? vec3(0,0,1) : vec3(1,0,0);
    vec3 ex = normalize(e0 - dot(e0, up) * up);
    vec3 ez = cross(up, ex);
    float az = atan(dot(w, ex), dot(w, ez)) - az0;
    az = mod(az + PI, 2.0 * PI) - PI;
    float el = asin(clamp(dot(w, up), -1.0, 1.0));
    float h  = fract(az / (2.0 * PI) + 0.5);
    vec3 rgb = 0.5 + 0.5 * cos(2.0 * PI * (h + vec3(0.0, 0.34, 0.67)));
    float L  = el / (PI * 0.5);
    float hi = clamp(L, 0.0, 1.0) * 0.55;
    float lo = clamp(-L, 0.0, 1.0) * 0.50;
    rgb = rgb * (1.0 - hi - lo) + hi;
    f = vec4(clamp(rgb * strength, 0.0, 1.0), 1.0);
}
"""

_BG_VS = "#version 330\nin vec2 p;out vec2 uv;void main(){uv=p*0.5+0.5;gl_Position=vec4(p,1,1);}"
_BG_FS = "#version 330\nuniform sampler2D t;in vec2 uv;out vec4 f;void main(){f=vec4(texture(t,uv).rgb,1);}"


def bake_bev(faces, res=2048, pad=2.0):
    """Walkable polygons -> BEV mask texture, one per ground region."""
    polys = []
    for fc in faces:
        v = fc.get("walkable_polygons_xz")
        cand = v if (v and np.asarray(v[0]).ndim == 2) else None
        if cand is None:
            v = fc.get("walkable_polygon_xz")
            cand = [v] if v else []
        for p in cand:
            a = np.asarray(p, float)
            if a.ndim == 2 and len(a) >= 3:
                polys.append(a)
    if not polys:
        return None, None
    allp = np.concatenate(polys, 0)
    lo = allp.min(0) - pad
    span = np.maximum(allp.max(0) + pad - lo, 1e-6)
    m = np.zeros((res, res), np.uint8)
    for p in polys:
        uv = (p - lo) / span
        cv2.fillPoly(m, [np.round(np.column_stack([uv[:, 0] * (res - 1), uv[:, 1] * (res - 1)])).astype(np.int32)], 255)
    return m, (float(lo[0]), float(lo[1]), float(span[0]), float(span[1]))


def smooth_poses(c2w, sigma):
    """Light temporal Gaussian smoothing of camera poses; rotations are re-orthogonalised by SVD."""
    if not sigma:
        return c2w
    from scipy.ndimage import gaussian_filter1d

    out = np.asarray(c2w, float).copy()
    out[:, :3, 3] = gaussian_filter1d(out[:, :3, 3], sigma, axis=0, mode="nearest")
    Rm = gaussian_filter1d(np.asarray(c2w, float)[:, :3, :3], sigma, axis=0, mode="nearest")
    u, _, vt = np.linalg.svd(Rm)
    out[:, :3, :3] = u @ vt
    return out


def world_lattice(cam_t, origin, S, N, R_max, fade):
    """World-fixed lattice of S x S x S cells around the camera."""
    o = np.asarray(origin, float)
    t = np.asarray(cam_t, float)
    c = np.round((t - o) / S).astype(int)

    rng = [np.arange(c[a] - N, c[a] + N + 1) for a in range(3)]
    I, J, Kk = np.meshgrid(*rng, indexing="ij")
    P = o + np.stack([I, J, Kk], -1) * S
    d = np.linalg.norm(P - t, axis=-1)
    keep = d <= R_max
    if not keep.any():
        return [], []
    a = np.where(d <= fade, 1.0, np.clip(1 - (d - fade) / max(R_max - fade, 1e-9), 0.0, 1.0))
    rods = []
    for ax in range(3):
        sl0 = [slice(None)] * 3
        sl1 = [slice(None)] * 3
        sl0[ax] = slice(None, -1)
        sl1[ax] = slice(1, None)
        both = keep[tuple(sl0)] & keep[tuple(sl1)]
        if not both.any():
            continue
        p0 = P[tuple(sl0)][both]
        p1 = P[tuple(sl1)][both]
        sh = np.minimum(a[tuple(sl0)][both], a[tuple(sl1)][both])
        rods += list(zip(p0, p1, sh))
    nodes = list(zip(P[keep], a[keep]))
    return rods, nodes


class GLSceneRenderer:
    """Renders one clip in a single GL context (shaders, textures and framebuffers are built once)."""

    def __init__(
        self,
        scene,
        W,
        H,
        samples=8,
        layers=("ground", "dynamic"),
        lattice_levels=0,
        lattice_color="cyan",
        env_strength=0.0,
        env_az0=None,
        bound="walkable",
        extent="auto",
        face_shade=1.0,
        box_color="single",
        ground_ruler=True,
        grid_px=2.0,
        lattice_px=2.0,
        node_scale=2.0,
        lat_pitch=None,
        cross_sp=None,
        v_m=None,
        lattice_mode="ground",
        lattice_s=1.5,
        lattice_n=3,
        lattice_origin=None,
        lattice_kw=None,
    ):
        import moderngl

        self.mgl, self.sc = moderngl, scene
        self.W, self.H = int(W), int(H)
        self.layers, self.bound, self.extent = layers, bound, extent
        self.face_shade, self.box_color = face_shade, box_color
        self.lattice_levels, self.lattice_color = lattice_levels, lattice_color
        self.ground_ruler = ground_ruler

        self.grid_px, self.lattice_px = float(grid_px), float(lattice_px)
        self.node_scale = float(node_scale)

        self.lat_pitch = float(lat_pitch or R.RULER_LAT_PITCH_M)
        self.cross_sp = float(cross_sp or R.RULER_CROSS_SPACING_M)
        self.v_m = float(v_m or R.LATTICE_V_M)
        self.lattice_mode = lattice_mode
        self.lattice_s, self.lattice_n = float(lattice_s), int(lattice_n)
        self.lattice_origin = lattice_origin
        self.env_strength, self.env_az0 = env_strength, env_az0
        self.lattice_kw = lattice_kw or {}

        import fcntl

        _lkf = open("/tmp/.gl_egl_init.lock", "w")

        _di = os.environ.get("GL_DEVICE_INDEX")
        _kw = {"device_index": int(_di)} if _di else {}
        try:
            fcntl.flock(_lkf, fcntl.LOCK_EX)
            self.ctx = moderngl.create_context(standalone=True, backend="egl", **_kw)
        finally:
            fcntl.flock(_lkf, fcntl.LOCK_UN)
            _lkf.close()

        _rname = str(self.ctx.info.get("GL_RENDERER", ""))
        if "NVIDIA" not in _rname and os.environ.get("GL_ALLOW_SOFT") != "1":
            raise RuntimeError(f"EGL fell back to a software renderer: {_rname!r} (set GL_ALLOW_SOFT=1 to allow it)")
        s = min(int(samples), int(self.ctx.info.get("GL_MAX_SAMPLES", 4)))
        self.ms = self.ctx.framebuffer(
            self.ctx.renderbuffer((self.W, self.H), samples=s), self.ctx.depth_renderbuffer((self.W, self.H), samples=s)
        )
        self.solo = self.ctx.simple_framebuffer((self.W, self.H))
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        self.p_ground = self.ctx.program(vertex_shader=_GROUND_VS, fragment_shader=_GROUND_FS)
        self.p_flat = self.ctx.program(vertex_shader=_FLAT_VS, fragment_shader=_FLAT_FS)
        self.p_node = self.ctx.program(vertex_shader=_NODE_VS, fragment_shader=_NODE_FS)
        self.p_env = self.ctx.program(vertex_shader=_ENV_VS, fragment_shader=_ENV_FS)
        self.p_bg = self.ctx.program(vertex_shader=_BG_VS, fragment_shader=_BG_FS)
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.bg_vbo = self.ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], "f4").tobytes())

        self.faces = R.ground_faces(scene) if "ground" in layers else []
        self.bev = []
        for fc in self.faces:
            mask, box = bake_bev([fc])
            if mask is None:
                self.bev.append(None)
                continue

            t = self.ctx.texture((mask.shape[1], mask.shape[0]), 1, np.ascontiguousarray(mask).tobytes())
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.bev.append((t, box))

    def frame(self, K, c2w, f=0):
        self.ms.use()
        self.ctx.viewport = (0, 0, self.W, self.H)
        self.ms.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)
        mvp = (gl_projection(K, self.W, self.H) @ gl_view(c2w)).astype("f4")
        mvp_b = mvp.T.tobytes()
        cam_c = np.asarray(c2w, float)[:3, 3]

        if self.env_strength > 0:
            self._env(K, c2w)

        for i in range(len(self.faces)):
            if self.bev[i] is not None:
                self._ground(mvp_b, cam_c, i)
        self._boxes(mvp_b, f, c2w)

        if self.lattice_levels > 0 and (self.faces or self.lattice_mode == "world"):
            self._lattice(mvp_b, cam_c, K, c2w)
        return self._read()

    def _env(self, K, c2w):
        gv = self.sc.get("gravity")
        up = -np.asarray(gv, float) if gv is not None else np.array([0.0, 1.0, 0.0])
        up = up / max(np.linalg.norm(up), 1e-9)
        p = self.p_env
        p["Rc2w"].write(np.asarray(c2w, "f4")[:3, :3].T.copy().tobytes())
        p["Kpx"].value = (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
        p["res"].value = (float(self.W), float(self.H))
        p["up"].value = tuple(float(x) for x in up)
        p["az0"].value = float(self.env_az0 or 0.0)
        p["strength"].value = float(self.env_strength)
        vao = self.ctx.vertex_array(p, [(self.bg_vbo, "2f", "p")])
        self.ctx.depth_func = "1"
        vao.render(self.mgl.TRIANGLES)
        self.ctx.depth_func = "<"
        vao.release()

    def _blit(self, bg):
        tex = self.ctx.texture(
            (self.W, self.H), 3, np.ascontiguousarray(np.asarray(bg, np.uint8)[::-1, :, ::-1]).tobytes()
        )
        vao = self.ctx.vertex_array(self.p_bg, [(self.bg_vbo, "2f", "p")])
        self.ctx.depth_func = "1"
        tex.use(0)
        vao.render(self.mgl.TRIANGLES)
        self.ctx.depth_func = "<"
        tex.release()
        vao.release()

    def _ground(self, mvp_b, cam_c, i):
        p = self.p_ground
        tex, box = self.bev[i]
        fc = self.faces[i]
        pd = np.asarray(fc.get("path_dir_xz") or [0.0, 1.0], float)
        pd = pd / max(np.linalg.norm(pd), 1e-9)
        lat = np.array([-pd[1], pd[0]])
        foot = np.array([cam_c[0], cam_c[2]])

        poly = fc.get("walkable_polygon_xz")
        d_back, d_fwd, half = R.ruler_window(
            np.asarray(poly, float) if poly else None, pd, lat, float(foot @ pd), float(foot @ lat)
        )
        tex.use(0)
        p["mvp"].write(mvp_b)
        p["bev"].value = 0
        p["bev_box"].value = box
        p["fwd"].value = tuple(pd)
        p["lat"].value = tuple(lat)
        p["pitch"].value = (self.lat_pitch, self.cross_sp)
        p["cam_st"].value = (float(foot @ pd), float(foot @ lat))
        p["cam"].value = tuple(float(x) for x in cam_c)
        p["grid_rgb"].value = tuple(np.asarray(R.RGB["grid"], float) / 255.0)
        p["fill_alpha"].value = float(R.GROUND_ALPHA)
        p["fade_start"].value = float(R.RULER_FADE_START_M)
        p["fade_floor"].value = float(R.RULER_FADE_FLOOR)
        p["line_px"].value = self.grid_px if self.ground_ruler else 0.0
        p["win_back"].value = float(d_back)
        p["win_fwd"].value = float(d_fwd)
        p["win_half"].value = float(half)

        lift = R.make_lift(fc)
        x0, z0, sx, sz = box
        q = lift(np.array([[x0, z0], [x0 + sx, z0], [x0 + sx, z0 + sz], [x0, z0 + sz]], float))
        vbo = self.ctx.buffer(np.asarray(q[[0, 1, 2, 0, 2, 3]], "f4").tobytes())
        vao = self.ctx.vertex_array(p, [(vbo, "3f", "pos")])
        vao.render(self.mgl.TRIANGLES)
        vbo.release()
        vao.release()

    def _boxes(self, mvp_b, f, c2w=None):
        """Draw the object boxes of frame f."""
        ex = np.array([1.0, 0.0, 0.0])
        ey = np.cross(R.UP, ex)
        tris, cols = [], []

        def emit(center, size, yaw, rgb):
            corners, ax, _ = box_corners(center, size, yaw, R.UP, ex, ey)
            for fcs in BOX_FACES:
                quad = corners[list(fcs)]
                sh = self.face_shade if (quad.mean(0) - np.asarray(center)) @ ax > 0 else 1.0
                c = [*(np.asarray(rgb, float) / 255.0 * sh).clip(0, 1), 1.0]
                for t in ((0, 1, 2), (0, 2, 3)):
                    for i in t:
                        tris.append(quad[i])
                        cols.append(c)

        for b in self.sc.get("boxes", []) if "static" in self.layers else []:
            emit(b["center"], b["size"], b.get("yaw", 0.0), R.RGB["red"])
        for b in self.sc.get("boxes_dynamic", []) if "dynamic" in self.layers else []:
            fr = b.get("frames", {}).get(str(f))
            if fr is None or not fr.get("visible", True):
                continue
            st = fr.get("state", "default")
            rgb = (
                R.instance_rgb(b.get("track_id", 0))
                if self.box_color == "instance"
                else tuple(st)
                if isinstance(st, (list, tuple))
                else R.RGB[R.STATE_RGB.get(st, "purple")]
            )

            if fr.get("holder") == "camera" and fr.get("offset_cam") is not None:
                cw = np.asarray(c2w, float)
                cen = cw[:3, 3] + cw[:3, :3] @ np.asarray(fr["offset_cam"], float)
                hd = cw[:3, :3] @ np.array([0.0, 0.0, 1.0])
                yaw = float(fr.get("yaw_base", 0.0) + math.atan2(-hd[2], hd[0]))
                emit(cen, fr["size"], yaw, rgb)
            else:
                emit(fr["center"], fr["size"], fr["yaw"], rgb)
        if tris:
            self._flat(mvp_b, np.asarray(tris, "f4"), np.asarray(cols, "f4"), self.mgl.TRIANGLES)

    def _lattice(self, mvp_b, cam_c, K, c2w):
        """Draw lattice rods (lines) and nodes (points)."""

        _lp, _cs = R.RULER_LAT_PITCH_M, R.RULER_CROSS_SPACING_M
        R.RULER_LAT_PITCH_M, R.RULER_CROSS_SPACING_M = (self.lat_pitch, self.cross_sp)
        if self.lattice_mode == "world":
            rods, nodes = world_lattice(
                cam_c,
                self.lattice_origin if self.lattice_origin is not None else cam_c,
                self.lattice_s,
                self.lattice_n,
                self.lattice_kw.get("radius", R.LATTICE_RADIUS_M),
                self.lattice_kw.get("fade", R.LATTICE_FADE_M),
            )
            self._emit_lattice(mvp_b, rods, nodes, K, c2w)
            return
        gv = self.sc.get("gravity")
        up = -np.asarray(gv, float) if gv is not None and not self.lattice_kw.get("rod_normal") else None
        polys = [np.asarray(fc["walkable_polygon_xz"], float) for fc in self.faces if fc.get("walkable_polygon_xz")]
        rods, nodes = R.lattice_rods_nodes(
            self.faces[0],
            cam_c,
            grow_m=self.lattice_kw.get("grow", R.LATTICE_GROW_M),
            levels=self.lattice_levels,
            radius=self.lattice_kw.get("radius", R.LATTICE_RADIUS_M),
            v_m=self.v_m,
            polys=polys,
            up=up,
        )
        R.RULER_LAT_PITCH_M, R.RULER_CROSS_SPACING_M = _lp, _cs
        self._emit_lattice(mvp_b, rods, nodes, K, c2w)

    def _emit_lattice(self, mvp_b, rods, nodes, K, c2w):
        base = np.asarray(R.RGB[self.lattice_color], float) / 255.0

        if rods:
            P, C = [], []
            for a, b, sh in rods:
                P += [np.asarray(a, "f4"), np.asarray(b, "f4")]

                C += [[*base, float(sh)]] * 2
            self.ctx.line_width = self.lattice_px
            self._flat(mvp_b, np.asarray(P, "f4"), np.asarray(C, "f4"), self.mgl.LINES)
        if nodes:
            P = np.asarray([np.asarray(p, "f4") for p, _ in nodes], "f4")
            C = np.asarray([[*base, float(sh)] for _, sh in nodes], "f4")
            p = self.p_node
            p["mvp"].write(mvp_b)
            p["view"].write(gl_view(c2w).T.astype("f4").tobytes())
            p["foc"].value = float(0.5 * (K[0, 0] + K[1, 1]))
            p["k_world"].value = float(R.LATTICE_NODE_K) * self.node_scale
            p["px_clamp"].value = tuple(float(x) for x in R.LATTICE_NODE_PX)
            vbo = self.ctx.buffer(P.tobytes())
            cbo = self.ctx.buffer(C.tobytes())
            vao = self.ctx.vertex_array(p, [(vbo, "3f", "pos"), (cbo, "4f", "rgba")])
            vao.render(self.mgl.POINTS)
            for o in (vbo, cbo, vao):
                o.release()

    def _flat(self, mvp_b, V, C, mode):
        vbo = self.ctx.buffer(V.tobytes())
        cbo = self.ctx.buffer(C.tobytes())
        vao = self.ctx.vertex_array(self.p_flat, [(vbo, "3f", "pos"), (cbo, "4f", "rgba")])
        self.p_flat["mvp"].write(mvp_b)
        vao.render(mode)
        for o in (vbo, cbo, vao):
            o.release()

    def _read(self):
        self.ctx.copy_framebuffer(self.solo, self.ms)
        a = np.frombuffer(self.solo.read(components=3), np.uint8)
        return np.ascontiguousarray(a.reshape(self.H, self.W, 3)[::-1, :, ::-1])

    def release(self):
        self.ctx.release()


def render(scene, c2ws, Ks, out, anchor_c2w=None, fps=24, size=(832, 480)):
    """Render the condition video.

    scene: scene dict (ground planes, walkable polygons, boxes_dynamic); c2ws / Ks: per-frame OpenCV camera-to-world
    poses and pixel intrinsics at scene["intrinsics"] resolution. anchor_c2w fixes the world lattice and the backdrop
    azimuth; a chained segment passes the first pose of the clip it continues, otherwise the first pose is used.
    """
    import imageio.v2 as imageio

    c2ws = smooth_poses(np.asarray(c2ws, float), 0.9)
    Ks = np.asarray(Ks, float)
    Ks = np.tile(Ks, (len(c2ws), 1, 1)) if Ks.ndim == 2 else Ks.copy()
    anchor_c2w = c2ws[0] if anchor_c2w is None else np.asarray(anchor_c2w, float)
    W, H = size
    Ks[:, 0] *= W / scene["intrinsics"]["w"]
    Ks[:, 1] *= H / scene["intrinsics"]["h"]
    r = GLSceneRenderer(
        scene,
        W,
        H,
        samples=8,
        layers=("ground", "dynamic"),
        lattice_levels=2,
        lattice_color="cyan",
        env_strength=0.35,
        box_color="single",
        ground_ruler=True,
        grid_px=2.0,
        lattice_px=2.0,
        node_scale=2.0,
        lat_pitch=None,
        cross_sp=None,
        v_m=None,
        lattice_mode="world",
        lattice_s=1.5,
        lattice_n=int(np.ceil(LATTICE_RADIUS_M / 1.5)) + 1,
        lattice_kw={"radius": LATTICE_RADIUS_M, "fade": LATTICE_FADE_M},
        lattice_origin=anchor_c2w[:3, 3],
        env_az0=_env_az0(scene, anchor_c2w),
    )
    tmp = f"{out}.{os.getpid()}.tmp.mp4"
    wr = imageio.get_writer(tmp, fps=fps, codec="libx264", quality=8, macro_block_size=1)
    for f in range(len(c2ws)):
        wr.append_data(r.frame(Ks[f], c2ws[f], f)[:, :, ::-1])
    wr.close()
    r.release()
    os.replace(tmp, out)
    return out
