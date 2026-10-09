
import os

os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
PARALLEL = os.environ.get("WAVESIM_PARALLEL", "1") != "0"

import numpy as np
from numba import njit, prange

from config import Params


@njit(parallel=PARALLEL, cache=False)
def _one_step(u, up, un, c_next, c_prev, C2, sy, sx, sv, x0, x1, I_det, avg):
    """

    un = mask * ( dn * (2u + C2*lap(u)) - dn*dp * up + source )
    with c_next = dn*mask and c_prev = dn*dp*mask precomputed.
    Border cells have lap == 0, exactly as in the original implementation.
    Its just DP
    """
    ny, nx = u.shape
    two = np.float32(2.0)
    four = np.float32(4.0)
    inv_w = 1.0 / (x1 - x0 + 1)

    for i in prange(ny):
        if i == 0 or i == ny - 1:
            for j in range(nx):
                un[i, j] = c_next[i, j] * (two * u[i, j]) - c_prev[i, j] * up[i, j]
        else:
            un[i, 0] = c_next[i, 0] * (two * u[i, 0]) - c_prev[i, 0] * up[i, 0]
            un[i, nx - 1] = (
                c_next[i, nx - 1] * (two * u[i, nx - 1])
                - c_prev[i, nx - 1] * up[i, nx - 1]
            )
            for j in range(1, nx - 1):
                lap = C2 * (
                    u[i + 1, j] + u[i - 1, j] + u[i, j + 1] + u[i, j - 1]
                    - four * u[i, j]
                )
                un[i, j] = c_next[i, j] * (two * u[i, j] + lap) - c_prev[i, j] * up[i, j]

        if i == sy:
            un[i, sx] += sv  
        s = 0.0
        for j in range(x0, x1 + 1):
            v = un[i, j]
            s += v * v
        I_det[i] += np.float32((s * inv_w - I_det[i]) / avg)


@njit(cache=False)
def _run_steps(u, up, un, c_next, c_prev, C2, sy, sx, sv_arr, x0, x1, I_det, avg):
    for k in range(sv_arr.shape[0]):
        _one_step(u, up, un, c_next, c_prev, C2, sy, sx, sv_arr[k], x0, x1, I_det, avg)
        u, up, un = un, u, up
    return u, up, un


def _warmup():
    a = np.zeros((8, 8), np.float32)
    b = np.zeros_like(a)
    c = np.zeros_like(a)
    ones = np.ones_like(a)
    I = np.zeros(8, np.float32)
    _run_steps(a, b, c, ones, ones, np.float32(0.25), 4, 2,
               np.zeros(1, np.float32), 4, 6, I, 480.0)


_WARMED = False
class WaveSim:
    AVG_STEPS = 480

    def __init__(self, p: Params):
        global _WARMED
        if not _WARMED:
            _warmup()
            _WARMED = True

        self.p = p

        shape = (p.ny, p.nx)

        self.u = np.zeros(shape, np.float32)
        self.u_prev = np.zeros_like(self.u)
        self.u_next = np.zeros_like(self.u)

        self.I_det = np.zeros(p.ny, np.float32)

        self.damping = self._build_absorber()
        self.damping_prev = (1 - 0.5 * self.damping).astype(np.float32)
        self.damping_next = (1 / (1 + 0.5 * self.damping)).astype(np.float32)

        self.rng = np.random.default_rng()

        self.t_step = 0

        self.rebuild(fresh=True)

    def _build_absorber(self):
        p = self.p

        x = np.arange(p.nx)
        y = np.arange(p.ny)

        depth_x = np.maximum(0, p.sponge - np.minimum(x, p.nx - 1 - x)) / p.sponge
        depth_y = np.maximum(0, p.sponge - np.minimum(y, p.ny - 1 - y)) / p.sponge
        edge_depth = np.maximum(depth_y[:, None], depth_x[None, :])
        return (0.6 * edge_depth ** 2).astype(np.float32)

    def rebuild(self, fresh=False):
        p = self.p
        yc = p.ny / 2
        y = np.arange(p.ny)

        self.slit_centers = (
            yc + (np.arange(p.n_slits) - (p.n_slits - 1) / 2) * p.slit_spacing
        )

        self.slit_open = np.zeros(p.ny, dtype=bool)

        for c in self.slit_centers:
            self.slit_open |= np.abs(y - c) <= p.slit_width / 2

        open_ = np.ones((p.ny, p.nx), dtype=np.float32)
        open_[:, p.x_wall:p.x_wall + p.wall_thickness] = self.slit_open[:, None]

        self.wall_mask = open_

        tan_t = np.tan(np.deg2rad(p.source_angle_deg))
        yc_src = np.clip(
            yc - (p.x_wall - p.x_src) * tan_t,
            p.sponge,
            p.ny - p.sponge - 1,
        )
        self.src_y = np.array([int(round(yc_src))])
        k = 2 * np.pi / p.wavelength

        self.phase_offset = (
            -k * (self.src_y - yc_src) * np.sin(np.deg2rad(p.source_angle_deg))
        )
        self.phase_noise = np.zeros(len(self.src_y))
        self.omega = 2 * np.pi / p.wavelength

        self.C2 = np.float32(p.courant ** 2)

        self.c_next = np.ascontiguousarray(
            self.damping_next * self.wall_mask, dtype=np.float32
        )
        self.c_prev = np.ascontiguousarray(
            self.damping_prev * self.damping_next * self.wall_mask, dtype=np.float32
        )
        self.src_mask = np.float32(self.wall_mask[self.src_y[0], p.x_src])

        propagation_steps = (p.x_det - p.x_src) / p.courant

        self.settle = int(propagation_steps + 2 * self.AVG_STEPS)

        if fresh:
            self.settle += int((p.x_wall - p.x_src) / p.courant)

    def reset(self):
        self.u.fill(0)
        self.u_prev.fill(0)
        self.u_next.fill(0)
        self.I_det.fill(0)

        self.t_step = 0

        self.rebuild(fresh=True)

    def step_n(self, n=1):
        if n <= 0:
            return

        p = self.p

        t = self.t_step + np.arange(n)
        phase = self.omega * p.courant * t + self.phase_offset[0]

        if not p.coherent:
            noise = self.phase_noise[0] + np.cumsum(0.1 * self.rng.standard_normal(n))
            self.phase_noise[0] = noise[-1]
            phase = phase + noise

        amplitude = np.minimum(1.0, t / 100)
        sv = (amplitude * np.sin(phase) * self.src_mask).astype(np.float32)

        x0 = max(0, p.x_det - p.detector_width + 1)
        ny, nx = self.u.shape
        if not (0 <= int(self.src_y[0]) < ny and 0 <= p.x_src < nx and 0 <= p.x_det < nx):
            raise ValueError(
                f"index outside grid: src=({self.src_y[0]},{p.x_src}) x_det={p.x_det} grid=({ny},{nx})"
            )
        for a in (self.u, self.u_prev, self.u_next, self.c_next, self.c_prev):
            if a.shape != (ny, nx) or a.dtype != np.float32 or not a.flags.c_contiguous:
                raise ValueError("simulation arrays must be C-contiguous float32 of identical shape")

        self.u, self.u_prev, self.u_next = _run_steps(
            self.u, self.u_prev, self.u_next,
            self.c_next, self.c_prev, self.C2,
            int(self.src_y[0]), int(p.x_src), sv,
            int(x0), int(p.x_det),
            self.I_det, float(self.AVG_STEPS),
        )

        self.t_step += n
        self.settle -= n

    def step(self):
        self.step_n(1)

def _numerical_wavenumber(p, angle):
    omega_dt = 2 * np.pi * p.courant / p.wavelength
    target = np.sin(omega_dt / 2) ** 2 / p.courant ** 2
    angle = np.asarray(angle, dtype=float)
    low = np.zeros_like(angle, dtype=float)
    high = np.full_like(angle, np.pi, dtype=float)

    for _ in range(32):
        wave_number = (low + high) / 2
        dispersion = (
            np.sin(wave_number * np.cos(angle) / 2) ** 2
            + np.sin(wave_number * np.sin(angle) / 2) ** 2
        )
        below_target = dispersion < target
        low = np.where(below_target, wave_number, low)
        high = np.where(below_target, high, wave_number)

    return (low + high) / 2


def fraunhofer(p, detector_rows, slit_open, source_row):
    center = p.ny / 2
    detector_y = np.asarray(detector_rows, dtype=float) - center

    idx = np.flatnonzero(np.asarray(slit_open, dtype=bool))
    aperture_y = idx.astype(float) - center

    source_x = p.x_wall + p.wall_thickness / 2 - p.x_src
    screen_x = p.x_det - p.x_wall - p.wall_thickness / 2
    source_y = float(source_row) - center

    incident_angle = np.arctan2(aperture_y - source_y, source_x)
    incident_k = _numerical_wavenumber(p, incident_angle)
    source_distance = np.hypot(source_x, aperture_y - source_y)
    aperture_field = np.exp(1j * incident_k * source_distance) / np.sqrt(source_distance)

    sin_theta = detector_y / np.hypot(detector_y, screen_x)
    detector_angle = np.arctan2(detector_y, screen_x)
    detector_k = _numerical_wavenumber(p, detector_angle)
    propagation = np.exp(
        -1j * detector_k[:, None] * sin_theta[:, None] * aperture_y[None, :]
    )
    intensity = np.abs(propagation @ aperture_field) ** 2

    peak = float(np.max(intensity)) if intensity.size else 0.0
    return intensity / peak if peak > 0 else intensity