"""Grid-based scalar wave propagation and a matching Fraunhofer reference."""

import numpy as np

from config import Params


class WaveSim:
    AVG_STEPS = 480

    def __init__(self, p: Params):
        self.p = p

        shape = (p.ny, p.nx)

        self.u = np.zeros(shape, np.float32)
        self.u_prev = np.zeros_like(self.u)
        self.u_next = np.zeros_like(self.u)
        self.lap = np.zeros_like(self.u)

        self.I_det = np.zeros(p.ny, np.float32)

        self.damping = self._build_absorber()
        self.damping_prev = 1 - 0.5 * self.damping
        self.damping_next = 1 / (1 + 0.5 * self.damping)

        self.rng = np.random.default_rng()

        self.t_step = 0

        self.rebuild(fresh=True)

    def _build_absorber(self):
        p = self.p
        
        x = np.arange(p.nx)
        y = np.arange(p.ny)

        depth_x = np.maximum(0,p.sponge - np.minimum(x, p.nx - 1 - x)) / p.sponge
        depth_y = np.maximum(0,p.sponge - np.minimum(y, p.ny - 1 - y)) / p.sponge
        edge_depth = np.maximum(depth_y[:, None], depth_x[None, :])
        return (0.6 * edge_depth**2).astype(np.float32)

    def rebuild(self, fresh=False):
        p = self.p
        yc = p.ny / 2
        y = np.arange(p.ny)


        self.slit_centers = (yc+ (np.arange(p.n_slits)- (p.n_slits - 1) / 2) *p.slit_spacing)

        self.slit_open = np.zeros(p.ny, dtype=bool)

        for c in self.slit_centers:
            self.slit_open |= (
                np.abs(y - c) <= p.slit_width / 2
            )

        open_ = np.ones((p.ny, p.nx),dtype=np.float32)
        open_[:,p.x_wall:p.x_wall + p.wall_thickness] = self.slit_open[:, None]

        self.wall_mask = open_
        tan_t = np.tan(np.deg2rad(p.source_angle_deg))
        yc_src = np.clip(yc- (p.x_wall - p.x_src) * tan_t,p.sponge,p.ny - p.sponge - 1)
        self.src_y = np.array([int(round(yc_src))])
        k = 2 * np.pi / p.wavelength

        self.phase_offset = (-k* (self.src_y - yc_src)* np.sin(np.deg2rad(p.source_angle_deg)))        
        self.phase_noise = np.zeros(len(self.src_y))
        self.omega = 2 * np.pi / p.wavelength

        self.C2 = np.float32(p.courant ** 2)
        
        propagation_steps = ((p.x_det - p.x_src)/ p.courant)
        
        self.settle = int(
            propagation_steps
            + 2 * self.AVG_STEPS
        )

        if fresh:
            self.settle += int(
                (p.x_wall - p.x_src)
                / p.courant
            )

    def reset(self):

        self.u.fill(0)
        self.u_prev.fill(0)
        self.u_next.fill(0)
        self.I_det.fill(0)

        self.t_step = 0

        self.rebuild(fresh=True)

    def step(self):

        p = self.p

        u = self.u
        up = self.u_prev
        un = self.u_next
        lap = self.lap

        lap[1:-1, 1:-1] = (
            u[2:, 1:-1]
            + u[:-2, 1:-1]
            + u[1:-1, 2:]
            + u[1:-1, :-2]
            - 4 * u[1:-1, 1:-1]
        )

        lap *= self.C2

        np.multiply(up, self.damping_prev, out=un)
        un *= -1
        un += 2 * u
        un += lap
        un *= self.damping_next

        phase = (
            self.omega
            * p.courant
            * self.t_step
            + self.phase_offset
        )

        if not p.coherent:

            self.phase_noise += (
                0.1
                * self.rng.standard_normal(
                    len(phase)
                )
            )

            phase = phase + self.phase_noise

        amplitude = min(
            1.0,
            self.t_step / 100
        )

        un[
            self.src_y,
            p.x_src
        ] += amplitude * np.sin(phase)

        un *= self.wall_mask

        self.u_prev, self.u, self.u_next = (
            u,
            un,
            up
        )

        x0 = max(
            0,
            p.x_det - p.detector_width + 1
        )

        detector = self.u[
            :,
            x0:p.x_det + 1
        ]

        instantaneous_I = np.mean(
            detector ** 2,
            axis=1
        )

        self.I_det += (
            instantaneous_I
            - self.I_det
        ) / self.AVG_STEPS

        self.t_step += 1
        self.settle -= 1


def _numerical_wavenumber(p, angle):
    omega_dt = 2 * np.pi * p.courant / p.wavelength
    target = np.sin(omega_dt / 2) ** 2 / p.courant**2
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
    aperture_y = np.arange(p.ny, dtype=float) - center
    detector_y = np.asarray(detector_rows, dtype=float) - center
    aperture_mask = np.asarray(slit_open, dtype=float)

    source_x = p.x_wall + p.wall_thickness / 2 - p.x_src
    screen_x = p.x_det - p.x_wall - p.wall_thickness / 2
    source_y = float(source_row) - center

    incident_angle = np.arctan2(aperture_y - source_y, source_x)
    incident_k = _numerical_wavenumber(p, incident_angle)
    source_distance = np.hypot(source_x, aperture_y - source_y)
    aperture_field = (
        aperture_mask
        * np.exp(1j * incident_k * source_distance)
        / np.sqrt(source_distance)
    )

    sin_theta = detector_y / np.hypot(detector_y, screen_x)
    detector_angle = np.arctan2(detector_y, screen_x)
    detector_k = _numerical_wavenumber(p, detector_angle)
    propagation = np.exp(
        -1j * detector_k[:, None] * sin_theta[:, None] * aperture_y[None, :]
    )
    intensity = np.abs(propagation @ aperture_field) ** 2

    peak = float(np.max(intensity))
    return intensity / peak if peak > 0 else intensity