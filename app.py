"""Interactive double-slit wave and detector simulation."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button, RadioButtons, Slider

from config import Params
from physics import WaveSim, fraunhofer
from sampling import ParticleSampler

p = Params()
sim = WaveSim(p)
sampler = ParticleSampler(p.ny, p.x_det)
state = {"paused": False, "scale": 1.0}
screen_arrival_step = int(np.ceil(
    (p.x_det - p.x_src) / p.courant
))
screen_pixel_size_mm = 2 * p.screen_half_height_mm / p.ny
screen_rows = np.arange(p.ny)
screen_y_mm = (
    (np.arange(p.ny) + 0.5) * screen_pixel_size_mm
    - p.screen_half_height_mm
)
fig = plt.figure(figsize=(10.5, 8.2))
gs = GridSpec(1, 3, width_ratios=[4, 0.3, 1.1], left=0.05, right=0.98, top=0.95,
              bottom=0.32, wspace=0.03)
ax_f = fig.add_subplot(gs[0])
ax_s = fig.add_subplot(gs[1])
ax_p = fig.add_subplot(gs[2])

im = ax_f.imshow(sim.u, origin="lower", cmap="coolwarm", vmin=-1, vmax=1,
                 interpolation="nearest", aspect="auto")
wall_im = ax_f.imshow(np.zeros((p.ny, p.nx, 4), np.uint8), origin="lower",
                      interpolation="nearest", aspect="auto")
src_pts, = ax_f.plot([], [], "g.", ms=3)
detector_line = ax_f.axvline(p.x_det, color="k", lw=2)
hit_sc = ax_f.scatter([], [], s=4, c="darkorange", edgecolors="none", visible=False)
status = ax_f.text(0.02, 0.96, "", transform=ax_f.transAxes, fontsize=9,
                   bbox=dict(fc="white", ec="none", alpha=0.8))
SCREEN_SCALE = p.detector_oversampling

screen_rgb = np.zeros((p.ny * SCREEN_SCALE,1,3),dtype=np.float32)

strip = ax_s.imshow(
    screen_rgb,
    origin="lower",
    extent=(0, 1, -p.screen_half_height_mm, p.screen_half_height_mm),
    vmin=0,
    vmax=1,
    aspect="auto",
    interpolation="nearest"
)
screen_hit_sc = ax_s.scatter(
    [],
    [],
    s=10,
    c="darkorange",
    edgecolors="white",
    linewidths=0.35,
    zorder=3,
    visible=False,
)
ax_s.set_xticks([])
ax_s.set_xlim(0, 1)
ax_s.set_ylabel("position (mm)", fontsize=8)
ax_s.set_title("detector", fontsize=8)
ax_s.set_ylim(-p.screen_half_height_mm, p.screen_half_height_mm)
ax_f.set_autoscale_on(False)
ax_f.set_xlim(-0.5, p.x_det + 8.5)
ax_f.set_title("qualitative wave field (grid units)", fontsize=9)
ax_f.set_xlabel("x (cells)")
ax_f.set_ylabel("y (grid cells)")

line_I, = ax_p.plot([], [], "b", lw=1.4, label="wave intensity", visible=False)
line_H, = ax_p.plot([], [], "k", lw=1, drawstyle="steps-mid", label="cumulative detections", visible=False)
line_F, = ax_p.plot([], [], "r--", lw=1.2, label="grid Fraunhofer approximation", visible=True)
ax_p.set_autoscale_on(False)
ax_p.set_xlim(0, 1)
ax_p.set_ylim(-p.screen_half_height_mm, p.screen_half_height_mm)
ax_p.set_title("screen profile", fontsize=9)
ax_p.grid(True, alpha=0.25)
ax_p.set_xlabel("normalized intensity")
ax_p.set_ylabel("position (mm)")
ax_p.legend(handles=[line_I, line_F], loc="upper right", fontsize=7)


def refresh_static():
    global fraunhofer_profile

    rgba = np.zeros((p.ny, p.nx, 4), np.uint8)
    rgba[~sim.slit_open, p.x_wall:p.x_wall + p.wall_thickness] = (25, 25, 25, 255)
    wall_im.set_data(rgba)
    src_pts.set_data(np.full(len(sim.src_y), p.x_src), sim.src_y)
    fraunhofer_profile = fraunhofer(
        p,
        screen_rows,
        sim.slit_open,
        sim.src_y[0],
    )
    line_F.set_data(fraunhofer_profile, screen_y_mm)


refresh_static()


def make_slider(rect, label, lo, hi, init, step):
    return Slider(fig.add_axes(rect), label, lo, hi, valinit=init, valstep=step)


X1, X2, W = 0.30, 0.72, 0.20
s_lam = make_slider([X1, 0.22, W, 0.03], "wavelength (nm)", 400, 700, p.wavelength_nm, 0.1)
s_n = make_slider([X1, 0.17, W, 0.03], "# slits", 1, 7, p.n_slits, 1)
s_a = make_slider([X1, 0.12, W, 0.03], "slit width (μm)", 10, 80, p.slit_width_um, 1)
s_d = make_slider([X1, 0.07, W, 0.03], "slit pitch (μm)", 100, 500, p.slit_spacing_um, 5)
s_L = make_slider([X2, 0.22, W, 0.03], "screen L (m)", 0.5, 3.5, p.screen_distance_m, 0.1)
s_ang = make_slider([X2, 0.17, W, 0.03], "incidence (deg)", -5, 5, p.source_angle_deg, 0.1)
s_spf = make_slider([X2, 0.12, W, 0.03], "steps/frame", 1, 30, 10, 1)
s_ppf = make_slider([X2, 0.07, W, 0.03], "mean photons/frame", 1, 100, 10, 1)

rb_mode = RadioButtons(fig.add_axes([0.005, 0.14, 0.10, 0.10]), ["Wave", "Particle"])
rb_coh = RadioButtons(fig.add_axes([0.005, 0.02, 0.10, 0.10]), ["Coherent", "Incoherent"])
b_reset = Button(fig.add_axes([0.12, 0.17, 0.08, 0.04]), "Reset")
b_pause = Button(fig.add_axes([0.12, 0.10, 0.08, 0.04]), "Pause")


def clear_particles():
    sampler.reset()


def wavelength_to_rgb(wavelength_nm):
    w = np.asarray(wavelength_nm, dtype=float)

    r = np.zeros_like(w)
    g = np.zeros_like(w)
    b = np.zeros_like(w)

    # 380-440: violet -> blue
    m = (w >= 380) & (w < 440)
    r[m] = -(w[m] - 440) / (440 - 380)
    g[m] = 0
    b[m] = 1

    # 440-490: blue -> cyan
    m = (w >= 440) & (w < 490)
    r[m] = 0
    g[m] = (w[m] - 440) / (490 - 440)
    b[m] = 1

    # 490-510: cyan -> green
    m = (w >= 490) & (w < 510)
    r[m] = 0
    g[m] = 1
    b[m] = -(w[m] - 510) / (510 - 490)

    # 510-580: green -> yellow
    m = (w >= 510) & (w < 580)
    r[m] = (w[m] - 510) / (580 - 510)
    g[m] = 1
    b[m] = 0

    # 580-645: yellow -> red
    m = (w >= 580) & (w < 645)
    r[m] = 1
    g[m] = -(w[m] - 645) / (645 - 580)
    b[m] = 0

    # 645-750: red
    m = (w >= 645) & (w <= 750)
    r[m] = 1
    g[m] = 0
    b[m] = 0

    # Fade near spectrum limits
    factor = np.ones_like(w)

    m = (w >= 380) & (w < 420)
    factor[m] = 0.3 + 0.7 * (
        (w[m] - 380) / 40
    )

    m = (w > 700) & (w <= 750)
    factor[m] = 0.3 + 0.7 * (
        (750 - w[m]) / 50
    )

    return np.stack([ r * factor, g * factor, b * factor], axis=-1)
def high_res_screen(intensity, oversampling=4):
    ny = len(intensity)

    y = np.arange(ny)
    y_hi = (
        (np.arange(ny * oversampling) + 0.5) / oversampling
        - 0.5
    )

    return np.interp(
        y_hi,
        y,
        intensity
    )


screen_color = wavelength_to_rgb(p.wavelength_nm)


def on_param(_=None):
    global screen_arrival_step, screen_color

    p.wavelength_nm = s_lam.val
    p.slit_width_um = s_a.val
    p.slit_spacing_um = s_d.val
    p.screen_distance_m = s_L.val
    p.x_det = p.x_wall + int(round(220 * p.screen_distance_m))
    p.source_angle_deg = s_ang.val

    p.wavelength = np.interp(p.wavelength_nm, [400, 700], [8, 24])
    screen_color = wavelength_to_rgb(p.wavelength_nm)
    p.n_slits = int(s_n.val)
    p.slit_width = p.slit_width_um * 0.18
    p.slit_spacing = p.slit_spacing_um * 0.18
    screen_arrival_step = int(np.ceil(
        (p.x_det - p.x_src) / p.courant
    ))
    sampler.x_detector = p.x_det
    detector_line.set_xdata([p.x_det, p.x_det])
    ax_f.set_xlim(-0.5, p.x_det + 8.5)
    sim.reset()
    refresh_static()
    clear_particles()


def on_mode(label):
    p.particle_mode = label == "Particle"
    im.set_alpha(0.3 if p.particle_mode else 1.0)
    hit_sc.set_visible(p.particle_mode)
    screen_hit_sc.set_visible(p.particle_mode)
    line_I.set_visible(not p.particle_mode)
    line_H.set_visible(False)
    line_F.set_label(
        "Fraunhofer expected counts" if p.particle_mode
        else "grid Fraunhofer approximation"
    )
    ax_p.set_xlabel(
        "detections per screen bin" if p.particle_mode
        else "normalized wave intensity"
    )
    ax_p.set_title(
        "cumulative photon detections" if p.particle_mode
        else "simulated detector intensity"
    )
    ax_p.set_xlim(0, 1.05)
    ax_p.legend(
        handles=[line_H, line_F] if p.particle_mode else [line_I, line_F],
        loc="upper right",
        fontsize=7,
    )
    clear_particles()


def on_coherence(label):
    p.coherent = label == "Coherent"
    sim.reset()
    refresh_static()
    clear_particles()


def on_reset(_):
    sim.reset()
    clear_particles()


def on_pause(_):
    state["paused"] = not state["paused"]
    b_pause.label.set_text("Resume" if state["paused"] else "Pause")


for s in (s_lam, s_n, s_a, s_d, s_L, s_ang):
    s.on_changed(on_param)
rb_mode.on_clicked(on_mode)
rb_coh.on_clicked(on_coherence)
b_reset.on_clicked(on_reset)
b_pause.on_clicked(on_pause)

def update(frame):
    if not state["paused"]:
        for _ in range(int(s_spf.val)):
            sim.step()
        if p.particle_mode and sim.t_step >= screen_arrival_step:
            sampler.sample(sim.I_det, int(s_ppf.val))

    disp = np.sign(sim.u) * np.sqrt(np.abs(sim.u))
    im.set_data(disp)
    if frame % 5 == 0:
        target = max(float(np.percentile(np.abs(disp[::2, p.x_src + 20::2]), 99.5)), 0.02)
        state["scale"] = 0.8 * state["scale"] + 0.2 * target
        im.set_clim(-state["scale"], state["scale"])

    arrived = sim.t_step >= screen_arrival_step

    if p.particle_mode:
        counts = sampler.hist
        max_count = float(counts.max())
        total_count = float(counts.sum())
        line_H.set_data(counts, screen_y_mm)
        line_H.set_visible(max_count > 0)
        expected_counts = (
            fraunhofer_profile * total_count / max(fraunhofer_profile.sum(), 1e-12)
        )
        line_F.set_data(expected_counts, screen_y_mm)
        ax_p.set_xlim(0, max(1, max(max_count, float(expected_counts.max())) * 1.1))
        hit_sc.set_offsets(sampler.recent)
        screen_hit_sc.set_offsets(
            np.column_stack(
                (
                    np.full(len(sampler.recent), 0.5),
                    (sampler.recent[:, 1] + 0.5)
                    * screen_pixel_size_mm
                    - p.screen_half_height_mm,
                )
            )
        )
        screen_hits = high_res_screen(counts, p.detector_oversampling)
        strip.set_data(screen_hits[:, None])
        strip.set_clim(0, max(1, max_count))
        if sim.t_step < screen_arrival_step:
            status.set_text("wave travelling to screen...")
        elif sampler.hist.sum() == 0:
            status.set_text("wave reached screen — accumulating detections...")
        else:
            status.set_text(f"{int(sampler.hist.sum())} detections")
    else:
        detector_peak = float(sim.I_det.max())
        wave_profile = sim.I_det / max(detector_peak, 1e-12)
        line_I.set_data(wave_profile, screen_y_mm)
        line_I.set_visible(arrived and detector_peak > 1e-12)
        screen_I = high_res_screen(wave_profile, p.detector_oversampling)
        screen_image = screen_I[:, None, None] * screen_color[None, None, :]
        strip.set_data(screen_image)
        line_H.set_visible(False)
        ax_p.set_xlim(0, 1.05)
        strip.set_clim(0, 1)
        if sim.t_step < screen_arrival_step:
            status.set_text("wave travelling to screen...")
        else:
            status.set_text("")


anim = FuncAnimation(fig, update, interval=15, cache_frame_data=False)
plt.show()