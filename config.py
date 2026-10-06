from dataclasses import dataclass


@dataclass
class Params:
    x_src: int = 60
    x_wall: int = 150
    x_det: int = 480

    nx: int = 1000
    ny: int = 400

    wall_thickness: int = 4
    sponge: int = 40

    courant: float = 0.5

    
    wavelength: float = 12.0
    n_slits: int = 2
    slit_width: float = 10.0
    slit_spacing: float = 40.0

    source_angle_deg: float = 0.0
    source_phase: float = 0.0
    source_y_offset: float = 0.0
    coherent: bool = True
    particle_mode: bool = False

    detector_oversampling: int = 4
    detector_width: int = 4

    wavelength_nm: float = 632.8
    slit_width_um: float = 50.0
    slit_spacing_um: float = 250.0
    screen_distance_m: float = 1.5
    screen_half_height_mm: float = 15.0