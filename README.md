## Physical Model

The simulation models the propagation, diffraction, and interference of a two-dimensional scalar wave through an opaque barrier containing two slits. The wave field evolves according to a discretized wave equation, while the detector measures the time-averaged squared field.

### 1. Wave Generation

A harmonic source generates a sinusoidal disturbance:

$$
u_s(t)=A(t)\sin(\omega t+\phi_0),
$$

where $A(t)$ is the source amplitude, $\omega$ is the angular frequency, and $\phi_0$ is the initial phase.

The angular frequency and wavenumber are related to the wavelength by

$$
k=\frac{2\pi}{\lambda},
\qquad
\omega=ck,
$$

where $c$ is the propagation speed and $\lambda$ is the wavelength. For a grid with unit spatial spacing and a normalized propagation speed of $c=1$, the time step is $\Delta t=C$, where $C$ is the Courant number.

The source amplitude is gradually increased to avoid introducing an abrupt initial disturbance:

$$
A_n=\min\left(1,\frac{n}{100}\right).
$$

### 2. Wave Propagation

In free space, the continuous scalar wave equation is

$$
\frac{\partial^2 u}{\partial t^2}=c^2\nabla^2u+S(x,y,t),
$$

where $S$ represents the source and

$$
\nabla^2u=
\frac{\partial^2u}{\partial x^2}
+\frac{\partial^2u}{\partial y^2}.
$$

For a uniform square grid, the discrete spatial Laplacian is

$$
L_{i,j}^{n}=
u_{i+1,j}^{n}+u_{i-1,j}^{n}
+u_{i,j+1}^{n}+u_{i,j-1}^{n}
-4u_{i,j}^{n}.
$$

Using a centered finite-difference approximation in time, the undamped update equation becomes

$$
u_{i,j}^{n+1}
=2u_{i,j}^{n}-u_{i,j}^{n-1}
+C^2L_{i,j}^{n},
$$

where

$$
C=\frac{c\Delta t}{\Delta}.
$$

For the standard two-dimensional scheme on a square grid, the usual stability condition is

$$
C\leq\frac{1}{\sqrt{2}}.
$$

### 3. Absorbing Boundaries

To reduce artificial reflections from the edges of the computational domain, the simulation introduces spatially varying damping:

$$
\frac{\partial^2u}{\partial t^2}
+\gamma(x,y)\frac{\partial u}{\partial t}
=c^2\nabla^2u+S(x,y,t).
$$

The damping coefficient $\gamma(x,y)$ is small in the central region and increases near the boundaries. This approximates an absorbing layer that reduces the amplitude of outgoing waves before they reach the grid edges.

The corresponding discrete update is

$$
\begin{aligned}
u_{i,j}^{n+1}={}&
\frac{2u_{i,j}^{n}+C^2L_{i,j}^{n}}
{1+\gamma_{i,j}\Delta t/2}\\
&-\frac{1-\gamma_{i,j}\Delta t/2}
{1+\gamma_{i,j}\Delta t/2}u_{i,j}^{n-1}.
\end{aligned}
$$

### 4. The Double-Slit Barrier

An opaque barrier blocks the field everywhere except at the slit openings. This is represented by a binary mask:

$$
M(x,y)=
\begin{cases}
1,&\text{in free space and inside the slits},\\
0,&\text{inside the opaque barrier}.
\end{cases}
$$

The allowed field is

$$
u_{\mathrm{allowed}}(x,y,t)=M(x,y)u(x,y,t).
$$

The slit width and separation determine the geometry of the openings. When the incident wave reaches the barrier, the portions passing through the slits continue propagating and diffract into the region beyond the barrier.

The slits are not treated as independent artificial sources. Their outgoing waves emerge from the numerical propagation of the incident field through the openings.

### 5. Superposition and Interference

Because the scalar wave equation is linear, the total field is the sum of the contributions from the two slits:

$$
u(x,y,t)=u_1(x,y,t)+u_2(x,y,t).
$$

For two harmonic waves with equal amplitude $A$ and phase difference $\Delta\phi$, their sum is

$$
u=
2A\cos\left(\frac{\Delta\phi}{2}\right)
\cos\left(\omega t+\frac{\phi_1+\phi_2}{2}\right).
$$

The resulting amplitude depends on the phase difference:

* **Constructive interference:** $\Delta\phi=2m\pi$.
* **Destructive interference:** $\Delta\phi=(2m+1)\pi$.

For two slits separated by a distance $d$, the far-field path difference is approximately

$$
\Delta r=d\sin\theta.
$$

The phase difference is therefore

$$
\Delta\phi=\frac{2\pi d\sin\theta}{\lambda}.
$$

The approximate bright-fringe condition is

$$
d\sin\theta=m\lambda,
$$

and the dark-fringe condition is

$$
d\sin\theta=\left(m+\frac12\right)\lambda,
$$

where $m$ is an integer. These conditions assume coherent illumination and the far-field approximation.

### 6. Intensity Measurement

The detector measures a quantity proportional to the time-averaged squared wave field:

$$
I(x,y)\propto\left\langle u(x,y,t)^2\right\rangle_t.
$$

For two contributions,

$$
\begin{aligned}
I&\propto\left\langle(u_1+u_2)^2\right\rangle\\
&=\langle u_1^2\rangle+\langle u_2^2\rangle
+2\langle u_1u_2\rangle.
\end{aligned}
$$

The cross term $2\langle u_1u_2\rangle$ produces interference. Therefore, the fields must be added before calculating intensity; adding the individual intensities would remove the interference term.

The simulation averages the squared field over time and across a finite number of detector columns. It uses an exponential running average:

$$
I_i^{n+1}
=I_i^n+
\frac{\overline{u_i^2}^{\,n+1}-I_i^n}{N},
$$

where $\overline{u_i^2}$ is the spatial average across the detector columns and $N$ controls the averaging rate.

### 7. Coherence

For coherent illumination, the phase relationship between waves remains approximately stable, producing a stationary interference pattern.

The simulation can also introduce random-walk phase fluctuations:

$$
\phi_n=\omega\Delta t\,n+\phi_0+\eta_n,
$$

with

$$
\eta_n=\eta_{n-1}+\sigma\xi_n,
\qquad
\xi_n\sim\mathcal{N}(0,1).
$$

These fluctuations model a source with a changing phase. Time averaging can reduce the visibility of the fringes when the phase varies sufficiently during the observation period.

### 8. Fraunhofer Diffraction Reference

In the far-field regime, the complex field at observation angle $\theta$ is proportional to the Fourier transform of the field across the aperture:

$$
U(\theta)\propto
\int_{\mathrm{aperture}}
U_{\mathrm{ap}}(y')
e^{-iky'\sin\theta}\,dy'.
$$

The corresponding intensity is

$$
I(\theta)\propto|U(\theta)|^2.
$$

For a point source at a finite distance, the incident aperture field can be approximated by

$$
U_{\mathrm{ap}}(y')=
\frac{e^{ikr(y')}}{\sqrt{r(y')}},
$$

where

$$
r(y')=\sqrt{L_s^2+(y'-y_s)^2}.
$$

Here, $L_s$ is the source-to-aperture distance and $y_s$ is the source's vertical position. The factor $1/\sqrt{r}$ models the amplitude decay of a two-dimensional cylindrical wave.

For two identical slits of width $a$, separated by distance $d$ and illuminated by a coherent plane wave, the ideal Fraunhofer intensity is

$$
I(\theta)=I_{\max}
\cos^2\left(\frac{\pi d\sin\theta}{\lambda}\right)
\operatorname{sinc}^2\left(\frac{\pi a\sin\theta}{\lambda}\right),
$$

where

$$
\operatorname{sinc}(z)=\frac{\sin z}{z}.
$$

The cosine-squared factor describes interference between the slits, while the sinc-squared factor describes single-slit diffraction.

The numerical reference additionally corrects the wavenumber for the finite-difference grid's numerical dispersion. For dimensionless grid wavenumber $K$, propagation angle $\theta$, and Courant number $C$, the discrete dispersion relation is

$$
\sin^2\left(\frac{\omega\Delta t}{2}\right)
=C^2\left[
\sin^2\left(\frac{K\cos\theta}{2}\right)
+\sin^2\left(\frac{K\sin\theta}{2}\right)
\right].
$$

Solving this relation for each observation angle makes the reference more consistent with the numerical propagation scheme.

### Summary

The interference pattern arises from the physical superposition of waves passing through the two openings. The numerical wave equation propagates the field, the barrier imposes the slit geometry, and the detector measures the time-averaged squared total field. The Fraunhofer reference provides an independent far-field comparison, with corrections for the grid's numerical dispersion.
