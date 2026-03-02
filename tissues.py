"""
Tissue Simulation (Stable Neo-Hookean FEM)

Deformable tissue block using Newton's tetrahedral FEM with the Stable
Neo-Hookean constitutive model. The energy is decomposed into:

  Psi = (lambda/2)(J - gamma)^2 + (mu/2)(tr(F^T F) - 3)
        ^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^
        hydrostatic (volume)        deviatoric (shape)

where gamma = 1 + mu/lambda ensures rest stability (Smith et al. 2018).

Real soft tissue is nearly incompressible (nu ~ 0.4999, Glozman & Azhari 2010).
Force-based solvers lose accuracy at high Poisson ratios (Macklin & Muller 2021,
Table 1), so we cap at nu=0.495 for semi-implicit stability.

Command: python -m newton.labstuff.tissues [--tissue TYPE]

References:
- Smith et al. 2018. Stable Neo-Hookean Flesh Simulation. ACM Trans. Graph.
- Macklin & Muller 2021. Constraint-based Stable Neo-Hookean Materials. MIG.
- Macklin et al. 2016. XPBD: Position-Based Simulation of Compliant
  Constrained Dynamics. MIG.
"""

import numpy as np
import warp as wp

import newton
import newton.examples


# ============================================================================
# Tissue Material Properties
#
# Real soft tissue is nearly incompressible: nu ~ 0.4999 (Glozman & Azhari 2010).
# Force-based solvers (SemiImplicit) lose convergence at high nu (Macklin 2021,
# Table 1 shows failure at nu=0.49 for Newton methods). We use nu=0.495 as a
# practical limit -- high enough for good volume preservation, low enough for
# semi-implicit stability. Switching to SolverXPBD would allow nu -> 0.5.
#
# Young's modulus values from literature:
#   Fat:    0.5-5 kPa   (Krouskop 1998, Samani 2007)
#   Liver:  1-6 kPa     (Yeh 2002)
#   Muscle: 6-100 kPa   (relaxed vs contracted, Duck 1990)
#   Skin:   50-100 kPa  (Agache 1980, depends on body site)
# ============================================================================
TISSUE_PROPERTIES = {
    "fat": {
        "young_modulus": 3_000.0,      # Pa (soft adipose)
        "poisson_ratio": 0.495,        # nearly incompressible
        "density": 950.0,              # kg/m^3
        "damping": 50.0,
    },
    "liver": {
        "young_modulus": 5_000.0,      # Pa (parenchymal organ)
        "poisson_ratio": 0.495,
        "density": 1060.0,
        "damping": 80.0,
    },
    "muscle_relaxed": {
        "young_modulus": 20_000.0,     # Pa
        "poisson_ratio": 0.495,
        "density": 1050.0,
        "damping": 100.0,
    },
    "muscle_contracted": {
        "young_modulus": 80_000.0,     # Pa (4x stiffer when activated)
        "poisson_ratio": 0.495,
        "density": 1050.0,
        "damping": 150.0,
    },
    "skin": {
        "young_modulus": 60_000.0,     # Pa (dermis layer)
        "poisson_ratio": 0.490,        # slightly less than deep tissue
        "density": 1100.0,
        "damping": 120.0,
    },
    "generic_soft_tissue": {
        "young_modulus": 15_000.0,     # Pa (moderate stiffness)
        "poisson_ratio": 0.495,
        "density": 1000.0,
        "damping": 80.0,
    },
}


def compute_lame_parameters(young_modulus: float, poisson_ratio: float) -> tuple[float, float]:
    """Convert Young's modulus and Poisson's ratio to Lamé parameters.

    Returns (k_mu, k_lambda) where:
      k_mu     = shear modulus (resists shape distortion)
      k_lambda = bulk modulus  (resists volume change)

    Newton's FEM kernel uses the rest-stability correction from Smith et al. 2018:
      gamma = 1 + k_mu/k_lambda - k_mu/(4*k_lambda)
    so the hydrostatic energy becomes (lambda/2)(J - gamma)^2, ensuring zero
    force at rest configuration.
    """
    # Cap at 0.4999 for numerical safety in the semi-implicit solver.
    # XPBD can handle 0.5 (see Macklin 2021, Table 1).
    nu = min(poisson_ratio, 0.4999)
    k_mu = 0.5 * young_modulus / (1.0 + nu)
    k_lambda = young_modulus * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return k_mu, k_lambda


# ============================================================================
# Surface template helpers
# ============================================================================

def _smoothstep(edge0, edge1, x):
    """Hermite interpolation: 0 at edge0, 1 at edge1."""
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def build_heightfield(u_norm, template, params):
    """Return per-particle height offset (meters) for the given template."""
    if template == "sine":
        N = params.get("N", 2)
        A = params.get("A", 0.02)
        return A * np.sin(2.0 * np.pi * N * u_norm)
    elif template == "cosine":
        N = params.get("N", 2)
        A = params.get("A", 0.02)
        return A * np.cos(2.0 * np.pi * N * u_norm)
    elif template == "linear":
        h0 = params.get("h0", 0.0)
        h1 = params.get("h1", 0.02)
        return h0 + (h1 - h0) * u_norm
    elif template == "polynomial":
        coeffs = params.get("coeffs", [0.0, 0.0, 0.0])
        degree = len(coeffs) - 1
        result = np.zeros_like(u_norm, dtype=np.float64)
        for i, c in enumerate(coeffs):
            result += c * u_norm ** (degree - i)
        return result.astype(np.float32)
    elif template == "exponential":
        h0 = params.get("h0", 0.0)
        h1 = params.get("h1", 0.02)
        k = params.get("k", 4.0)
        denom = np.exp(k) - 1.0
        if abs(denom) < 1e-12:
            g = u_norm
        else:
            g = (np.exp(k * u_norm) - 1.0) / denom
        return h0 + (h1 - h0) * g
    return np.zeros_like(u_norm, dtype=np.float32)


def apply_surface_from_template(state, template, direction, params,
                                preview=False):
    """Apply surface heightfield to a state using auto-computed mapping.

    Mapping (origin, length) is derived from the top-face particles so the
    function always spans the full slab along the chosen direction.
    Top-face particles receive full displacement (w=1.0); particles below
    taper smoothly over two cell layers to zero.
    """
    q = state.particle_q.numpy()
    top_z = float(q[:, 2].max())
    if top_z < 1e-8:
        return

    # Choose u axis: 0 = X, 1 = Y
    u_axis = 0 if direction == "X" else 1

    # Identify top-face particles (generous tolerance for FP jitter)
    face_tol = 0.01
    top_face_z = top_z - face_tol
    top_mask = q[:, 2] > top_face_z

    # Auto-compute mapping bounds from top-face particles
    u_top = q[top_mask, u_axis]
    u_min = float(u_top.min())
    u_max = float(u_top.max())
    L = u_max - u_min
    if L < 1e-6:
        L = 1.0

    # u_norm for ALL particles, clamped [0, 1]
    u_norm = np.clip((q[:, u_axis] - u_min) / L, 0.0, 1.0)

    # Height offsets from template (meters)
    dz_raw = build_heightfield(u_norm, template, params)

    # Weight: top face = 1.0, taper smoothly over 2 cell layers below
    taper = 0.12
    w = np.where(
        q[:, 2] >= top_face_z,
        1.0,
        _smoothstep(top_face_z - taper, top_face_z, q[:, 2]),
    )
    dz = w * dz_raw
    q[:, 2] += dz

    # Write back to warp array
    device = state.particle_q.device
    q_wp = wp.array(q, dtype=wp.vec3, device=device)
    try:
        wp.copy(state.particle_q, q_wp)
    except Exception:
        state.particle_q = q_wp

    if preview:
        top_z_after = float(q[:, 2].max())
        print(f"  Surface: template={template}, direction={direction}")
        print(f"  top_z: {top_z:.4f} -> {top_z_after:.4f}")
        print(f"  dz min/max: {dz.min():.6f} / {dz.max():.6f}")


def prompt_surface_picker():
    """Interactive surface template picker.

    Returns (template, direction, params) or None for flat.
    Default (Enter through all): sine, X, N=2, A=0.02.
    """
    print("\nSurface template:")
    print("  0) flat")
    print("  1) sine")
    print("  2) cosine")
    print("  3) linear")
    print("  4) polynomial")
    print("  5) exponential")

    try:
        raw = input("\nTemplate [1]: ").strip()
    except EOFError:
        return None

    if raw == "0":
        return None
    choice = raw if raw in ("1", "2", "3", "4", "5") else "1"
    template_map = {
        "1": "sine", "2": "cosine", "3": "linear",
        "4": "polynomial", "5": "exponential",
    }
    template = template_map[choice]

    # Direction (X or Y only)
    print("\nU direction:")
    print("  X) along X (left-right)")
    print("  Y) along Y (front-back)")
    try:
        raw = input("Direction [X]: ").strip().upper()
    except EOFError:
        raw = ""
    direction = raw if raw in ("X", "Y") else "X"

    # Template-specific parameters
    params = {}

    if template in ("sine", "cosine"):
        try:
            raw = input("Number of bumps N [2]: ").strip()
            params["N"] = int(raw) if raw else 2
        except Exception:
            params["N"] = 2
        try:
            raw = input("Amplitude A in meters [0.02]: ").strip()
            params["A"] = float(raw) if raw else 0.02
        except Exception:
            params["A"] = 0.02

    elif template == "linear":
        try:
            raw = input("Start height h0 in meters [0.0]: ").strip()
            params["h0"] = float(raw) if raw else 0.0
        except Exception:
            params["h0"] = 0.0
        try:
            raw = input("End height h1 in meters [0.02]: ").strip()
            params["h1"] = float(raw) if raw else 0.02
        except Exception:
            params["h1"] = 0.02

    elif template == "polynomial":
        try:
            raw = input("Degree (2 or 3) [2]: ").strip()
            degree = int(raw) if raw else 2
        except Exception:
            degree = 2
        degree = max(2, min(3, degree))
        if degree == 2:
            try:
                raw = input("Coefficients a,b,c [0,0,0]: ").strip()
                coeffs = [float(x) for x in raw.split(",")] if raw else [0.0, 0.0, 0.0]
            except Exception:
                coeffs = [0.0, 0.0, 0.0]
            while len(coeffs) < 3:
                coeffs.append(0.0)
            params["coeffs"] = coeffs[:3]
        else:
            try:
                raw = input("Coefficients a,b,c,d [0,0,0,0]: ").strip()
                coeffs = [float(x) for x in raw.split(",")] if raw else [0.0, 0.0, 0.0, 0.0]
            except Exception:
                coeffs = [0.0, 0.0, 0.0, 0.0]
            while len(coeffs) < 4:
                coeffs.append(0.0)
            params["coeffs"] = coeffs[:4]

    elif template == "exponential":
        try:
            raw = input("Start height h0 in meters [0.0]: ").strip()
            params["h0"] = float(raw) if raw else 0.0
        except Exception:
            params["h0"] = 0.0
        try:
            raw = input("End height h1 in meters [0.02]: ").strip()
            params["h1"] = float(raw) if raw else 0.02
        except Exception:
            params["h1"] = 0.02
        try:
            raw = input("Shape k [4.0]: ").strip()
            params["k"] = float(raw) if raw else 4.0
        except Exception:
            params["k"] = 4.0

    return template, direction, params


def _recompute_tet_poses(model, state):
    """Recompute tet rest-pose matrices (Dm_inv) from current particle positions.

    This makes the displaced configuration stress-free so the FEM forces
    do not pull the surface back to its original flat shape.
    """
    q = state.particle_q.numpy()
    idx = model.tet_indices.numpy()                  # (num_tets, 4) int32
    n_tets = model.tet_count
    new_poses = np.zeros((n_tets, 3, 3), dtype=np.float32)
    for ti in range(n_tets):
        i, j, k, l = idx[ti]
        p0, p1, p2, p3 = q[i], q[j], q[k], q[l]
        Dm = np.array([p1 - p0, p2 - p0, p3 - p0], dtype=np.float64).T
        new_poses[ti] = np.linalg.inv(Dm).astype(np.float32)
    device = model.tet_poses.device
    model.tet_poses = wp.array(new_poses, dtype=wp.mat33, device=device)


@wp.kernel
def apply_probe_contact(
    particle_q: wp.array(dtype=wp.vec3),
    particle_f: wp.array(dtype=wp.vec3),
    probe_xy: wp.vec3,
    radius: float,
    force_mag: float,
):
    tid = wp.tid()
    pos = particle_q[tid]
    dx = pos[0] - probe_xy[0]
    dy = pos[1] - probe_xy[1]
    dist_sq = dx * dx + dy * dy
    radius_sq = radius * radius
    if dist_sq < radius_sq:
        falloff = 1.0 - dist_sq / radius_sq
        force = wp.vec3(0.0, 0.0, -force_mag * falloff)
        wp.atomic_add(particle_f, tid, force)


@wp.kernel
def set_probe_kinematic(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    idx: int,
    pos: wp.vec3,
):
    if wp.tid() == 0:
        body_q[idx] = wp.transform(pos, wp.quat_identity())
        body_qd[idx] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class Example:
    """
    Hyper-realistic human tissue simulation using Newton's FEM system.

    Creates a deformable tissue block that responds to gravity and can be
    interacted with through the viewer (click and drag to apply forces).
    """

    def __init__(self, viewer, tissue_type="generic_soft_tissue",
                 surface_template=None, surface_direction="X",
                 surface_params=None, surface_preview=False):
        # Simulation parameters
        # Higher Poisson ratios produce stiffer volumetric forces (k_lambda),
        # requiring smaller dt for semi-implicit stability. At nu=0.495 with
        # E=15kPa, k_lambda ~ 150kPa, so we need more substeps than typical.
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 32
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.tissue_type = tissue_type

        # Get tissue properties
        props = TISSUE_PROPERTIES[tissue_type]
        k_mu, k_lambda = compute_lame_parameters(
            props["young_modulus"],
            props["poisson_ratio"]
        )

        # Build the model
        builder = newton.ModelBuilder()
        builder.default_particle_radius = 0.01

        # Tissue dimensions - a visible slab of meat
        dim_x, dim_y, dim_z = 12, 6, 12  # cells (width, height, depth)
        cell_size = 0.025  # 2.5cm per cell -> 30cm x 15cm x 30cm slab

        # Calculate particle density from material density
        total_volume = (dim_x * cell_size) * (dim_y * cell_size) * (dim_z * cell_size)
        total_mass = props["density"] * total_volume
        num_particles = (dim_x + 1) * (dim_y + 1) * (dim_z + 1)
        particle_mass = total_mass / num_particles
        cell_volume = cell_size ** 3
        particle_density = particle_mass / cell_volume

        # Position: centered in x/y, sitting on ground (z=0)
        tissue_width = dim_x * cell_size
        tissue_depth = dim_z * cell_size

        # Add soft tissue grid (tetrahedral FEM)
        # Rotate to lie flat (thin dimension becomes vertical)
        tissue_height = dim_y * cell_size
        builder.add_soft_grid(
            pos=wp.vec3(-tissue_width / 2, -tissue_depth / 2, tissue_height),
            rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -wp.pi / 2.0),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=dim_x,
            dim_y=dim_y,
            dim_z=dim_z,
            cell_x=cell_size,
            cell_y=cell_size,
            cell_z=cell_size,
            density=particle_density,
            k_mu=k_mu,
            k_lambda=k_lambda,
            k_damp=props["damping"],
            tri_ke=0.0,
            tri_ka=0.0,
            tri_kd=0.0,
            tri_drag=0.0,
            tri_lift=0.0,
            fix_top=True,  # After -90° rotation, grid top = world bottom (z=0)
        )

        # Add ground plane with soft contact properties
        ke = 1.0e4
        kd = 10.0
        kf = 0.0
        mu = 0.5
        builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(ke=ke, kd=kd, kf=kf, mu=mu)
        )

        # Probe: sphere pressed down onto tissue (like a finger tip)
        self.probe_radius = 0.02  # 2cm sphere
        self.probe_surface_z = tissue_height + self.probe_radius
        self.probe_rest_z = self.probe_surface_z + 0.03  # hover 3cm above
        self.probe_z = self.probe_rest_z

        self.probe_id = builder.add_body(
            xform=wp.transform(
                p=wp.vec3(0.0, 0.0, self.probe_rest_z),
                q=wp.quat_identity(),
            ),
        )
        builder.add_shape_sphere(
            self.probe_id,
            radius=self.probe_radius,
            cfg=newton.ModelBuilder.ShapeConfig(ke=ke, kd=kd, kf=kf, mu=mu, density=0.0),
        )

        # Finalize model
        self.model = builder.finalize()

        # Set soft contact parameters
        self.model.soft_contact_ke = ke
        self.model.soft_contact_kd = kd
        self.model.soft_contact_kf = kf
        self.model.soft_contact_mu = mu
        self.model.soft_contact_restitution = 0.1

        # Create solver (SemiImplicit for soft bodies)
        self.solver = newton.solvers.SolverSemiImplicit(self.model)

        # Allocate states (double buffered)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # Apply surface heightfield if a template was chosen
        if surface_template is not None:
            _params = surface_params or {}
            apply_surface_from_template(
                self.state_0, surface_template, surface_direction,
                _params, preview=surface_preview,
            )
            apply_surface_from_template(
                self.state_1, surface_template, surface_direction,
                _params, preview=False,
            )
            # Recompute FEM rest poses so displaced shape is stress-free
            _recompute_tet_poses(self.model, self.state_0)

        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0, soft_contact_margin=0.02)

        # Probe controls
        self.poke_active = False
        self.depth_levels = [2.0, 5.0, 10.0, 15.0, 20.0, 30.0]  # mm
        self.depth_index = 2  # Start at 10mm

        # Find initial top surface for deformation measurement
        q = self.state_0.particle_q.numpy()
        self.initial_top_z = float(q[:, 2].max())
        top_mask = q[:, 2] > (self.initial_top_z - 0.01)
        self.top_particle_indices = np.where(top_mask)[0]

        # Key state tracking for press detection
        self._prev_keys = {"p": False, "up": False, "down": False}
        self._frame_count = 0

        # Setup viewer
        self.viewer.set_model(self.model)

        # Print info
        gamma = 1.0 + k_mu / k_lambda - k_mu / (4.0 * k_lambda)
        print(f"Tissue Simulation:")
        print(f"  Type:           {tissue_type}")
        print(f"  Particles:      {self.model.particle_count}")
        print(f"  Tetrahedra:     {self.model.tet_count}")
        print(f"  E (Young's):    {props['young_modulus']:,.0f} Pa")
        print(f"  nu (Poisson):   {props['poisson_ratio']}")
        print(f"  k_mu (shear):   {k_mu:,.1f} Pa")
        print(f"  k_lambda (bulk):{k_lambda:,.1f} Pa")
        print(f"  gamma (rest):   {gamma:.4f}")
        print(f"  Substeps:       {self.sim_substeps}")
        print(f"  dt:             {self.sim_dt:.6f} s")
        print(f"\nControls:")
        print(f"  P          Toggle probe down/up")
        print(f"  UP/DOWN    Change probe depth")
        print(f"  Depth:     {self.depth_levels[self.depth_index]:.0f}mm")

        # Capture for GPU acceleration
        self.capture()

    def capture(self):
        # Dynamic poke forces require direct execution (no graph capture)
        self.graph = None

    def simulate(self):
        """Run simulation substeps."""
        probe_pos = wp.vec3(0.0, 0.0, self.probe_z)

        # Contact force scales with probe penetration into tissue
        probe_bottom = self.probe_z - self.probe_radius
        penetration = max(0.0, self.initial_top_z - probe_bottom)
        force_mag = 1000.0 * penetration  # ke * depth
        contact_radius = 0.075  # 7.5cm influence zone

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # Pin probe at current position (kinematic body)
            wp.launch(
                kernel=set_probe_kinematic,
                dim=1,
                inputs=[
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.probe_id,
                    probe_pos,
                ],
            )

            # Apply contact force from probe to tissue particles
            if force_mag > 0.0:
                wp.launch(
                    kernel=apply_probe_contact,
                    dim=self.model.particle_count,
                    inputs=[
                        self.state_0.particle_q,
                        self.state_0.particle_f,
                        probe_pos,
                        contact_radius,
                        force_mag,
                    ],
                )

            self.viewer.apply_forces(self.state_0)
            self.contacts = self.model.collide(self.state_0, soft_contact_margin=0.02)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        """Advance simulation by one frame."""
        self._check_keys()
        self._animate_probe()

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

        # Print deformation periodically when probing
        if self.poke_active:
            self._frame_count += 1
            if self._frame_count % 10 == 0:
                indent_mm = self._measure_deformation()
                depth_mm = self.depth_levels[self.depth_index]
                print(f"\r  Depth: {depth_mm:4.0f}mm | Indent: {indent_mm:6.1f}mm  ", end="", flush=True)

    def _animate_probe(self):
        """Smoothly move probe toward target position."""
        if self.poke_active:
            target_z = self.probe_surface_z - self.depth_levels[self.depth_index] / 1000.0
        else:
            target_z = self.probe_rest_z

        speed = 0.3  # m/s
        max_move = speed * self.frame_dt
        delta = target_z - self.probe_z
        if abs(delta) > max_move:
            self.probe_z += max_move * (1.0 if delta > 0 else -1.0)
        else:
            self.probe_z = target_z

    def _check_keys(self):
        """Detect key presses for probe controls."""
        if not hasattr(self.viewer, "is_key_down"):
            return

        keys = {k: self.viewer.is_key_down(k) for k in self._prev_keys}

        # P: toggle probe down/up
        if keys["p"] and not self._prev_keys["p"]:
            self.poke_active = not self.poke_active
            if self.poke_active:
                self._frame_count = 0
                print(f"\nProbe DOWN | Depth: {self.depth_levels[self.depth_index]:.0f}mm")
            else:
                print(f"\nProbe UP")

        # UP: push deeper
        if keys["up"] and not self._prev_keys["up"]:
            if self.depth_index < len(self.depth_levels) - 1:
                self.depth_index += 1
                print(f"\n  Depth: {self.depth_levels[self.depth_index]:.0f}mm")

        # DOWN: push less
        if keys["down"] and not self._prev_keys["down"]:
            if self.depth_index > 0:
                self.depth_index -= 1
                print(f"\n  Depth: {self.depth_levels[self.depth_index]:.0f}mm")

        self._prev_keys = keys

    def _measure_deformation(self):
        """Measure indentation depth (mm) of top surface near probe."""
        q = self.state_0.particle_q.numpy()
        top_q = q[self.top_particle_indices]

        # Particles near probe center (x=0, y=0)
        measure_radius = self.probe_radius * 3.0
        dist_sq = top_q[:, 0] ** 2 + top_q[:, 1] ** 2
        near_probe = dist_sq < measure_radius ** 2

        if near_probe.any():
            indent = self.initial_top_z - float(top_q[near_probe, 2].min())
            return indent * 1000.0  # mm
        return 0.0

    def render(self):
        """Render current state."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    # Create argument parser
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--tissue", type=str, default="generic_soft_tissue",
        choices=list(TISSUE_PROPERTIES.keys()),
        help="Type of tissue to simulate",
    )
    parser.add_argument("--surface_preview", action="store_true",
                        help="Print debug stats for applied surface offsets")

    # Initialize viewer
    viewer, args = newton.examples.init(parser)

    # Interactive surface template picker
    result = prompt_surface_picker()

    surface_kwargs = {}
    if result is not None:
        template, direction, params = result
        surface_kwargs = dict(
            surface_template=template,
            surface_direction=direction,
            surface_params=params,
        )

    # Create example
    example = Example(
        viewer,
        tissue_type=args.tissue,
        surface_preview=args.surface_preview,
        **surface_kwargs,
    )

    # Run example
    newton.examples.run(example, args)
