"""
Tissue Simulation (Stable Neo-Hookean FEM)

Deformable tissue block using Newton's tetrahedral FEM with the Stable
Neo-Hookean constitutive model:

  Psi = (lambda/2)(J - gamma)^2 + (mu/2)(tr(F^T F) - 3)

where gamma = 1 + mu/lambda ensures rest stability (Smith et al. 2018).

Real soft tissue is nearly incompressible (nu ~ 0.4999, Glozman & Azhari 2010).
Force-based solvers lose accuracy at high Poisson ratios, so we cap at nu=0.495.

Usage:
  python -m newton.labstuff.tissues [--tissue TYPE]

References:
- Smith et al. 2018. Stable Neo-Hookean Flesh Simulation. ACM Trans. Graph.
- Macklin & Muller 2021. Constraint-based Stable Neo-Hookean Materials. MIG.
"""

import numpy as np
import warp as wp

import newton
import newton.examples


# ============================================================================
# Tissue material library
# ============================================================================

TISSUE_PROPERTIES = {
    "fat": {
        "young_modulus": 3_000.0,
        "poisson_ratio": 0.495,
        "density": 950.0,
        "damping": 50.0,
    },
    "liver": {
        "young_modulus": 5_000.0,
        "poisson_ratio": 0.495,
        "density": 1060.0,
        "damping": 80.0,
    },
    "muscle_relaxed": {
        "young_modulus": 20_000.0,
        "poisson_ratio": 0.495,
        "density": 1050.0,
        "damping": 100.0,
    },
    "muscle_contracted": {
        "young_modulus": 80_000.0,
        "poisson_ratio": 0.495,
        "density": 1050.0,
        "damping": 150.0,
    },
    "skin": {
        "young_modulus": 60_000.0,
        "poisson_ratio": 0.490,
        "density": 1100.0,
        "damping": 120.0,
    },
    "generic_soft_tissue": {
        "young_modulus": 15_000.0,
        "poisson_ratio": 0.495,
        "density": 1000.0,
        "damping": 80.0,
    },
}


def compute_lame_parameters(young_modulus: float, poisson_ratio: float) -> tuple[float, float]:
    """Return (k_mu, k_lambda) from Young's modulus and Poisson's ratio."""
    nu = min(poisson_ratio, 0.4999)
    k_mu = 0.5 * young_modulus / (1.0 + nu)
    k_lambda = young_modulus * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return k_mu, k_lambda


# ============================================================================
# Surface template system
# ============================================================================

def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """Hermite interpolation returning 0 at edge0, 1 at edge1."""
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def build_heightfield(u_norm: np.ndarray, template: str, params: dict) -> np.ndarray:
    """Return per-particle Z offset (meters) for the chosen surface template."""
    if template == "sine":
        N = params.get("N", 2)
        A = params.get("A", 0.02)
        return A * np.sin(2.0 * np.pi * N * u_norm)
    if template == "cosine":
        N = params.get("N", 2)
        A = params.get("A", 0.02)
        return A * np.cos(2.0 * np.pi * N * u_norm)
    if template == "linear":
        h0 = params.get("h0", 0.0)
        h1 = params.get("h1", 0.02)
        return h0 + (h1 - h0) * u_norm
    if template == "polynomial":
        coeffs = params.get("coeffs", [0.0, 0.0, 0.0])
        degree = len(coeffs) - 1
        result = np.zeros_like(u_norm, dtype=np.float64)
        for i, c in enumerate(coeffs):
            result += c * u_norm ** (degree - i)
        return result.astype(np.float32)
    if template == "exponential":
        h0 = params.get("h0", 0.0)
        h1 = params.get("h1", 0.02)
        k = params.get("k", 4.0)
        denom = np.exp(k) - 1.0
        g = (np.exp(k * u_norm) - 1.0) / denom if abs(denom) > 1e-12 else u_norm
        return h0 + (h1 - h0) * g
    return np.zeros_like(u_norm, dtype=np.float32)


def apply_surface_from_template(state, template: str, direction: str, params: dict,
                                preview: bool = False) -> None:
    """Deform the top surface of the tissue slab with the chosen heightfield.

    Only the top band of particles (world Z near max) receives the offset.
    A smoothstep taper over 0.12 m blends the deformation to zero below the top face.
    """
    q = state.particle_q.numpy()
    top_z = float(q[:, 2].max())
    if top_z < 1e-6:
        return

    u_axis = 0 if direction == "X" else 1
    face_tol = 0.01
    top_face_z = top_z - face_tol
    top_mask = q[:, 2] > top_face_z

    u_vals = q[top_mask, u_axis]
    u_min, u_max = float(u_vals.min()), float(u_vals.max())
    L = u_max - u_min
    if L < 1e-6:
        L = 1.0
    u_norm = np.clip((q[:, u_axis] - u_min) / L, 0.0, 1.0)

    dz_raw = build_heightfield(u_norm, template, params)

    taper = 0.12
    w = np.where(
        q[:, 2] >= top_face_z,
        1.0,
        _smoothstep(top_face_z - taper, top_face_z, q[:, 2]),
    )
    q[:, 2] += w * dz_raw

    device = state.particle_q.device
    q_wp = wp.array(q, dtype=wp.vec3, device=device)
    try:
        wp.copy(state.particle_q, q_wp)
    except Exception:
        state.particle_q = q_wp

    if preview:
        print(f"  surface={template} dir={direction}  "
              f"top_z: {top_z:.4f} → {float(q[:, 2].max()):.4f}  "
              f"dz in [{(w * dz_raw).min():.4f}, {(w * dz_raw).max():.4f}]")


def prompt_surface_picker() -> tuple | None:
    """Interactive terminal prompt.  Returns (template, direction, params) or None for flat."""
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
    template = {"1": "sine", "2": "cosine", "3": "linear",
                "4": "polynomial", "5": "exponential"}[choice]

    print("\nU direction:  X) along X (left-right)  Y) along Y (front-back)")
    try:
        raw = input("Direction [X]: ").strip().upper()
    except EOFError:
        raw = ""
    direction = raw if raw in ("X", "Y") else "X"

    params: dict = {}
    if template in ("sine", "cosine"):
        try:
            raw = input("Bumps N [2]: ").strip()
            params["N"] = int(raw) if raw else 2
        except Exception:
            params["N"] = 2
        try:
            raw = input("Amplitude A meters [0.02]: ").strip()
            params["A"] = float(raw) if raw else 0.02
        except Exception:
            params["A"] = 0.02
    elif template == "linear":
        try:
            params["h0"] = float(input("h0 meters [0.0]: ").strip() or "0.0")
        except Exception:
            params["h0"] = 0.0
        try:
            params["h1"] = float(input("h1 meters [0.02]: ").strip() or "0.02")
        except Exception:
            params["h1"] = 0.02
    elif template == "polynomial":
        try:
            deg = int(input("Degree 2 or 3 [2]: ").strip() or "2")
        except Exception:
            deg = 2
        deg = max(2, min(3, deg))
        n = deg + 1
        try:
            raw = input(f"Coefficients ({n} floats, comma-sep) [{', '.join(['0.0']*n)}]: ").strip()
            coeffs = [float(v) for v in raw.split(",")] if raw else [0.0] * n
        except Exception:
            coeffs = [0.0] * n
        while len(coeffs) < n:
            coeffs.append(0.0)
        params["coeffs"] = coeffs[:n]
    elif template == "exponential":
        try:
            params["h0"] = float(input("h0 meters [0.0]: ").strip() or "0.0")
        except Exception:
            params["h0"] = 0.0
        try:
            params["h1"] = float(input("h1 meters [0.02]: ").strip() or "0.02")
        except Exception:
            params["h1"] = 0.02
        try:
            params["k"] = float(input("Shape k [4.0]: ").strip() or "4.0")
        except Exception:
            params["k"] = 4.0

    return template, direction, params


def _recompute_tet_poses(model, state) -> None:
    """Recompute Dm_inv for each tet from current positions, making the current
    configuration stress-free.  Must be called after any surface deformation."""
    q = state.particle_q.numpy()
    idx = model.tet_indices.numpy()
    n_tets = model.tet_count
    new_poses = np.zeros((n_tets, 3, 3), dtype=np.float32)
    for ti in range(n_tets):
        i0, i1, i2, i3 = idx[ti]
        p0, p1, p2, p3 = q[i0], q[i1], q[i2], q[i3]
        Dm = np.array([p1 - p0, p2 - p0, p3 - p0], dtype=np.float64).T
        new_poses[ti] = np.linalg.inv(Dm).astype(np.float32)
    model.tet_poses = wp.array(new_poses, dtype=wp.mat33, device=model.tet_poses.device)


# ============================================================================
# Warp kernels
# ============================================================================

@wp.kernel
def apply_probe_contact(
    particle_q: wp.array(dtype=wp.vec3),
    particle_f: wp.array(dtype=wp.vec3),
    probe_center: wp.vec3,
    radius: float,
    force_mag: float,
):
    """Push particles inside the contact radius downward."""
    tid = wp.tid()
    pos = particle_q[tid]
    dx = pos[0] - probe_center[0]
    dy = pos[1] - probe_center[1]
    dist_sq = dx * dx + dy * dy
    if dist_sq < radius * radius:
        falloff = 1.0 - dist_sq / (radius * radius)
        wp.atomic_add(particle_f, tid, wp.vec3(0.0, 0.0, -force_mag * falloff))


@wp.kernel
def set_probe_kinematic(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    idx: int,
    pos: wp.vec3,
):
    """Pin a rigid body to a given world position (zero velocity)."""
    if wp.tid() == 0:
        body_q[idx] = wp.transform(pos, wp.quat_identity())
        body_qd[idx] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


# ============================================================================
# Example class
# ============================================================================

class Example:
    """Deformable tissue slab with interactive surface template and a poke probe."""

    def __init__(self, viewer, args):
        # ---- timing ----
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 32
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        # ---- tissue material ----
        tissue_type = getattr(args, "tissue", "generic_soft_tissue")
        props = TISSUE_PROPERTIES[tissue_type]
        k_mu, k_lambda = compute_lame_parameters(props["young_modulus"], props["poisson_ratio"])

        # ---- grid parameters ----
        # cell_size=0.04 gives smooth sine surface (16 sample points per 0.60 m
        # direction → ~8 points per bump at N=2) while staying well under 35k particles.
        cell_size = 0.04
        dim_x = 15   # world X: 15*0.04 = 0.60 m  (width)
        dim_y = 5    # local Y: 5*0.04 = 0.20 m  → world Z height after -90° X rotation
        dim_z = 15   # local Z: 15*0.04 = 0.60 m  → world Y depth  after -90° X rotation

        assert (dim_x + 1) * (dim_y + 1) * (dim_z + 1) <= 35_000, "too many particles"

        tissue_width  = dim_x * cell_size   # 0.60 m  (world X)
        tissue_height = dim_y * cell_size   # 0.20 m  (world Z)
        tissue_depth  = dim_z * cell_size   # 0.60 m  (world Y)

        # ---- density → particle density ----
        total_volume = tissue_width * tissue_height * tissue_depth
        total_mass = props["density"] * total_volume
        n_particles = (dim_x + 1) * (dim_y + 1) * (dim_z + 1)
        particle_density = (total_mass / n_particles) / (cell_size ** 3)

        # ---- model builder ----
        builder = newton.ModelBuilder()
        # particle radius must scale with cell size to avoid overlapping blobs
        builder.default_particle_radius = 0.25 * cell_size

        # Placement of the soft grid.
        # The grid is built with a -90° rotation around X so that:
        #   local X (dim_x) → world X
        #   local Y (dim_y) → world Z  (the thin "height" direction, negated by rotation)
        #   local Z (dim_z) → world Y  (the "depth" direction)
        #
        # With rot = R_x(-pi/2):  (x,y,z)_local → (x, z, -y)_world
        # So particle (i, j, k) lands at world position:
        #   x = (cx - Wx/2) + i*cell
        #   y = (cy - Wz/2) + k*cell
        #   z = (tissue_height + 0.01) - j*cell
        #
        # j = 0       → z = tissue_height + 0.01  (TOP of slab)
        # j = dim_y   → z = 0.01                  (BOTTOM, just above ground)
        #
        # fix_top=True pins j == dim_y (the world-bottom), providing a static foundation.
        cx, cy = 0.0, 1.0   # XY center: X=0, Y=1.0 keeps tissue away from camera origin
        builder.add_soft_grid(
            pos=wp.vec3(cx - tissue_width / 2,
                        cy - tissue_depth / 2,
                        tissue_height + 0.01),
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
            fix_top=True,  # j=dim_y → world z=0.01, static foundation
        )

        # ---- ground plane ----
        ke, kd, kf, mu = 1.0e4, 10.0, 0.0, 0.5
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(ke=ke, kd=kd, kf=kf, mu=mu))

        # ---- probe sphere (placeholder position; corrected below) ----
        self.probe_radius = 0.02   # 2 cm — finger-tip scale
        self.probe_id = builder.add_body(
            xform=wp.transform(p=wp.vec3(cx, cy, 1.0), q=wp.quat_identity()),
        )
        builder.add_shape_sphere(
            self.probe_id,
            radius=self.probe_radius,
            cfg=newton.ModelBuilder.ShapeConfig(ke=ke, kd=kd, kf=kf, mu=mu, density=0.0),
        )

        # ---- finalize ----
        self.model = builder.finalize()
        self.model.soft_contact_ke = ke
        self.model.soft_contact_kd = kd
        self.model.soft_contact_kf = kf
        self.model.soft_contact_mu = mu
        self.model.soft_contact_restitution = 0.1

        self.solver = newton.solvers.SolverSemiImplicit(self.model)

        # ---- states ----
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # ---- interactive surface picker ----
        surface_result = prompt_surface_picker()
        if surface_result is not None:
            tmpl, direction, params = surface_result
            preview = getattr(args, "surface_preview", False)
            apply_surface_from_template(self.state_0, tmpl, direction, params, preview=preview)
            apply_surface_from_template(self.state_1, tmpl, direction, params, preview=False)
            # Make deformed shape stress-free by resetting rest poses
            _recompute_tet_poses(self.model, self.state_0)

        # ---- probe placement from actual particle positions ----
        q0 = self.state_0.particle_q.numpy()
        bbox_min = q0.min(axis=0)
        bbox_max = q0.max(axis=0)
        top_z = float(bbox_max[2])

        self.probe_xy = (cx, cy)
        self.probe_surface_z = top_z + self.probe_radius
        self.probe_rest_z = self.probe_surface_z + 0.05  # hover 5 cm above surface
        self.probe_z = self.probe_rest_z

        probe_init = wp.vec3(cx, cy, self.probe_rest_z)
        wp.launch(set_probe_kinematic, dim=1,
                  inputs=[self.state_0.body_q, self.state_0.body_qd, self.probe_id, probe_init])
        wp.launch(set_probe_kinematic, dim=1,
                  inputs=[self.state_1.body_q, self.state_1.body_qd, self.probe_id, probe_init])

        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0, soft_contact_margin=0.02)

        # ---- probe interaction state ----
        self.poke_active = False
        self.depth_levels = [2.0, 5.0, 10.0, 15.0, 20.0, 30.0]  # mm
        self.depth_index = 2  # start at 10 mm

        # Top-face particles for deformation readout
        self.initial_top_z = top_z
        top_mask = q0[:, 2] > (top_z - 0.01)
        self.top_particle_indices = np.where(top_mask)[0]

        self._prev_keys = {"p": False, "up": False, "down": False}
        self._frame_count = 0

        # ---- viewer ----
        self.viewer.set_model(self.model)

        # ---- diagnostics ----
        dx = float(bbox_max[0] - bbox_min[0])
        dy = float(bbox_max[1] - bbox_min[1])
        dz = float(bbox_max[2] - bbox_min[2])
        print(f"[tissue] type={tissue_type}")
        print(f"[tissue] cell_size={cell_size}  particle_radius={builder.default_particle_radius:.4f}")
        print(f"[tissue] dim_x/dim_y/dim_z = {dim_x}/{dim_y}/{dim_z}  "
              f"particles={self.model.particle_count}  tets={self.model.tet_count}")
        print(f"[tissue] bbox_min=({bbox_min[0]:.3f}, {bbox_min[1]:.3f}, {bbox_min[2]:.3f})")
        print(f"[tissue] bbox_max=({bbox_max[0]:.3f}, {bbox_max[1]:.3f}, {bbox_max[2]:.3f})")
        print(f"[tissue] spans: dx={dx:.3f}  dy={dy:.3f}  dz={dz:.3f}")
        print(f"[tissue] top_z={top_z:.3f}  probe_rest_z={self.probe_rest_z:.3f}")
        print(f"[tissue] probe_xy=({cx:.3f}, {cy:.3f})")
        print("\nControls:  P = toggle probe  UP/DOWN = change depth")
        print(f"           current depth = {self.depth_levels[self.depth_index]:.0f} mm")

        self.capture()

    # ------------------------------------------------------------------

    def capture(self):
        # Dynamic force kernels prevent graph capture; always run eagerly.
        self.graph = None

    def simulate(self):
        """Run one frame's worth of substeps."""
        px, py = self.probe_xy
        probe_pos = wp.vec3(px, py, self.probe_z)

        # Penetration-proportional contact force
        probe_bottom = self.probe_z - self.probe_radius
        penetration = max(0.0, self.initial_top_z - probe_bottom)
        force_mag = 1_000.0 * penetration
        contact_radius = 0.075

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            wp.launch(
                set_probe_kinematic,
                dim=1,
                inputs=[self.state_0.body_q, self.state_0.body_qd, self.probe_id, probe_pos],
            )

            if force_mag > 0.0:
                wp.launch(
                    apply_probe_contact,
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
        """Advance one display frame."""
        self._check_keys()
        self._animate_probe()

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

        if self.poke_active:
            self._frame_count += 1
            if self._frame_count % 10 == 0:
                mm = self._measure_deformation()
                depth = self.depth_levels[self.depth_index]
                print(f"\r  target={depth:4.0f} mm  indent={mm:6.1f} mm  ", end="", flush=True)

    def _animate_probe(self):
        """Smoothly move probe to its target Z."""
        if self.poke_active:
            target_z = self.probe_surface_z - self.depth_levels[self.depth_index] / 1000.0
        else:
            target_z = self.probe_rest_z
        max_move = 0.3 * self.frame_dt
        delta = target_z - self.probe_z
        self.probe_z += max_move * np.sign(delta) if abs(delta) > max_move else delta

    def _check_keys(self):
        if not hasattr(self.viewer, "is_key_down"):
            return
        keys = {k: self.viewer.is_key_down(k) for k in self._prev_keys}
        if keys["p"] and not self._prev_keys["p"]:
            self.poke_active = not self.poke_active
            status = "DOWN" if self.poke_active else "UP"
            depth = self.depth_levels[self.depth_index]
            print(f"\nProbe {status}  depth={depth:.0f} mm")
            if self.poke_active:
                self._frame_count = 0
        if keys["up"] and not self._prev_keys["up"]:
            if self.depth_index < len(self.depth_levels) - 1:
                self.depth_index += 1
                print(f"\n  depth → {self.depth_levels[self.depth_index]:.0f} mm")
        if keys["down"] and not self._prev_keys["down"]:
            if self.depth_index > 0:
                self.depth_index -= 1
                print(f"\n  depth → {self.depth_levels[self.depth_index]:.0f} mm")
        self._prev_keys = keys

    def _measure_deformation(self) -> float:
        """Return indentation depth in mm near the probe centre."""
        q = self.state_0.particle_q.numpy()
        top_q = q[self.top_particle_indices]
        cx, cy = self.probe_xy
        r = self.probe_radius * 3.0
        near = (top_q[:, 0] - cx) ** 2 + (top_q[:, 1] - cy) ** 2 < r ** 2
        if near.any():
            return (self.initial_top_z - float(top_q[near, 2].min())) * 1000.0
        return 0.0

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--tissue",
        type=str,
        default="generic_soft_tissue",
        choices=list(TISSUE_PROPERTIES.keys()),
        help="Tissue type to simulate",
    )
    parser.add_argument(
        "--surface_preview",
        action="store_true",
        help="Print surface deformation stats at startup",
    )

    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
