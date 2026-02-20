# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct-USD reproduction of Bug 1: Scale / collider mismatch ("floating objects").

A rigid body referenced from a USD file with ``xformOp:scale != (1,1,1)``
on the **same prim** that carries ``UsdPhysics.RigidBodyAPI`` results in
a collider whose size does not match the visual mesh.  The object appears
to "float" above the surface it should be resting on.

The script places two blocks side-by-side on a table:
  - **BlockNormal** (blue, scale=1.0) — control
  - **BlockScaled** (red, scale=0.7) — should rest lower but floats

No IsaacLab scene/asset/spawner APIs are used — only raw USD, PhysX,
Fabric, and omni.physics.tensors.

Usage:

    # GPU (default) — should exhibit the bug
    ./isaaclab.sh -p scripts/tools/repro_scale_bug1_collider_mismatch.py --num_envs 1

    # CPU — should behave correctly
    ./isaaclab.sh -p scripts/tools/repro_scale_bug1_collider_mismatch.py --device cpu --num_envs 1
"""

# ── Launch Isaac Sim first ──────────────────────────────────────────────────
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Direct-USD repro for Bug 1: collider mismatch.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs (manual clones).")
parser.add_argument("--fabric", action="store_true", default=False, help="Enable Fabric.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Everything below runs after the simulator is up ─────────────────────────
import contextlib
import math

import carb
import omni.kit.app
import omni.physx
import omni.physics.tensors
import omni.timeline
import omni.usd

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils, Vt

# Asset paths — resolve from carb setting so the version always matches
_ASSET_ROOT = carb.settings.get_settings().get("/persistent/isaac/asset_root/cloud")
ISAAC_NUCLEUS_DIR = f"{_ASSET_ROOT}/Isaac"

BLUE_BLOCK_USD = f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd"
RED_BLOCK_USD = f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd"
TABLE_USD = f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"

DT = 0.01  # physics timestep


# ============================================================================
# Pure USD helpers
# ============================================================================

def _check_usd_path(usd_path: str) -> str:
    """Verify that a USD path can be opened, raise if not."""
    layer = Sdf.Layer.FindOrOpen(usd_path)
    if layer:
        return usd_path
    raise FileNotFoundError(f"USD file not found: {usd_path}")


def _create_stage() -> Usd.Stage:
    """Create a new stage via the omni.usd context so Kit can render it."""
    usd_context = omni.usd.get_context()
    usd_context.new_stage()
    omni.kit.app.get_app().update()

    stage = usd_context.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    return stage


def _add_physics_scene(stage: Usd.Stage, path: str, dt: float, device: str) -> None:
    """Add a UsdPhysics.Scene + PhysxSceneAPI to the stage."""
    scene = UsdPhysics.Scene.Define(stage, path)
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)

    prim = stage.GetPrimAtPath(path)
    PhysxSchema.PhysxSceneAPI.Apply(prim)
    api = PhysxSchema.PhysxSceneAPI(prim)
    api.CreateTimeStepsPerSecondAttr(int(1.0 / dt))

    is_gpu = "cuda" in device
    api.CreateBroadphaseTypeAttr("GPU" if is_gpu else "MBP")
    api.CreateEnableGPUDynamicsAttr(is_gpu)
    api.CreateEnableSceneQuerySupportAttr(False)
    api.CreateEnableCCDAttr(False)
    api.CreateSolverTypeAttr("TGS")


def _add_ground_plane(stage: Usd.Stage, path: str = "/World/GroundPlane") -> None:
    """Add a large static collider plane."""
    UsdGeom.Xform.Define(stage, path)
    mesh_path = f"{path}/CollisionMesh"
    plane = UsdGeom.Mesh.Define(stage, mesh_path)

    half = 5.0
    plane.CreatePointsAttr(
        [Gf.Vec3f(-half, -half, 0), Gf.Vec3f(half, -half, 0), Gf.Vec3f(half, half, 0), Gf.Vec3f(-half, half, 0)]
    )
    plane.CreateFaceVertexCountsAttr([4])
    plane.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    plane.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)

    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(mesh_path))

    xform_prim = stage.GetPrimAtPath(path)
    UsdGeom.Xformable(xform_prim).AddTranslateOp().Set(Gf.Vec3d(0, 0, -1.05))


def _spawn_usd_rigid_body(
    stage: Usd.Stage,
    prim_path: str,
    usd_path: str,
    translation: tuple[float, float, float],
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    kinematic: bool = False,
    mass: float | None = None,
) -> None:
    """Spawn a USD reference as a rigid body.

    Places ``xformOp:scale`` and ``UsdPhysics.RigidBodyAPI`` on the
    **same prim** (the root Xform) — the pattern that triggers the bug.
    """
    usd_path = _check_usd_path(usd_path)

    prim = stage.DefinePrim(prim_path, "Xform")
    prim.GetReferences().AddReference(usd_path)

    root_layer = stage.GetRootLayer()
    prim_spec = root_layer.GetPrimAtPath(prim_path)
    with Sdf.ChangeBlock():
        translate_spec = Sdf.AttributeSpec(prim_spec, "xformOp:translate", Sdf.ValueTypeNames.Double3)
        translate_spec.default = Gf.Vec3d(*translation)

        orient_spec = Sdf.AttributeSpec(prim_spec, "xformOp:orient", Sdf.ValueTypeNames.Quatd)
        orient_spec.default = Gf.Quatd(1.0, 0.0, 0.0, 0.0)

        scale_spec = Sdf.AttributeSpec(prim_spec, "xformOp:scale", Sdf.ValueTypeNames.Double3)
        scale_spec.default = Gf.Vec3d(*scale)

        order_spec = prim_spec.GetAttributeAtPath(f"{prim_path}.xformOpOrder")
        if order_spec is None:
            order_spec = Sdf.AttributeSpec(prim_spec, "xformOpOrder", Sdf.ValueTypeNames.TokenArray)
        order_spec.default = Vt.TokenArray(["xformOp:translate", "xformOp:orient", "xformOp:scale"])

    UsdPhysics.RigidBodyAPI.Apply(prim)
    if kinematic:
        prim.GetAttribute("physics:kinematicEnabled").Set(True)

    if mass is not None:
        UsdPhysics.MassAPI.Apply(prim)
        prim.GetAttribute("physics:mass").Set(mass)


def _add_distant_light(stage: Usd.Stage, path: str = "/World/Light") -> None:
    """Add a simple distant light."""
    UsdGeom.Xform.Define(stage, path)
    stage.DefinePrim(f"{path}/DistantLight", "DistantLight")
    prim = stage.GetPrimAtPath(f"{path}/DistantLight")
    prim.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(3000.0)
    prim.CreateAttribute("inputs:color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.75, 0.75, 0.75))


# ============================================================================
# Main
# ============================================================================

def main():
    device = args_cli.device if args_cli.device else "cuda:0"
    num_envs = args_cli.num_envs
    is_gpu = "cuda" in device

    settings = carb.settings.get_settings()

    # ── 1. Carb/PhysX settings ───────────────────────────────────────────────
    settings.set_bool("/persistent/omnihydra/useSceneGraphInstancing", True)
    settings.set_bool("/physics/physxDispatcher", True)
    settings.set_bool("/physics/disableContactProcessing", True)
    settings.set_bool("/physics/collisionConeCustomGeometry", False)
    settings.set_bool("/physics/collisionCylinderCustomGeometry", False)
    settings.set_bool("/physics/autoPopupSimulationOutputWindow", False)

    if is_gpu:
        parts = device.split(":")
        device_id = int(parts[1]) if len(parts) > 1 else 0
        settings.set_int("/physics/cudaDevice", device_id)
        settings.set_bool("/physics/suppressReadback", True)
    else:
        settings.set_bool("/physics/suppressReadback", False)

    settings.set_bool("/physics/visualizationDisplaySimulationOutput", False)

    use_fabric = args_cli.fabric

    # ── 2. Create USD stage FIRST (Fabric needs a valid stage) ───────────────
    stage = _create_stage()

    # ── 2b. Enable Fabric AFTER stage exists ─────────────────────────────────
    ext_mgr = omni.kit.app.get_app().get_extension_manager()
    if use_fabric:
        if not ext_mgr.is_extension_enabled("omni.physx.fabric"):
            ext_mgr.set_extension_enabled_immediate("omni.physx.fabric", True)
        settings.set_bool("/physics/fabricEnabled", True)
    else:
        if ext_mgr.is_extension_enabled("omni.physx.fabric"):
            ext_mgr.set_extension_enabled_immediate("omni.physx.fabric", False)
        settings.set_bool("/physics/fabricEnabled", False)

    for key in [
        "updateToUsd", "updateParticlesToUsd", "updateVelocitiesToUsd",
        "updateForceSensorsToUsd", "updateResidualsToUsd",
    ]:
        settings.set_bool(f"/physics/{key}", not use_fabric)

    omni.kit.app.get_app().update()

    # Stage cache
    stage_cache = UsdUtils.StageCache.Get()
    if stage_cache.GetId(stage).ToLongInt() < 0:
        stage_cache.Insert(stage)
    stage_id = stage_cache.GetId(stage).ToLongInt()

    # ── 3. Physics scene ─────────────────────────────────────────────────────
    _add_physics_scene(stage, "/physicsScene", DT, device)

    # ── 4. Ground plane + light ──────────────────────────────────────────────
    _add_ground_plane(stage)
    _add_distant_light(stage)

    # ── 5. Spawn objects ─────────────────────────────────────────────────────
    ENV_SPACING = 2.5
    normal_paths: list[str] = []
    scaled_paths: list[str] = []

    for env_idx in range(num_envs):
        env_offset = (env_idx * ENV_SPACING, 0.0, 0.0)
        base_path = f"/World/envs/env_{env_idx}"
        stage.DefinePrim(base_path, "Xform")

        # Table (kinematic)
        _spawn_usd_rigid_body(
            stage, f"{base_path}/Table", TABLE_USD,
            translation=(0.5 + env_offset[0], 0.0, 0.0),
            kinematic=True,
        )

        # Table surface is ~0.76 m above the table origin.
        # Place blocks just above the table surface.
        TABLE_SURFACE_Z = 0.5
        DROP_Z = TABLE_SURFACE_Z + 0.005  # 5 mm above surface

        # Place blocks at the table centre (x=0.5) with wide y-spacing
        table_x = 0.5 + env_offset[0]

        # Blue block at scale=1.0 (CONTROL)
        blue_path = f"{base_path}/BlockNormal"
        _spawn_usd_rigid_body(
            stage, blue_path, BLUE_BLOCK_USD,
            translation=(table_x, -0.15, DROP_Z),
            scale=(1.0, 1.0, 1.0),
            mass=0.1,
            kinematic=True,
        )
        normal_paths.append(blue_path)

        # Red block at scale=0.7 (BUG — collider mismatch)
        red_path = f"{base_path}/BlockScaled"
        _spawn_usd_rigid_body(
            stage, red_path, RED_BLOCK_USD,
            translation=(table_x, 0.15, DROP_Z),
            scale=(0.7, 0.7, 0.7),
            mass=0.1,
            kinematic=True,
        )
        scaled_paths.append(red_path)

    # ── 6. Attach stage to PhysX ─────────────────────────────────────────────
    physx_sim = omni.physx.get_physx_simulation_interface()
    physx_iface = omni.physx.get_physx_interface()
    physx_sim.attach_stage(stage_id)

    # ── 7. Warm up physics + create tensor views ─────────────────────────────
    physx_iface.force_load_physics_from_usd()
    physx_iface.start_simulation()
    physx_iface.update_simulation(DT, 0.0)
    physx_sim.fetch_results()

    sim_view = omni.physics.tensors.create_simulation_view("warp", stage_id=stage_id)
    sim_view.set_subspace_roots("/")
    rb_view = sim_view.create_rigid_body_view("/World/envs/env_*/Block*")

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    settings.set_bool("/app/player/playSimulations", False)
    app = omni.kit.app.get_app()
    for _ in range(2):
        app.update()
    settings.set_bool("/app/player/playSimulations", True)

    # ── 8. Print header ──────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("BUG 1 REPRO: Scale / Collider Mismatch — Floating Objects")
    print(f"  device   = {device}")
    print(f"  num_envs = {num_envs}")
    print(f"  fabric   = {'ON' if use_fabric else 'OFF'}")
    print(f"  rb_view  = {rb_view.count} bodies")
    print()
    print("  BlockNormal (blue, scale=1.0) vs BlockScaled (red, scale=0.7)")
    print("  If the bug is present, the scaled block floats at the SAME or")
    print("  HIGHER Z as the normal block despite being smaller.")
    print(f"{'=' * 80}\n")

    # ── 9. Simulation loop ───────────────────────────────────────────────────
    import warp as wp

    fabric_update = None
    if use_fabric:
        from omni.physxfabric import get_physx_fabric_interface
        _fabric_iface = get_physx_fabric_interface()
        fabric_update = getattr(_fabric_iface, "force_update", _fabric_iface.update)

    def _get_block_heights() -> tuple[list[float], list[float]]:
        """Return (normal_z_list, scaled_z_list) from the tensor API."""
        z_normal, z_scaled = [], []
        if rb_view.count > 0:
            tf = wp.to_torch(rb_view.get_transforms())
            for i in range(rb_view.count):
                name = rb_view.prim_paths[i].rsplit("/", 1)[-1]
                z = tf[i, 2].item()
                if name == "BlockNormal":
                    z_normal.append(z)
                elif name == "BlockScaled":
                    z_scaled.append(z)
        return z_normal, z_scaled

    def _print_status(step_idx: int):
        z_normal, z_scaled = _get_block_heights()
        n_str = ", ".join(f"{z:.4f}" for z in z_normal) if z_normal else "N/A"
        s_str = ", ".join(f"{z:.4f}" for z in z_scaled) if z_scaled else "N/A"
        print(f"  step {step_idx:>5d}  |  BlockNormal z=[{n_str}]  |  BlockScaled z=[{s_str}]")

    def _check_floating(step_idx: int):
        z_normal, z_scaled = _get_block_heights()
        if not z_normal or not z_scaled:
            print("  (cannot check — missing block data)")
            return

        avg_normal = sum(z_normal) / len(z_normal)
        avg_scaled = sum(z_scaled) / len(z_scaled)
        diff = avg_scaled - avg_normal

    step = 0
    _print_status(step)

    with contextlib.suppress(KeyboardInterrupt):
        while simulation_app.is_running() and not simulation_app.is_exiting():
            step += 1

            physx_sim.simulate(DT, 0.0)
            physx_sim.fetch_results()

            if fabric_update is not None:
                fabric_update(DT, 0.0)

            app.update()

            if step <= 10 or step % 50 == 0:
                _print_status(step)

            if step == 100:
                _check_floating(step)

    # ── Final summary ────────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("FINAL STATUS")
    print(f"{'=' * 80}")
    _print_status(step)
    _check_floating(step)
    print()


if __name__ == "__main__":
    main()
    simulation_app.close()
