# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct-USD reproduction of Bug 2: Infinite growth on GPU.

Certain USD assets (e.g. a steering wheel) with ``xformOp:scale != (1,1,1)``
on the **same prim** that carries ``UsdPhysics.RigidBodyAPI`` grow toward
infinite size once the GPU simulation starts.  The effect is visible both
in the tensor-API positions (Z diverges) and visually in the viewport.

The script places a steering wheel on a table:
  - **SteeringWheel** (scale=0.75) — the asset that exhibits infinite growth

No IsaacLab scene/asset/spawner APIs are used — only raw USD, PhysX,
Fabric, and omni.physics.tensors.

Usage:

    # GPU (default) — should exhibit the bug
    ./isaaclab.sh -p scripts/tools/repro_scale_bug2_infinite_growth.py --num_envs 1

    # CPU — should behave correctly
    ./isaaclab.sh -p scripts/tools/repro_scale_bug2_infinite_growth.py --device cpu --num_envs 1
"""

# ── Launch Isaac Sim first ──────────────────────────────────────────────────
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Direct-USD repro for Bug 2: infinite growth.")
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
ISAACLAB_NUCLEUS_DIR = f"{ISAAC_NUCLEUS_DIR}/IsaacLab"

STEERING_WHEEL_USD = f"{ISAACLAB_NUCLEUS_DIR}/Mimic/pick_place_task/pick_place_assets/steering_wheel.usd"
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

    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI(prim).CreateApproximationAttr("convexHull")
    # Also tag reachable Mesh descendants (non-instanceable)
    for desc in Usd.PrimRange(prim):
        if desc.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(desc)
            UsdPhysics.MeshCollisionAPI.Apply(desc)
            UsdPhysics.MeshCollisionAPI(desc).CreateApproximationAttr("convexHull")

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


def _read_prim_scale(stage: Usd.Stage, path: str) -> tuple[float, float, float]:
    """Read the xformOp:scale attribute of a prim."""
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return (float("nan"),) * 3
    attr = prim.GetAttribute("xformOp:scale")
    if attr and attr.Get() is not None:
        v = attr.Get()
        return (v[0], v[1], v[2])
    return (1.0, 1.0, 1.0)


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
    sw_paths: list[str] = []

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

        # Steering wheel at scale=0.75 (BUG — infinite growth)
        sw_path = f"{base_path}/SteeringWheel"
        _spawn_usd_rigid_body(
            stage, sw_path, STEERING_WHEEL_USD,
            translation=(0.45 + env_offset[0], 0.0, 0.3),
            scale=(0.75, 0.75, 0.75),
            mass=0.1,
        )
        sw_paths.append(sw_path)

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
    sw_view = sim_view.create_rigid_body_view("/World/envs/env_*/SteeringWheel")

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    settings.set_bool("/app/player/playSimulations", False)
    app = omni.kit.app.get_app()
    for _ in range(2):
        app.update()
    settings.set_bool("/app/player/playSimulations", True)

    # ── 8. Print header ──────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("BUG 2 REPRO: Infinite Growth on GPU")
    print(f"  device   = {device}")
    print(f"  num_envs = {num_envs}")
    print(f"  fabric   = {'ON' if use_fabric else 'OFF'}")
    print(f"  sw_view  = {sw_view.count} bodies")
    print()
    print("  SteeringWheel (scale=0.75) — exhibits infinite growth on GPU")
    print(f"{'=' * 80}\n")

    # ── 9. Simulation loop ───────────────────────────────────────────────────
    import warp as wp

    fabric_update = None
    if use_fabric:
        from omni.physxfabric import get_physx_fabric_interface
        _fabric_iface = get_physx_fabric_interface()
        fabric_update = getattr(_fabric_iface, "force_update", _fabric_iface.update)

    def _get_sw_positions() -> list[float]:
        """Read Z heights of steering wheels from the tensor API."""
        zs = []
        if sw_view.count > 0:
            tf = wp.to_torch(sw_view.get_transforms())
            for i in range(sw_view.count):
                zs.append(tf[i, 2].item())
        return zs

    def _print_status(step_idx: int):
        zs = _get_sw_positions()
        z_str = ", ".join(f"{z:.4f}" for z in zs) if zs else "N/A"
        print(f"  step {step_idx:>5d}  |  SteeringWheel z=[{z_str}]")

    def _check_infinite_growth(step_idx: int):
        # Check via USD scale
        for p in sw_paths:
            sc = _read_prim_scale(stage, p)
            if any(math.isinf(v) or math.isnan(v) or abs(v) > 100 for v in sc):
                print(f"    *** BUG 2 — INFINITE SCALE: {p}  scale={sc} ***")

        # Check via tensor positions
        for z in _get_sw_positions():
            if math.isinf(z) or math.isnan(z) or abs(z) > 100:
                print(f"    *** BUG 2 — INFINITE POSITION: SteeringWheel  z={z:.4f} ***")

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
                _check_infinite_growth(step)

    # ── Final summary ────────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("FINAL STATUS")
    print(f"{'=' * 80}")
    _print_status(step)
    _check_infinite_growth(step)
    print()


if __name__ == "__main__":
    main()
    simulation_app.close()
