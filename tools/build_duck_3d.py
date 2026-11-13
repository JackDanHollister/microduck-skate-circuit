"""Export the roller robot from the pinned Arena wheel and check its replay kinematics.

Run from the repository root in the simulation environment:
    python tools/build_duck_3d.py
Only visual meshes are exported. Geometry-to-body transforms incorporate MuJoCo's
compiled mesh centering and alignment, which differ from the source STL frames.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
from arena import course
from arena.constants import JOINT_NAMES
from arena.manifest import load_event

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
PREFIX = "duck-3d"


def load_model():
    # Stage physics usually strips display meshes. Keep them for this exporter.
    with patch.object(course, "_strip_visuals", lambda root: None):
        xml, assets = course.build_course_xml(replace(load_event("got-a-talent"), rig="rollers"))
    return course._compiled(course._motorized(xml, assets))


def simplified_mesh(vertices, triangles, detail, name):
    """Quadric decimation for distant ducks; each rigid part keeps its own material."""
    import fast_simplification

    if detail=="mid":
        # Preserve the silhouettes of the head shell, jaw, roller boots and wheels.
        exterior=("shell","jaw","foot","roller_blade","tire","ankle","trunk","upper_leg","neck","leg","rim")
        ratio=.32 if any(part in name for part in exterior) else .12
        target=min(len(triangles),max(100,round(len(triangles)*ratio)))
    else:
        target=min(len(triangles),max(28,round(len(triangles)*.025)))
    if target < len(triangles):
        original_vertices = vertices.astype(np.float64)
        original_faces = triangles.astype(np.int32)
        # Many small mechanical details are already disconnected or open.
        # Weld neighbouring vertices before decimating such assemblies.
        dimensions = np.ptp(original_vertices, axis=0)
        for bins in ((240, 140, 85, 50) if detail=="mid" else (30, 18, 12, 8)):
            spacing = max(dimensions) / bins
            keys = np.rint(original_vertices / spacing).astype(np.int32)
            _, ids = np.unique(keys, axis=0, return_inverse=True)
            count = np.bincount(ids)
            vertices = np.column_stack([np.bincount(ids, weights=original_vertices[:, i]) / count for i in range(3)])
            candidates = ids[original_faces]
            distinct = ((candidates[:, 0] != candidates[:, 1]) &
                        (candidates[:, 0] != candidates[:, 2]) &
                        (candidates[:, 1] != candidates[:, 2]))
            candidates = candidates[distinct]
            if len(candidates) < 4:
                continue
            _, first = np.unique(np.sort(candidates,axis=1), axis=0, return_index=True)
            triangles = candidates[np.sort(first)]
            if len(triangles)>target:
                vertices, triangles = fast_simplification.simplify(
                    vertices, triangles.astype(np.int32), target_count=target, agg=7)
            if 0 < len(triangles) <= target * 1.5:
                break
    if len(triangles) == 0:
        raise ValueError("A simplified visual part has no surfaces")
    face = vertices[triangles]
    normals = np.zeros_like(vertices)
    face_normals = np.cross(face[:, 1] - face[:, 0], face[:, 2] - face[:, 0])
    for corner in range(3):
        np.add.at(normals, triangles[:, corner], face_normals)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-18)
    return vertices, normals, triangles.ravel()


def export(model, detail="full"):
    data = bytearray()

    def append(array, dtype):
        offset = len(data)
        data.extend(np.ascontiguousarray(array, dtype=dtype).tobytes())
        return {"offset": offset, "count": len(array)}

    meshes = []
    mesh_ids = {}
    bodies = []
    body_ids = {}
    geom_ids = []
    trunk = model.body("trunk_base").id
    for body in range(1, model.nbody):
        parent = int(model.body_parentid[body])
        if body != trunk and parent not in body_ids:
            continue
        body_ids[body] = len(bodies)
        joint = None
        if body != trunk and model.body_jntnum[body]:
            assert model.body_jntnum[body] == 1
            j = model.body_jntadr[body]
            assert model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
            joint = {"name": model.joint(j).name, "axis": model.jnt_axis[j].tolist(),
                     "pivot": model.jnt_pos[j].tolist(), "reference": float(model.qpos0[model.jnt_qposadr[j]])}
        geoms = []
        for geom in np.flatnonzero((model.geom_bodyid == body) & (model.geom_group == 2)):
            assert model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH
            mesh = model.geom_dataid[geom]
            if mesh not in mesh_ids:
                mesh_ids[mesh] = len(meshes)
                start = model.mesh_faceadr[mesh]
                faces = slice(start, start + model.mesh_facenum[mesh])
                corners = np.stack((model.mesh_face[faces], model.mesh_facenormal[faces]), axis=-1).reshape(-1, 2)
                # Preserve the original normals on sharp edges where one vertex has two normals.
                pair, remapped = np.unique(corners, axis=0, return_inverse=True)
                positions = model.mesh_vert[model.mesh_vertadr[mesh] + pair[:, 0]]
                normals = model.mesh_normal[model.mesh_normaladr[mesh] + pair[:, 1]]
                if detail!="full":
                    start_vertex = model.mesh_vertadr[mesh]
                    vertices = model.mesh_vert[start_vertex:start_vertex + model.mesh_vertnum[mesh]]
                    positions, normals, remapped = simplified_mesh(vertices, model.mesh_face[faces], detail, model.mesh(mesh).name)
                meshes.append({"name": model.mesh(mesh).name, "positions": append(positions, "<f4"),
                               "normals": append(normals, "<f4"), "indices": append(remapped.ravel(), "<u4")})
            mat = model.geom_matid[geom]
            colour = model.mat_rgba[mat] if mat >= 0 else model.geom_rgba[geom]
            geoms.append({"mesh": mesh_ids[mesh], "pos": model.geom_pos[geom].tolist(),
                          "quat": model.geom_quat[geom].tolist(), "rgba": colour.tolist()})
            geom_ids.append(int(geom))
        bodies.append({"name": model.body(body).name, "parent": body_ids.get(parent, -1),
                       "pos": model.body_pos[body].tolist(), "quat": model.body_quat[body].tolist(),
                       "joint": joint, "geoms": geoms})
    rig = {"format": 1, "attribution": "Pollen Robotics · see duck-attribution.md",
           "bodies": bodies, "meshes": meshes}
    return rig, data, geom_ids


def rotation(quat):
    """Unit quaternion, scalar first, to rotation matrix."""
    w, x, y, z = quat
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def verify_frames(rig, model, geom_ids):
    """Recompute world transforms as the browser does; use MuJoCo as the reference."""
    replay = json.loads((ROOT / "replays/race.json").read_text())
    names = replay["joints"]
    assert set(names) == set(JOINT_NAMES)
    adr = [int(model.joint(n).qposadr[0]) for n in names]
    root = model.jnt_qposadr[model.body_jntadr[model.body("trunk_base").id]]
    sim = mujoco.MjData(model)
    count = 0
    worst = 0.0
    for actor in ("baseline", "challenger"):
        frames = np.asarray(replay[actor]["frames"])
        for frame in frames[np.linspace(0, len(frames)-1, 30, dtype=int)]:
            quat = frame[4:8] / np.linalg.norm(frame[4:8])
            sim.qpos[root:root+7] = np.r_[frame[1:4], quat]
            sim.qpos[adr] = frame[8:]
            mujoco.mj_forward(model, sim)
            pose = dict(zip(names, frame[8:]))
            world = []
            it = iter(geom_ids)
            for b in rig["bodies"]:
                if b["parent"] < 0:
                    t, r = frame[1:4], rotation(quat)
                else:
                    pt, pr = world[b["parent"]]
                    t, r = pt + pr @ b["pos"], pr @ rotation(b["quat"])
                    joint = b["joint"]
                    if joint:
                        angle = pose.get(joint["name"], joint["reference"]) - joint["reference"]
                        angle *= .5
                        rr = rotation(np.r_[np.cos(angle), np.asarray(joint["axis"]) * np.sin(angle)])
                        pivot = np.array(joint["pivot"])
                        t = t + r @ (pivot - rr @ pivot)
                        r = r @ rr
                world.append((t, r))
                for geom in b["geoms"]:
                    g = next(it)
                    actual_translation = t + r @ geom["pos"]
                    actual_rotation = r @ rotation(geom["quat"])
                    diff = max(np.max(np.abs(actual_translation - sim.geom_xpos[g])),
                               np.max(np.abs(actual_rotation - sim.geom_xmat[g].reshape(3, 3))))
                    worst = max(worst, float(diff))
                    count += 1
    assert worst < 1e-8, f"Exported kinematics diverged by {worst}"
    return count, worst


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low-detail", action="store_true",
                        help="export distant ghost geometry; requires fast-simplification==0.1.12")
    parser.add_argument("--mid-detail", action="store_true",
                        help="export a medium-detail robot for full-sized nearby ghosts")
    args = parser.parse_args()
    if args.low_detail and args.mid_detail:
        parser.error("Choose one detail level per invocation")
    detail = "low" if args.low_detail else "mid" if args.mid_detail else "full"
    model = load_model()
    rig, blob, geom_ids = export(model, detail)
    prefix = PREFIX + ("" if detail=="full" else "-"+detail)
    ASSETS.mkdir(exist_ok=True)
    (ASSETS / f"{prefix}.bin").write_bytes(blob)
    path = ASSETS / f"{prefix}.json"
    path.write_text(json.dumps(rig, separators=(",", ":"), allow_nan=False) + "\n")
    verified, error = verify_frames(json.loads(path.read_text()), model, geom_ids)
    print(f"{len(rig['bodies'])} bodies, {len(geom_ids)} visual geoms, {len(rig['meshes'])} indexed meshes, "
          f"{len(blob):,} geometry bytes; verified {verified} geometry poses, maximum error {error:.2g}")


if __name__ == "__main__":
    main()
