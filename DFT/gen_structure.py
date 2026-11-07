# SPDX-License-Identifier: GPL-3.0-only
"""Build a Blender scene from native PySCF or supported GAMESS geometry.

blender --background --factory-startup --python-exit-code 1 --python DFT/gen_structure.py -- --render
Add --shader DFT/results/example_C22H14_alpha_homo.osl to embed a calculated MO volume.
Creates a new scene; existing scenes and their objects are retained.
"""

import argparse
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np

import bpy
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from DFT.workflow import DEFAULT_INPUT, infer_bonds, read_input

STYLES = {1: (0.14, (0.88, 0.91, 0.96, 1)), 6: (0.22, (0.10, 0.13, 0.18, 1)),
          7: (0.21, (0.08, 0.18, 0.75, 1)), 8: (0.20, (0.65, 0.06, 0.025, 1))}

ATOM_SCALE = 2.
BOND_RADIUS = .1

def create_scene(job, *, bonds=True, bond_factor=1.2, shader=None, mode="amplitude",
                 density=1.0, threshold=0.0, padding=4.0, resolution=960, samples=16):
    pairs = infer_bonds(job, bond_factor) if bonds else []
    if mode not in {"amplitude", "probability"}:
        raise ValueError("Mode must be amplitude or probability")
    if resolution < 32 or samples < 1 or padding < 0 or density < 0 or threshold < 0:
        raise ValueError("Resolution >= 32, samples >= 1, padding/density/threshold >= 0 are required")
    if not all(math.isfinite(value) for value in (padding, density, threshold)):
        raise ValueError("Volume controls must be finite")
    shader_source = None
    if shader:
        shader = Path(shader)
        shader_source = shader.read_text(encoding="utf8")
        if not bpy.app.build_options.cycles_osl:
            raise RuntimeError("The selected Blender build does not support Cycles OSL")
        manifest = shader.with_suffix(".selection.json")
        if manifest.is_file():
            selection = json.loads(manifest.read_text(encoding="utf8"))
            if (selection.get("atomic_numbers") != job.atomic_numbers
                    or not np.allclose(selection.get("coordinates_angstrom"), job.coordinates, atol=1e-5, rtol=0)):
                raise ValueError("Shader and structure have different atom coordinates/order")
    scene = bpy.data.scenes.new("DFT molecular structure")
    bpy.context.window.scene = scene
    # Cycles OSL must be enabled before updating an internal Script node.
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.shading_system = bool(shader)
    scene["source_title"] = job.title
    scene["coordinate_units"] = "1 Blender unit = 1 angstrom"
    scene["charge"] = job.charge
    scene["multiplicity"] = job.multiplicity
    scene["atom_count"] = len(job.symbols)
    scene["bond_count"] = len(pairs)
    scene["bond_definition"] = "Distance-based guides, without bond orders"
    collection = bpy.data.collections.new("Molecule")
    scene.collection.children.link(collection)

    def relocate(obj):
        for old in list(obj.users_collection):
            old.objects.unlink(obj)
        collection.objects.link(obj)
        return obj

    materials = {}
    for z in set(job.atomic_numbers):
        radius, color = STYLES.get(z, (0.22, (0.35, 0.45, 0.55, 1)))
        material = bpy.data.materials.new(f"Element {z}")
        material.diffuse_color = color
        material.use_nodes = True
        bsdf = material.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = 0.28
        materials[z] = material
    for index, (symbol, z, position) in enumerate(zip(job.symbols, job.atomic_numbers, job.coordinates)):
        radius = ATOM_SCALE*STYLES.get(z, (0.22, None))[0]
        bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=radius, location=position)
        atom = relocate(bpy.context.object)
        atom.name = f"Atom {index+1}: {symbol}"
        atom["atom_index_one_based"] = index + 1
        atom["atomic_number"] = z
        atom.data.materials.append(materials[z])
        for polygon in atom.data.polygons:
            polygon.use_smooth = True
    for first, second in pairs:
        start, end = Vector(job.coordinates[first]), Vector(job.coordinates[second])
        middle = (start + end) / 2
        # Two colored halves per inferred bond, including aromatic C-C guides.
        for index, lower, upper in ((first, start, middle), (second, middle, end)):
            direction = upper - lower
            bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=BOND_RADIUS,
                                                depth=direction.length, end_fill_type="NOTHING",
                                                location=(lower + upper) / 2)
            bond = relocate(bpy.context.object)
            bond.name = f"Bond {first+1}-{second+1} ({index+1})"
            bond["bond_atom_indices_one_based"] = [first + 1, second + 1]
            bond.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()
            bond.data.materials.append(materials[job.atomic_numbers[index]])
            for polygon in bond.data.polygons:
                polygon.use_smooth = True

    positions = np.asarray(job.coordinates)
    lower = positions.min(axis=0) - (padding if shader else 0.4)
    upper = positions.max(axis=0) + (padding if shader else 0.4)
    if shader:
        vertices = list(itertools.product(*zip(lower, upper)))
        faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
                 (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
        mesh = bpy.data.meshes.new("Molecular orbital boundary")
        mesh.from_pydata(vertices, [], faces)
        mesh.update()
        volume = bpy.data.objects.new("Molecular orbital volume", mesh)
        collection.objects.link(volume)
        # Vertices are physical coordinates; keep the object transform identity.
        volume.display_type = "WIRE"
        volume["coordinate_units"] = "angstrom; shader converts to bohr"
        material = bpy.data.materials.new("Calculated DFT orbital")
        material.use_nodes = True
        nodes = material.node_tree.nodes
        nodes.clear()
        script = nodes.new("ShaderNodeScript")
        text = bpy.data.texts.new(shader.name)
        text.write(shader_source)
        script.mode = "INTERNAL"
        script.script = text
        script.update()
        output_name = "PA" if mode == "amplitude" else "PD"
        if output_name not in script.outputs or "BohrInBlender" not in script.inputs:
            raise RuntimeError("Expected a compiled molecular OSL shader produced by gen_shader.py")
        script.inputs["Density"].default_value = density
        script.inputs["Threshold"].default_value = threshold
        out = nodes.new("ShaderNodeOutputMaterial")
        material.node_tree.links.new(script.outputs[output_name], out.inputs["Volume"])
        material.cycles.volume_step_rate = 0.25
        volume.data.materials.append(material)

    target = Vector((lower + upper) / 2)
    # View the molecular plane face-on, with its longest principal axis horizontal.
    _, _, axes = np.linalg.svd(positions - positions.mean(axis=0), full_matrices=True)
    right = Vector(axes[0])
    normal = Vector(axes[2])
    if right[int(np.argmax(np.abs(axes[0])))] < 0:
        right = -right
    if normal[int(np.argmax(np.abs(axes[2])))] < 0:
        normal = -normal
    up = normal.cross(right).normalized()
    distance = max(float(np.linalg.norm(upper - lower)), 4.0)
    bpy.ops.object.camera_add(location=target + normal * distance)
    camera = bpy.context.object
    camera.rotation_euler = Matrix((right, up, normal)).transposed().to_euler()
    camera.data.type = "ORTHO"
    camera.data.sensor_fit = "HORIZONTAL"
    corners = [Vector(corner) - target for corner in itertools.product(*zip(lower, upper))]
    width = 2 * max(abs(v.dot(right)) for v in corners)
    height = 2 * max(abs(v.dot(up)) for v in corners)
    camera.data.ortho_scale = max(width, 2.0 * height, 1.0) * 1.12
    camera.data.clip_end = distance * 4
    scene.camera = camera
    for name, location, power in (
            ("Key", target + normal * distance * 0.65 + up * distance * 0.3, 1800),
            ("Fill", target + normal * distance * 0.35 - right * distance * 0.4, 1100)):
        bpy.ops.object.light_add(type="AREA", location=location)
        light = bpy.context.object
        light.name = name
        light.rotation_euler = (target - light.location).to_track_quat("-Z", "Y").to_euler()
        light.data.energy = power
        light.data.shape = "DISK"
        light.data.size = distance * 0.5
    scene.world = bpy.data.worlds.new("Molecular background")
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0., 0., 0., 1)
    background.inputs["Strength"].default_value = 0.0
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution // 2
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "AgX"
    return scene


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--shader", type=Path)
    parser.add_argument("--mode", choices=("amplitude", "probability"), default="amplitude")
    parser.add_argument("--no-bonds", action="store_true")
    parser.add_argument("--bond-factor", type=float, default=1.2)
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--padding", type=float, default=4.0, help="Volume boundary padding in angstrom")
    parser.add_argument("--resolution", type=int, default=960, help="Image width; height is half the width")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args(argv)
    job = read_input(args.input)
    scene = create_scene(job, bonds=not args.no_bonds, bond_factor=args.bond_factor, shader=args.shader,
                         mode=args.mode, density=args.density, threshold=args.threshold, padding=args.padding,
                         resolution=args.resolution, samples=args.samples)
    output = (args.output or args.input.parent / "results" / f"{args.input.stem}_structure.blend").resolve()
    if output.suffix.lower() != ".blend":
        raise ValueError("Output must be a .blend file")
    output.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(output.with_suffix(".png"))
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    print(f"Created {scene['atom_count']} atoms and {scene['bond_count']} bonds: {output}")
    if args.render:
        bpy.ops.render.render(write_still=True, scene=scene.name)


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
