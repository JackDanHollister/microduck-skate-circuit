# Microduck roller model

The Microduck robot design and mesh geometry are by **Pollen Robotics**. I converted the visual geometry distributed in Microduck Arena 0.7.0 into indexed binary meshes, keeping the body hierarchy, joint pivots, colours and geometry placements for browser replay.

The adapted mesh geometry in [duck-3d.bin](duck-3d.bin) is © Pollen Robotics and shared under **Creative Commons Attribution–NonCommercial–ShareAlike 4.0 International**. [Original robot model](https://github.com/pollen-robotics/microduck_rl/tree/0daab160d3fb1c2cbaba16a4a19052f36709dd9e/src/mjlab_microduck/robot/microduck/assets) · [Pollen’s model licence and attribution instructions](duck-source-mesh-license.md) · [Full CC licence](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)

The [joint description](duck-3d.json) is derived from Pollen’s Apache 2.0 MJCF robot description. Coordinates are in metres, relative to the parent link; quaternions use MuJoCo’s WXYZ order. [Description licence](duck-description-license.txt) · [Source snapshot used by Arena](https://huggingface.co/spaces/pollen-robotics/microduck-simulator/tree/10c600e54ea52f239887308387cb8fc458ecb3e4)

The replay contains the body pose and all 14 actuated joint angles. Wheel locations follow the articulated skate assemblies. The four unpowered wheel spin angles were not recorded and are displayed at their reference angles.

Run `python tools/build_duck_3d.py` from the repository root with the racing requirements installed to reproduce the binary and rig and check their world-space geometry poses against MuJoCo.

The nearby-ghost geometry in [duck-3d-mid.bin](duck-3d-mid.bin) and distant-ghost geometry in [duck-3d-low.bin](duck-3d-low.bin) is the same Pollen mesh set, simplified part by part, with the same CC BY-NC-SA 4.0 licence as the detailed binary. Both reduced rigs use the same body and joint transforms. To regenerate this optional level of detail, install `fast-simplification==0.1.12` and run `python tools/build_duck_3d.py --low-detail`.
Use `--mid-detail` for the nearby-ghost version.
