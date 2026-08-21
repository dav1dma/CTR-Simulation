# CAD assets

Place the robot assembly here for the digital-twin integration. Prefer one GLB
assembly that preserves named component nodes. If that export is unavailable,
use separate STL or OBJ meshes for each rigid moving group.

Required moving groups:

- fixed base and frame
- inner, middle, and outer linear carriages
- inner, middle, and outer hollow shafts or chucks
- optional lead screws or actuator rods

Before export:

- use millimetres;
- place the tube insertion point at `(0, 0, 0)`;
- align undeformed tubes with global `+Z`;
- position each component origin on its intended translation or rotation axis;
- keep parts with different motion separate;
- remove unnecessary fastener detail and dense thread geometry.

Do not commit confidential CAD or very large meshes without first deciding
whether the repository is suitable or Git LFS is required.
