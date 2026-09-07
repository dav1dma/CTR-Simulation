# Configuration

`analysis_protocol_v1.toml` is the versioned methodological source of truth for
the CTR positional design study. It records the agreed scope, nominal baseline,
canonical sampling policy, rotational-symmetry rules, Stage-2 Jacobian and
spatial definitions, provisional IK solver settings, and the gates that keep
optimisation disabled until the full baseline methodology has been validated.
Each future formal run will embed a resolved
copy and semantic hash of this file in its manifest.

The directory will also hold CAD joint and actuator calibration data. Those
files will record each moving component's node name, axis, home offset,
movement direction, travel limit, shaft zero angle, and any screw lead or gear
ratio rather than hard-coding those values in the viewer. Analysis protocols
and hardware calibration files must remain separately named and versioned.

The later frozen protocols are:

- `analysis_protocol_stage4_extension.toml` — primary-endpoint convergence extension.
- `analysis_protocol_stage4_1_support.toml` — primary-endpoint IK spatial-support completion.
- `optimization_protocol_stage5a.toml` — design-variable bounds, objectives, evidence gates,
  and planned pilot/search/independent-validation stages. A frozen protocol is not
  an optimisation result.
