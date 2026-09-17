"""Visual and physical metadata for tools mounted on the A1 flange.

Tool descriptions are intentionally separate from the canonical arm URDFs.
They may drive Debug UI visualization and, after hardware validation, provide
payload/TCP defaults.  Merely selecting a visual model does not alter dynamics
or the Arm TCP.
"""

from __future__ import annotations

from copy import deepcopy

from .exceptions import OneroConfigError


_TOOL_MODELS = {
    "omnipicker": {
        "name": "omnipicker",
        "urdf": "omnipicker.urdf",
        # Fixed mount from the supplied A1L_OmniPicker URDF.
        "mount_xyz": [0.0, 0.0, 0.0],
        "mount_rpy": [0.0, 0.0, -1.5708],
        # Aggregated from the supplied fixed-link inertials and expressed in
        # the A1 Link7 frame. Informational until validated on both arm sides.
        "mass": 0.41689308,
        "cog": [0.0005728365, -0.0000112533, 0.0559801278],
        # The supplied model has no authoritative grasp-frame definition.
        "tcp": None,
        "animated": True,
        "opening_range": [0.0, 1.0],
        "opening_default": 0.5,
        # Visual approximation because the supplied OmniPicker URDF freezes
        # every linkage joint at one pose.
        "opening_joints": [
            {"suffix": "narrow1_joint", "offset": -0.35, "scale": 0.70},
            {"suffix": "narrow2_joint", "offset": 0.35, "scale": -0.70},
            {"suffix": "narrow_loop_joint", "offset": -0.35, "scale": 0.70},
            {"suffix": "wide1_joint", "offset": 0.35, "scale": -0.70},
            {"suffix": "wide2_joint", "offset": -0.35, "scale": 0.70},
            {"suffix": "wide_loop_joint", "offset": 0.35, "scale": -0.70},
        ],
    },
    "woan_picker": {
        "name": "woan_picker",
        "urdf": "woan_picker.urdf",
        # Exact mount from the validated H1 v1.3 assembled model. Both sides
        # use this transform from their respective Link7 to the gripper base.
        "mount_xyz": [0.0, 0.0, 0.07],
        # The validated H1 v1.3 assembled URDF uses this same Link7→tool
        # transform on both A1L and A1R. Their mirrored Link7 frames already
        # encode the side difference; applying another mirror would be wrong.
        "mount_rpy": [-1.5708, 0.0, -1.5708],
        "mass": 0.14679748577740057,
        "cog": [-0.0001852472, -0.0000109078, 0.0838795628],
        "tcp": None,
        "animated": True,
        "opening_range": [0.0, 1.0],
        "opening_default": 0.0,
        # Exact 0–100% coupling from the validated H1 v1.3 viewer model.
        "opening_joints": [
            {"suffix": "left_finger_joint", "offset": 0.0, "scale": 1.10},
            {"suffix": "left_finger_joint_2", "offset": 0.0, "scale": 0.37},
            {"suffix": "left_finger_joint_3", "offset": 0.0, "scale": 1.34},
            {"suffix": "left_finger_joint_4", "offset": 0.0, "scale": 1.27},
            {"suffix": "right_finger_joint", "offset": 0.0, "scale": 1.10},
            {"suffix": "right_finger_joint_2", "offset": 0.0, "scale": 0.37},
            {"suffix": "right_finger_joint_3", "offset": 0.0, "scale": 1.34},
            {"suffix": "right_finger_joint_4", "offset": 0.0, "scale": 1.27},
        ],
    },
}


def available_tool_models() -> tuple[str, ...]:
    """Return registered tool model identifiers."""
    return tuple(sorted(_TOOL_MODELS))


def get_tool_model(name: str | None, side: str | None = None) -> dict | None:
    """Return a side-resolved tool description, or None for no visual tool."""
    if name is None or str(name).lower() in ("", "none", "off"):
        return None
    key = str(name).lower()
    try:
        model = deepcopy(_TOOL_MODELS[key])
    except KeyError as exc:
        raise OneroConfigError(
            f"Unknown tool model '{name}'. Available: "
            f"{', '.join(available_tool_models())}"
        ) from exc
    if side is not None:
        mounts = model.pop("mount_rpy_by_side", None)
        if mounts is not None:
            if side not in mounts:
                raise OneroConfigError(f"Unknown arm side '{side}' for tool '{name}'")
            model["mount_rpy"] = mounts[side]
    return model
