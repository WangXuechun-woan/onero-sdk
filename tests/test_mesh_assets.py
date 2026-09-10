"""Regression tests for the canonical A1 mesh layout."""

from pathlib import Path
import xml.etree.ElementTree as ET


_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "onero_sdk"
_URDF_ROOT = _PACKAGE_ROOT / "urdf" / "A1"
_MESH_ROOT = _PACKAGE_ROOT / "meshes" / "A1"
_ROS2_RIGHT_MESHES = (
    _PACKAGE_ROOT.parent
    / "ros2"
    / "onero_ros2"
    / "description"
    / "meshes"
    / "a1_r"
)


def test_a1_urdfs_reference_canonical_meshes():
    for side in ("l", "r"):
        urdf = _URDF_ROOT / f"a1_{side}.urdf"
        root = ET.parse(urdf).getroot()
        references = [
            mesh.attrib["filename"]
            for mesh in root.iter("mesh")
            if "filename" in mesh.attrib
        ]

        assert len(references) == 16
        expected_dir = (_MESH_ROOT / f"A1_{side}").resolve()
        for reference in references:
            resolved = (urdf.parent / reference).resolve()
            assert resolved.is_file(), f"missing mesh referenced by {urdf}: {reference}"
            assert resolved.parent == expected_dir


def test_a1_stls_have_one_canonical_copy():
    canonical = sorted(_MESH_ROOT.rglob("*.STL"))
    assert len(canonical) == 16
    assert not list(_URDF_ROOT.rglob("*.STL"))
    assert not list(_URDF_ROOT.rglob("*.stl"))


def test_ros2_uses_canonical_right_arm_meshes():
    assert _ROS2_RIGHT_MESHES.is_symlink()
    assert _ROS2_RIGHT_MESHES.resolve() == (_MESH_ROOT / "A1_r").resolve()
