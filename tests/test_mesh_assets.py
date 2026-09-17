"""Regression tests for the canonical A1 mesh layout."""

from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET


_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "onero_sdk"
_URDF_ROOT = _PACKAGE_ROOT / "urdf" / "A1"
_MESH_ROOT = _PACKAGE_ROOT / "meshes" / "A1"
_TOOL_ROOT = _PACKAGE_ROOT / "tools"
_GRIPPER_MESH_ROOT = _PACKAGE_ROOT / "meshes" / "grippers"
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


def test_tool_urdfs_reference_packaged_gripper_meshes():
    for model in ("omnipicker", "woan_picker"):
        urdf = _TOOL_ROOT / f"{model}.urdf"
        root = ET.parse(urdf).getroot()
        references = sorted({
            mesh.attrib["filename"]
            for mesh in root.iter("mesh")
            if "filename" in mesh.attrib
        })
        assert len(references) == 9
        assert len(list((_GRIPPER_MESH_ROOT / model).glob("*.STL"))) == 9
        for reference in references:
            assert reference.startswith(f"meshes/grippers/{model}/")
            resolved = (_PACKAGE_ROOT / reference).resolve()
            assert resolved.is_file(), f"missing {model} mesh: {reference}"

        revolute = [j for j in root.findall("joint") if j.attrib["type"] == "revolute"]
        assert len(revolute) == (6 if model == "omnipicker" else 8)
        if model == "omnipicker":
            for joint in revolute:
                limit = joint.find("limit")
                assert limit.attrib["lower"] == "-0.45"
                assert limit.attrib["upper"] == "0.45"


def test_woan_picker_meshes_match_validated_h1_v13_assets():
    expected = {
        "base_link.STL": "1749a4d15f",
        "Link_L1.STL": "6da0382135",
        "Link_L2.STL": "f8ee5dfab8",
        "Link_L3.STL": "8d4183cbfa",
        "Link_L4.STL": "bdab510df0",
        "Link_R1.STL": "b080d275d4",
        "Link_R2.STL": "e206fa332f",
        "Link_R3.STL": "6ed20a3d57",
        "Link_R4.STL": "91f5f812ad",
    }
    root = _GRIPPER_MESH_ROOT / "woan_picker"
    actual = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()[:10]
        for path in root.glob("*.STL")
    }
    assert actual == expected


def test_woan_picker_uses_validated_h1_v13_kinematics():
    root = ET.parse(_TOOL_ROOT / "woan_picker.urdf").getroot()
    base = root.find("./link[@name='woan_picker_base_link']/visual")
    assert base.find("origin").attrib == {"xyz": "0 0 0", "rpy": "0 0 0"}
    assert base.find("geometry/mesh").attrib["filename"].endswith(
        "/woan_picker/base_link.STL")

    joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}
    assert len(joints) == 8
    left = joints["woan_picker_left_finger_joint"]
    right = joints["woan_picker_right_finger_joint"]
    assert left.find("origin").attrib == {"xyz": "0.03 0 0", "rpy": "0 0 -1.1"}
    assert right.find("origin").attrib == {"xyz": "-0.03 0 0", "rpy": "0 0 1.1"}
    assert left.find("limit").attrib["upper"] == "1.1"
    assert joints["woan_picker_left_finger_joint_4"].find("limit").attrib["upper"] == "1.27"


def test_ros2_uses_canonical_right_arm_meshes():
    assert _ROS2_RIGHT_MESHES.is_symlink()
    assert _ROS2_RIGHT_MESHES.resolve() == (_MESH_ROOT / "A1_r").resolve()
