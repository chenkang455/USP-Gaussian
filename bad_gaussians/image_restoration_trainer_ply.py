"""
Trainer extension that saves the trained Gaussians as a .ply file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple, Type, Union

import numpy as np
import torch

try:
    from bad_gaussians.image_restoration_trainer import ImageRestorationTrainer
except Exception:  # pragma: no cover - fallback for alternate package layouts
    from nerfstudio.engine.trainer import Trainer as ImageRestorationTrainer


def _is_useful_value(value: Any) -> bool:
    if value is None:
        return False
    if callable(value):
        return False
    if isinstance(value, (str, bytes, int, float, bool, dict)):
        return False
    return True


def _get_first(obj: Any, names: Tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        try:
            value = getattr(obj, name)
        except AttributeError:
            continue
        if _is_useful_value(value):
            return value
    return default


def _to_feature_tensor(value: Any) -> torch.Tensor:
    if value is None:
        return torch.zeros((0, 0), dtype=torch.float32)
    if isinstance(value, (list, tuple, torch.nn.ParameterList)):
        parts: List[torch.Tensor] = []
        for item in value:
            tensor = _to_feature_tensor(item)
            if tensor.numel() > 0:
                parts.append(tensor)
        if not parts:
            return torch.zeros((0, 0), dtype=torch.float32)
        return torch.cat(parts, dim=1)
    tensor = torch.as_tensor(value)
    return tensor


def _reshape_to_rows(arr: Any, cols: int, n: Optional[int] = None) -> np.ndarray:
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().float().numpy()
    else:
        arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        if cols > 0 and arr.size % cols == 0:
            arr = arr.reshape(-1, cols)
        else:
            arr = arr.reshape(1, -1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)

    if n is None:
        n = arr.shape[0]
    out = np.zeros((n, cols), dtype=np.float32)
    rows = min(arr.shape[0], n)
    copy_cols = min(arr.shape[1], cols)
    if rows > 0 and copy_cols > 0:
        out[:rows, :copy_cols] = arr[:rows, :copy_cols]
    return out


def _collect_containers(root: Any) -> List[Any]:
    containers: List[Any] = []
    seen = set()
    stack: List[Any] = [root]
    while stack:
        obj = stack.pop()
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        containers.append(obj)

        if isinstance(obj, torch.nn.Module):
            containers.extend(list(obj.modules()))

        try:
            attributes = vars(obj)
        except TypeError:
            attributes = {}

        for value in attributes.values():
            if isinstance(value, (torch.nn.Module, list, tuple, set, dict)):
                if isinstance(value, (list, tuple, set)):
                    stack.extend(value)
                elif isinstance(value, dict):
                    stack.extend(value.values())
                else:
                    stack.append(value)
            elif hasattr(value, "__dict__") and not isinstance(
                value, (torch.Tensor, torch.nn.Parameter, np.ndarray, str, bytes, int, float, bool)
            ):
                stack.append(value)
    return containers


_MEANS_NAMES = ("means", "_means", "gaussian_means", "centers", "positions")
_OPACITY_NAMES = ("opacities", "_opacities", "opacity", "opac")
_SCALE_NAMES = ("scales", "_scales", "scale", "log_scales")
_QUAT_NAMES = ("quats", "_quats", "rot", "rotation")
_FEATURES_DC_NAMES = ("features_dc", "_features_dc", "dc")
_FEATURES_REST_NAMES = ("features_rest", "_features_rest", "features_rest_list", "sh_rest")


def _find_gaussian_container(model: Any) -> Optional[Any]:
    for obj in _collect_containers(model):
        if (
            _get_first(obj, _MEANS_NAMES) is not None
            and _get_first(obj, _OPACITY_NAMES) is not None
            and _get_first(obj, _SCALE_NAMES) is not None
            and _get_first(obj, _QUAT_NAMES) is not None
            and _get_first(obj, _FEATURES_DC_NAMES) is not None
        ):
            return obj
    return None


def extract_gaussian_arrays(model: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    container = _find_gaussian_container(model)
    if container is None:
        raise RuntimeError("Could not locate Gaussian parameters on the model to export a PLY file.")

    means = _reshape_to_rows(_to_feature_tensor(_get_first(container, _MEANS_NAMES)), cols=3)
    n = means.shape[0]
    opacities = _reshape_to_rows(_to_feature_tensor(_get_first(container, _OPACITY_NAMES)), cols=1, n=n)[:, 0]
    scales = _reshape_to_rows(_to_feature_tensor(_get_first(container, _SCALE_NAMES)), cols=3, n=n)
    quats = _reshape_to_rows(_to_feature_tensor(_get_first(container, _QUAT_NAMES)), cols=4, n=n)
    features_dc = _reshape_to_rows(_to_feature_tensor(_get_first(container, _FEATURES_DC_NAMES)), cols=3, n=n)
    features_rest = _reshape_to_rows(_to_feature_tensor(_get_first(container, _FEATURES_REST_NAMES, default=torch.zeros((0, 0)))), cols=45, n=n)

    opacities = 1.0 / (1.0 + np.exp(-opacities))

    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    quats = quats / norms

    return means, opacities, scales, quats, features_dc, features_rest


def save_gaussians_ply(model: Any, output_path: Union[str, Path]) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    means, opacities, scales, quats, features_dc, features_rest = extract_gaussian_arrays(model)
    n = means.shape[0]

    means = means.astype(np.float32)
    opacities = opacities.astype(np.float32)
    scales = scales.astype(np.float32)
    quats = quats.astype(np.float32)
    features_dc = features_dc.astype(np.float32)
    features_rest = features_rest.astype(np.float32)

    normals = np.zeros((n, 3), dtype=np.float32)
    c0 = 0.28209479177387814
    rgb = 0.5 + c0 * features_dc
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = (rgb * 255.0).astype(np.uint8)

    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n}",
    ]
    for name in ("x", "y", "z", "nx", "ny", "nz"):
        header_lines.append(f"property float {name}")
    for name in ("red", "green", "blue"):
        header_lines.append(f"property uchar {name}")
    header_lines.append("property float opacity")
    for name in ("scale_0", "scale_1", "scale_2"):
        header_lines.append(f"property float {name}")
    for name in ("rot_0", "rot_1", "rot_2", "rot_3"):
        header_lines.append(f"property float {name}")
    for name in ("f_dc_0", "f_dc_1", "f_dc_2"):
        header_lines.append(f"property float {name}")
    for i in range(45):
        header_lines.append(f"property float f_rest_{i}")
    header_lines.append("end_header")
    header = "\n".join(header_lines) + "\n"

    arrays: List[np.ndarray] = [
        means[:, 0],
        means[:, 1],
        means[:, 2],
        normals[:, 0],
        normals[:, 1],
        normals[:, 2],
        rgb[:, 0],
        rgb[:, 1],
        rgb[:, 2],
        opacities,
        scales[:, 0],
        scales[:, 1],
        scales[:, 2],
        quats[:, 0],
        quats[:, 1],
        quats[:, 2],
        quats[:, 3],
        features_dc[:, 0],
        features_dc[:, 1],
        features_dc[:, 2],
    ]
    arrays.extend(features_rest[:, i] for i in range(45))

    with open(output_path, "wb") as f:
        f.write(header.encode("ascii"))
        for arr in arrays:
            f.write(np.ascontiguousarray(arr).tobytes())

    return output_path


class ImageRestorationTrainerWithPly(ImageRestorationTrainer):
    """Trainer that saves the trained Gaussians as a .ply file after training."""
