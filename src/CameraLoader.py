# =========================
# Camera Config & Loader
# =========================

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import json


@dataclass
class CameraConfig:
    camera_id: str
    video_path: Path
    video_save_path: Path
    crop_region: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    pixel_to_cm: float


class CameraConfigLoader:

    def get_camera_info(self, camera_id: str, video_path: str, video_save_path: str,
                        camera_config: str) -> CameraConfig:
        camera_id = camera_id.lower()

        if not Path(camera_config).is_file():
            raise FileNotFoundError(f"Camera config file not found: {camera_config}")

        with open(camera_config, "r") as f:
            cameras_dict = json.load(f)

        cameras = cameras_dict.get("cameras", {})

        if camera_id not in cameras:
            raise ValueError(f"Camera '{camera_id}' not found in config file.")

        crop = cameras[camera_id].get("crop")
        if crop is None:
            raise ValueError(f"Camera '{camera_id}' has no crop region defined.")

        pixel_to_cm = cameras[camera_id].get("pixet_to_cm")
        if pixel_to_cm == 0:
            raise ValueError(f"Camera '{camera_id}' has no pixel_to_cm defined.")

        return CameraConfig(
            camera_id=camera_id,
            video_path=Path(video_path),
            video_save_path=Path(video_save_path),
            crop_region=(int(crop['x1']), int(crop['y1']), int(crop['x2']), int(crop['y2'])),
            pixel_to_cm=pixel_to_cm
        )

