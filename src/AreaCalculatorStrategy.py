# =========================
# AreaCalculator Strategy
# =========================


from abc import ABC, abstractmethod
import numpy as np
from ObjectFilterStrategy import ObjectInfo


class AreaCalculator(ABC):
    @abstractmethod
    def volume_from_mask(self, obj: ObjectInfo) -> float:
        pass


class PixelAreaCalculator(AreaCalculator):
    def volume_from_mask(self, obj: ObjectInfo) -> float:
        pass


class CalibratedAreaCalculator(AreaCalculator):

    def __init__(self, pixel_to_cm: float, broiler_density: float = 900.0):  # density kg/m3
        self.pixel_to_cm = pixel_to_cm
        self.density = broiler_density

    def volume_from_mask(self, obj: ObjectInfo) -> ObjectInfo:

        mask = obj.mask
        mask_bool = mask > 0

        if not mask_bool.any():
            return 0.0

        H, W = mask_bool.shape

        col_has_obj = mask_bool.any(axis=0)
        if not col_has_obj.any():
            return 0.0

        mask_valid_cols = mask_bool[:, col_has_obj]

        y_min = mask_valid_cols.argmax(axis=0)

        flipped = np.flipud(mask_valid_cols)
        y_max_from_bottom = flipped.argmax(axis=0)
        y_max = (H - 1) - y_max_from_bottom
        width = (y_max - y_min + 1).astype(np.float32)
        radius = width * self.pixel_to_cm / 2.0
        V = float(np.pi * np.sum(radius ** 2) * self.pixel_to_cm)

        obj.volume = (V / 1_000_000) * self.density  # convert to cubic meters

        return obj