# =========================
# Factories
# =========================

from enum import Enum
from SegmentationStrategy import *
from AreaCalculatorStrategy import *


class SegmentationMethod(Enum):
    DEEP = "deep"
    CLASSICAL = "classical"


class AreaMethod(Enum):
    PIXEL = "pixel"
    CALIBRATED = "calibrated"


class SegmenterFactory:
    def create(self, method: SegmentationMethod, **kwargs) -> Segmenter:
        if method == SegmentationMethod.DEEP:
            return DeepLearningSegmenter(model_path=kwargs.get("model_path", "model.onnx"))
        elif method == SegmentationMethod.CLASSICAL:
            return ClassicalSegmenter()
        else:
            raise ValueError(f"Unknown segmentation method: {method}")


class AreaCalculatorFactory:
    def create(self, method: AreaMethod, **kwargs) -> AreaCalculator:
        if method == AreaMethod.PIXEL:
            return PixelAreaCalculator()
        elif method == AreaMethod.CALIBRATED:
            pixel_to_cm = kwargs.get("pixel_to_cm")
            if pixel_to_cm == 0 or pixel_to_cm is None:
                raise ValueError("pixel_to_cm is required for CALIBRATED area method")
            return CalibratedAreaCalculator(pixel_to_cm=pixel_to_cm)
        else:
            raise ValueError(f"Unknown area method: {method}")