# =========================
# Main Execution
# =========================

from CameraLoader import CameraConfigLoader
from Factories import AreaCalculatorFactory, AreaMethod, SegmentationMethod, SegmenterFactory
from ObjectFilterStrategy import SimpleObjectFilter
from OutputWriters import CSVOutputWriter
from VideoProcessorPipeline import VideoProcessor


def main():

    loader = CameraConfigLoader()

    cam = loader.get_camera_info(
        camera_id="cam1",
        video_path="cam1.mp4",
        video_save_path="cam1_output.mp4",
        camera_config="cameras_config.json"
    )

    seg_factory = SegmenterFactory()
    area_factory = AreaCalculatorFactory()

    segmenter = seg_factory.create(
        SegmentationMethod.CLASSICAL
    )

    obj_filter = SimpleObjectFilter(
        target_labels=["Broiler"],
        template_mask="broiler_template.png",
        min_area=500.0
    )

    area_calc = area_factory.create(
        AreaMethod.CALIBRATED,
        pixel_to_cm=cam.pixel_to_cm
    )

    output_writer = CSVOutputWriter(csv_path=f"{cam.camera_id}_results.csv")

    processor = VideoProcessor(
        camera_config=cam,
        segmenter=segmenter,
        area_calculator=area_calc,
        obj_filter=obj_filter,
        output_writer=output_writer,
        max_frames=None
    )
    processor.run()


if __name__ == "__main__":
    main()
