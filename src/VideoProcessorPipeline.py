# =========================
# VideoProcessor (Pipeline)
# =========================

from typing import Optional
from AreaCalculatorStrategy import AreaCalculator
from ObjectFilterStrategy import ObjectFilter
from OutputWriters import OutputWriter
from SegmentationStrategy import *
from CameraLoader import *
from VideoIOClass import *


class VideoProcessor:

    def __init__(
        self,
        camera_config: CameraConfig,
        segmenter: Segmenter,
        area_calculator: AreaCalculator,
        obj_filter: ObjectFilter,
        output_writer: OutputWriter,
        show_img: bool = True,
        max_frames: Optional[int] = None
    ):
        self.config = camera_config
        self.segmenter = segmenter
        self.area_calculator = area_calculator
        self.obj_filter = obj_filter
        self.output_writer = output_writer
        self.max_frames = max_frames
        self.video_reader = VideoReader(str(camera_config.video_path))
        self.show_window = show_img

        fps = self.video_reader.get_info()
        self.x1, self.y1, self.x2, self.y2 = camera_config.crop_region
        width, height = self.x2 - self.x1, self.y2 - self.y1
        self.video_writer = VideoWriter(
            output_path=str(camera_config.video_save_path),
            fps=fps,
            frame_size=(width, height)
        )

    def run(self):

        frame_idx = 0

        window_name = f"Camera: {self.config.camera_id} - Cropped"
        if self.show_window:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 800, 600)

        print(f"[VideoProcessor] Start processing camera: {self.config.camera_id}")

        while True:
            ret, frame = self.video_reader.read()
            if not ret:
                print("[VideoProcessor] End of video or cannot read frame.")
                break

            frame_idx += 1
            if self.max_frames is not None and frame_idx > self.max_frames:
                print(f"[VideoProcessor] Reached max_frames={self.max_frames}. Stopping.")
                break

            # Crop region
            cropped = frame[self.y1:self.y2, self.x1:self.x2]

            # 1) Segmentation
            objects = self.segmenter.segment(cropped)

            # 2) Filtering
            selected_objs, clean_mask = self.obj_filter.should_process(objects)

            # 3) Weight
            for obj in selected_objs:

                obj = self.area_calculator.volume_from_mask(obj)
                x, y, w, h = obj.crop_region
                volume = obj.volume

                cx = x + w // 2
                cy = y + h // 2
                cv2.rectangle(cropped, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(cropped, f"{volume:.2f} kg",
                            (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

                record = {
                    "camera_id": self.config.camera_id,
                    "frame_idx": frame_idx,
                    "label": obj.label,
                    "score": obj.confidence_dt + obj.confidence_tpl,
                    "volume": volume,
                    "crop_region": (x, y, w, h),
                }
                self.output_writer.write(record)

            if self.show_window:
                cv2.imshow(window_name, cropped)
                if cv2.waitKey(33) & 0xFF == ord('q'):
                    print("[VideoProcessor] Quitting due to user input.")
                    break

            self.video_writer.write(cropped)

        self.clean_up()

    def clean_up(self):
        self.video_reader.release()
        self.video_writer.release()
        self.output_writer.close()

        if self.show_window:
            cv2.destroyAllWindows()

        print(f"[VideoProcessor] Finished camera: {self.config.camera_id}")