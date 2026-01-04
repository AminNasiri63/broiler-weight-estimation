# Broiler Weight Estimation

A production-oriented computer vision pipeline for estimating broiler chicken weight from video streams.
The system is designed for real-world poultry houses, emphasizing robustness, interpretability, and low computational cost.

Unlike deep-learning–heavy solutions, this project focuses on classical CV and geometric modeling, making it suitable for edge deployment, rapid prototyping, and environments with limited labeled data.

---

## Demo Video

https://github.com/user-attachments/assets/fe745700-ba96-4303-a332-41783033921b

---

## Why This Project Matters

- Enables non-contact weight monitoring in poultry farms
- Reduces reliance on manual weighing
- Designed for continuous, camera-based operation
- Prioritizes explainability and engineering simplicity
- This makes it relevant for:
- AgTech & Precision Livestock companies
- Edge CV / applied computer vision roles
---

## Features

- **End-to-End Video Pipeline**: Frame ingestion, ROI handling, segmentation, object validation, and weight estimation
- **Classical, Learning-Free Segmentation**: HSV-based foreground extraction with morphological refinement
- **Robust Object Selection**: Overlap rejection, confidence scoring, and configurable Top-K candidate selection
- **Pose-Normalized, Density-Aware Weight Estimation**: Orientation normalization + geometry-based volume computation, converted to weight using configurable broiler density (default: 900 kg/m³).
- **Structured Outputs**: CSV logging and annotated video output for debugging and offline analysis
- **Modular Architecture**: Strategy and Factory patterns enabling extensibility and future DL integration

---

## Project Structure

```
├── src
    ├── main execution.py          # Main entry point
    ├── VideoProcessorPipeline.py  # Core video processing pipeline
    ├── CameraLoader.py           # Camera configuration loader
    ├── SegmentationStrategy.py   # Segmentation implementations (Classical/Deep Learning)
    ├── ObjectFilterStrategy.py   # Object filtering and scoring logic
    ├── AreaCalculatorStrategy.py # Volume/weight calculation strategies
    ├── OutputWriters.py          # Output formatters (CSV)
    ├── Factories.py              # Factory classes for strategy creation
    ├── VideoIOClass.py           # Video input/output utilities
    ├── cameras_config.json       # Camera configuration file
    ├── broiler_template.png      # Template mask for broiler matching
    └── cam1.mp4                  # Sample video file
├── requirements.txt
└── README.md
```
---

## Installation

### Prerequisites

- Python 3.7+
- OpenCV 4.x
- NumPy

### Dependencies

```bash
pip install opencv-python numpy
```
---

## Configuration

The system is configured via `cameras_config.json`. Each camera entry defines:

- `crop`: Region of interest coordinates (x1, y1, x2, y2)
- `feeder`: Feeder area coordinates (for calibration)
- `pixet_to_cm`: Pixel-to-centimeter calibration factor

Example configuration:

```json
{
  "cameras": {
    "cam1": {
      "crop": {"x1": 1900, "y1": 590, "x2": 3180, "y2": 1475},
      "feeder": {"x1": 1700, "y1": 700, "x2": 2250, "y2": 1100},
      "pixet_to_cm": 0.09
    }
  }
}
```
### Camera Calibration

Accurate weight estimation requires pixel-to-real-world calibration (pixel_to_cm in `cameras_config.json` file).  
In this project, the **Segment Anything Model (SAM)** is used to segment the feeder region from the image, producing a clean, reliable mask of a known reference object.
In the current implementation, the pixel-to-centimeter conversion factor is **set manually** based on the segmented feeder dimensions.
In commercial farm environments, feeders are well-suited for calibration because their **geometry and physical dimensions are typically fixed and known**. This makes feeder-based calibration a practical, repeatable approach that does not require additional calibration targets.





---

## Usage

1. Configure camera parameters in `cameras_config.json`
2. Place input video files in the project directory
3. (Optional) Adjust the Top-K candidate selection via the `top_k` parameter in the object filtering stage
4. Update the video path in `main_execution.py` if needed
5. Run the main script:

```bash
python "main execution.py"
```

The system processes the input video and saves an annotated output video along with a camera-specific CSV file named `<camera_id>_results.csv`.

### Output Format

The CSV output contains the following columns:

| Field        | Description |
|-------------|-------------|
| `camera_id` | Camera identifier |
| `frame_idx` | Frame number |
| `label`     | Object label (default: `"broiler"`) |
| `score`     | Combined confidence score |
| `volume`    | Estimated weight (kg) |
| `crop_region` | Bounding box coordinates `(x, y, w, h)` |

---

## Known Tradeoffs
**These are deliberate design choices aimed at reliability, transparency, and low deployment cost.**

- **Simplified Body Model:** Weight estimation is based on a cylindrical body assumption to ensure computational efficiency and interpretability.
- **Lighting Sensitivity:** Classical segmentation performs best under controlled illumination, which is a common constraint in commercial poultry houses.
- **Manual Calibration:** Pixel-to-centimeter calibration is currently configured on a per-camera basis to maintain measurement accuracy.
- **Template Dependency:** Shape-based validation relies on orientation-consistent templates, prioritizing precision over recall.

---

## Roadmap

- Integration of deep learning models for improved segmentation
- Posture-aware frame selection to improve volume-based weight estimation accuracy
- Multi-species support with configurable templates
- Web interface for configuration and monitoring
- Statistical analysis and reporting features
---

## License
MIT License.

---

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
