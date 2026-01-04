# broiler-weight-estimation

A production-oriented computer vision pipeline for estimating broiler chicken weight from video streams.
The system is designed for real-world poultry houses, emphasizing robustness, interpretability, and low computational cost.

Unlike deep learning–heavy solutions, this project focuses on classical CV + geometric modeling, making it suitable for edge deployment, rapid prototyping, and environments with limited labeled data.

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

- **Video Processing Pipeline**: Processes video frames from configured cameras with cropping and region-of-interest handling
- **Classical Segmentation**: Uses HSV thresholding and morphological operations to segment broiler objects from background
- **Object Detection & Filtering**: Advanced filtering using template matching, distance transforms, and confidence scoring to identify individual broilers
- **Weight Calculation**: Estimates broiler weight assuming cylindrical body shape with calibrated pixel-to-centimeter conversion
- **CSV Output**: Logs detection results including frame number, confidence scores, estimated weights, and crop regions
- **Extensible Architecture**: Strategy pattern implementation allowing for easy extension with deep learning models
- **Design Patterns**: Strategy, Factory
- **Deployment Target**: Desktop, edge devices (Jetson / Raspberry Pi class)
---

## Project Structure

```
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
---

## Usage

1. Configure your cameras in `cameras_config.json`
2. Place your video files in the project directory
3. Update the video path in `main execution.py` if necessary
4. Run the main script:

```bash
python "main execution.py"
```

The system will process the video and output results to `cam1_results.csv`.

### Output Format

The CSV output contains the following columns:

- `camera_id`: Camera identifier
- `frame_idx`: Frame number
- `label`: Object label (default: "broiler")
- `score`: Combined confidence score
- `volume`: Estimated weight in kg
- `crop_region`: Bounding box coordinates (x, y, w, h)
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
- Multi-species support with configurable templates
- Real-time processing optimization
- Web interface for configuration and monitoring
- Statistical analysis and reporting features
---

## License

This project is provided as-is for research and development purposes.
