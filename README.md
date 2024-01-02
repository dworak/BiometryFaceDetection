# OpenCV Feature Detection

This Python script utilizes the OpenCV library to perform feature detection using two different methods: **Harris Corner Detection** and **FAST (Features from Accelerated Segment Test)**. The script includes functionalities to capture frames from a camera or load images from a local file for feature detection.

## Harris Corner Detection

Harris Corner Detection is applied to identify corners in an image. The script includes functions for both camera input and local file input.

### Usage

#### Camera Input

```python
getImageFromCameraCornerHarris()
```

#### Local File Input
```python
getImageFromLocalFileCornerHarris(filepath)
```

#### FAST Feature Detection
FAST is a corner detection algorithm that selects key points in an image based on pixel intensities. The script includes functions for both camera input and local file input.

### Usage
```python
getImageFromCameraFAST()
```
#### Local File Input
```python
getImageFromLocalFileFAST(filepath)
```

#### Dependencies
NumPy
OpenCV
Matplotlib

#### How to Run
Ensure you have the required dependencies installed: NumPy, OpenCV, Matplotlib.
Uncomment the desired function call at the end of the script (getImageFromLocalFileCornerHarris, getImageFromCameraCornerHarris, getImageFromLocalFileFAST, getImageFromCameraFAST).
Run the script by executing python your_script_name.py.
Feel free to experiment with different images and parameters to explore the capabilities of Harris Corner Detection and FAST Feature Detection.

Note: This script is provided for educational purposes and may require adjustments based on your specific use case.
