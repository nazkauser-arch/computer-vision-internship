## DAY 01
### 1. OpenCV
OpenCV is a python library used to process, analyze and images and videos

### 2. Width, Height and Channels
- Width is no of pixels from left to right
- Height is no of pixels from top to bottom
- Channels represents the no of stored values for each pixel

### 3. Grayscale
- Converting a colored image into shades of gray
- 0 to 255

### 4. Edge detection
Edge detection finds the boundaries/ edges of object in an image

### 5. BGR vs RGB
OpenCV uses BGR to read image while matplotlib uses RGB to display image

### 6. Processed Images
The processed images represents the follwing:
- Resized: same as original but with different dimensions
- Edge: Boundaries and outlines
- Grayscale: Color information is lost

## Questions & Answers

1. **Why does a color image normally have three channels?**  
   A color image uses three channels (BGR/RGB) to represent the intensity of its three color components.

2. **Why does a grayscale image have one channel?**  
   A grayscale image uses one channel to represent the brightness or intensity of each pixel.

3. **What happens when the Canny thresholds are too low?**  
   Too many edges, including noise and unwanted details, are detected.

4. **What happens when the Canny thresholds are too high?**  
   Weak or important edges may not be detected.

5. **Why must the image colors be converted before displaying with Matplotlib?**  
   OpenCV uses BGR while Matplotlib expects RGB, so colors appear incorrect without conversion.

6. **What information is lost when converting an image to grayscale?**  
   Color information is lost, while intensity information is retained.

## Installation

### 1. Clone the repository

```bash
git clone <https://github.com/nazkauser-arch/computer-vision-internship>
cd <computer-vision-internship/image_processing_basics>
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

**Windows:**

```bash
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
source .venv/bin/activate
```

### 4. Install the dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the project

```bash
python main.py
```

## Requirements

* Python 3.x
* OpenCV
* Matplotlib
* NumPy