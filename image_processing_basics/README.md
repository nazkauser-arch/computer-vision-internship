## DAY 01
1. **OpenCV**
OpenCV is a python library used to process and analyze images and videos

2. **Width, Height and Channels**
- Width is no of pixels from left to right
- Height is no of pixels from top to bottom
- Channels represents the no of stored values for each pixel

3. **Grayscale**
- Converting a colored image into shades of gray
- 0 to 255

4. **Edge detection**
Edge detection finds the boundaries/ edges of object in an image

5. **BGR vs RGB**
OpenCV uses BGR to read image while matplotlib uses RGB to display image

6. **Processed Images**
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


## DAY 03
1. **Difference between classification and detection**
   Classification identifies what an object is, while detection identifies what objects are present and their individual locations

2. **What does bounding box represent?**
   Bounding box represents location and boundaries of an object

3. **Confidence score**
   Confidence score tells how sure YOLO is that the object belongs to the predicted class

4. **Confidence threshold is increased**
   If confidence threshold is increased, the objects with lower confidence levels are not detected

5. **Inference**
   Inference is the process of using a trained YOLO model to analyze new images or videos and make predictions about the objects present

6. **Pre-trained model**
   A model that has been already trained with a large dataset and can be used to classify objects without training

7. **Difference between YOLO and contour detection**
   YOLO detects objects using a trained model, while contour detection finds object boundaries 

8. **Why might YOLO miss an object**
   If confidence score is too low or if the object is too small, hidden or blurry

9. **Why can YOLO produce an incorrect detection**
   YOLO can produce an incorrect detection if objects are unclear, partially hidden or the model is not confident enough

10. **Why is a smaller model useful for local testing?**
   A smaller model is useful for local testing because it runs faster and requires less memory and computing power

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
