import cv2

def convert_to_grayscale(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Couldn't load image")

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image

def apply_threshold(grayscale_image):
    _, binary= cv2.threshold(
        grayscale_image,
        127,
        255,
        cv2.THRESH_BINARY
    )

    _, inverse = cv2.threshold(
        grayscale_image,
        127,
        255,
        cv2.THRESH_BINARY_INV
    )

    adaptive_binary = cv2.adaptiveThreshold(
        grayscale_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        199,
        2
    )

    adaptive_inverse = cv2.adaptiveThreshold(
        grayscale_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        199,
        2
    )
    return binary, inverse, adaptive_binary, adaptive_inverse

grayscale_image = convert_to_grayscale("images/input.jpg")
binary, inverse, adaptive_binary, adaptive_inverse = apply_threshold(grayscale_image)

# saving binary threshold
cv2.imwrite("output/day_02/binary.jpg", binary)
# saving inverse binary threshold
cv2.imwrite("output/day_02/inverse.jpg", inverse)
# saving adaptive threshold with binary thresholding
cv2.imwrite("output/day_02/adaptive_binary.jpg", adaptive_binary)
# saving adaptive threshold with inverse binary thresholding
cv2.imwrite("output/day_02/adaptive_inverse.jpg", adaptive_inverse)

