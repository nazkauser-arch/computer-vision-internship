import cv2

def convert_to_grayscale(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Couldn't load image")

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image

# thresholding without blur
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

def apply_blur(image_path):
    gaussian_blur = cv2.GaussianBlur(
        grayscale_image,
        (9, 9),
        0
    )

    median_blur = cv2.medianBlur(
        grayscale_image,
        9
    )
    return gaussian_blur, median_blur

def read_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Couldn't find the image")

    return image

def find_contours(thresholded_image):
    contours, _ = cv2.findContours(
        thresholded_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    return contours

def draw_all_contours(best_threshold, contours):
    result = best_threshold.copy()

    cv2.drawContours(
        result,
        contours,
        -1,
        (0, 255, 0),
        2
    )
    return result

def find_smallest_contour(contours, min_area):
    return[
        contour for contour in contours
        if cv2.contourArea(contour) > min_area
    ]

def draw_filtered_contour(best_threshold, filetered_contours):
    filtered_image = best_threshold.copy()

    cv2.drawContours(
        filtered_image,
        filtered_contours,
        -1,
        255,
        2
    )
    return filtered_image


grayscale_image = convert_to_grayscale("images/input.jpg")
binary, inverse, adaptive_binary, adaptive_inverse = apply_threshold(grayscale_image)
gaussian_blur, median_blur = apply_blur(grayscale_image)

# saving grayscale image
cv2.imwrite("output/day_02/grayscale.jpg", grayscale_image)

# saving binary threshold
cv2.imwrite("output/day_02/binary.jpg", binary)
# saving inverse binary threshold
cv2.imwrite("output/day_02/inverse.jpg", inverse)
# saving adaptive threshold with binary thresholding
cv2.imwrite("output/day_02/adaptive_binary.jpg", adaptive_binary)
# saving adaptive threshold with inverse binary thresholding
cv2.imwrite("output/day_02/adaptive_inverse.jpg", adaptive_inverse)

# saving image with gaussian blur
cv2.imwrite("output/day_02/gaussian_blurred.jpg", gaussian_blur)
# saving image with median blur
cv2.imwrite("output/day_02/median_blurred.jpg", median_blur)

# number of contours found
best_threshold = binary
contours = find_contours(best_threshold)
print("Total contours found:", len(contours))

# saving all contours
all_contours = draw_all_contours(best_threshold, contours)
cv2.imwrite("output/day_02/all_contours.jpg", all_contours)

# no of filtered contours
filtered_contours = find_smallest_contour(contours, 100)
print("Filtered contours: ", len(filtered_contours))

# saving filtered contours
filtered_image = draw_filtered_contour(best_threshold, filtered_contours)
cv2.imwrite("output/day_02/filtered_contours.jpg", filtered_image)
