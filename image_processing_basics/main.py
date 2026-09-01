import cv2
import matplotlib.pyplot as plt

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    else:
        return image

image = load_image("images/input.jpg")

def read_image(image_path):
    image = load_image("images/input.jpg")
    if image is None:
        raise FileNotFoundError(f"Could not load image")
    
    return image

def resize_image(image_path):
    image = load_image("images/input.jpg")
    resized_image = cv2.resize(image, (225, 225))

    return resized_image

def convert_to_grayscale(image_path):
    image = load_image("images/input.jpg")
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return grayscale_image

def detect_edges(image_path, lower_threshold, upper_threshold):
    image = load_image("images/input.jpg")
    edge_detected_image = cv2.Canny(image, lower_threshold, upper_threshold)

    return edge_detected_image




# print its width, height and number of channels
print(f"Width: {image.shape[0]}") # width
print(f"Height: {image.shape[1]}") # height
print(f"No of channels: {image.shape[2]}") # channels

# print its datatype
print(f"Datatype: {image.dtype}")

# max and min pixel values
print(f"Max pixel value: {image.max()}")
print(f"Min pixel value: {image.min()}")

# saving resized image
resized_image = resize_image("images/input.jpg")
cv2.imwrite("output/day_01/resized.jpg", resized_image)

# saving grayscale image
grayscale_image = convert_to_grayscale("images/input.jpg")
cv2.imwrite("output/day_01/grayscale.jpg", grayscale_image)

# saving edge detected image
edge_image_01 = detect_edges("images/input.jpg", 100, 200)
cv2.imwrite("output/day_01/edges_01.jpg", edge_image_01)
edge_image_02 = detect_edges("images/input.jpg", 150, 250)
cv2.imwrite("output/day_01/edges_02.jpg", edge_image_02)

# displaying images
def display_results():
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    plt.subplot(2, 2, 1)
    plt.imshow(rgb_image)
    plt.axis("off")
    plt.title("Original Image")

    plt.subplot(2, 2, 2)
    plt.imshow(resized_rgb_image)
    plt.axis("off")
    plt.title("Resized Image")

    plt.subplot(2, 2, 3)
    plt.imshow(grayscale_image, cmap="gray")
    plt.axis("off")
    plt.title("Grayscale Image")

    plt.subplot(2, 2, 4)
    plt.imshow(edge_image_01, cmap="gray")
    plt.axis("off")
    plt.title("Edged Image")

    plt.show()

print(display_results())