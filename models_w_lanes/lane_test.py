import cv2
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf


def lane_detection_with_hough(img):
    height, width = img.shape[:2]

    # ROI mask
    roi_mask = np.zeros((height, width), dtype=np.uint8)
    polygon = np.array([[
        (0, height),
        (width, height),
        (int(width * 0.7), int(height * 0.5)),
        (int(width * 0.5), int(height * 0.5))
    ]], np.int32)
    cv2.fillPoly(roi_mask, polygon, 255)

    # yellow detection
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    L_lab, A_lab, B_lab = cv2.split(lab)
    B_blur = cv2.blur(B_lab, (10, 10))
    yellow_mask = (B_lab > B_blur + 15)  # brighter yellow than local mean

    # convert to HLS
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(hls)

    # local mean filtering
    kernel_size = (10, 10)
    h_blur = cv2.blur(h, kernel_size)
    l_blur = cv2.blur(l, kernel_size)
    s_blur = cv2.blur(s, kernel_size)

    white_mask = (
        (l > l_blur + 5) &
        (s < s_blur)
    )

    color_mask = np.zeros_like(h, dtype=np.uint8)
    color_mask[yellow_mask | white_mask] = 255
    color_mask = cv2.bitwise_and(color_mask, roi_mask)

    # mask and canny
    masked = cv2.bitwise_and(img, img, mask=color_mask)
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # hough line detection
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=20,
        maxLineGap=20
    )

    # draw on filtered lines and darken background
    overlay = (img * 0.5).astype(np.uint8)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-5)
            length = np.hypot(x2 - x1, y2 - y1)

            # Filter out nearly horizontal lines and very short segments
            if 0.3 < abs(slope) < 2.0 and length > 40:
                cv2.line(overlay, (x1, y1), (x2, y2), (255, 255, 0), 3)  # bright cyan

    return overlay


def tf_preprocess_lane_features(image):
    gray = tf.image.rgb_to_grayscale(image)
    gray_blurred = tf.nn.avg_pool2d(gray[tf.newaxis, ...], ksize=5, strides=1, padding='SAME')[0]
    
    sobel_edges = tf.image.sobel_edges(gray_blurred)
    edge_magnitude = tf.reduce_sum(tf.abs(sobel_edges), axis=-1)  # shape [H, W]

    edge_norm = tf.clip_by_value(edge_magnitude / tf.reduce_max(edge_magnitude + 1e-6), 0, 1)
    edge_rgb = tf.stack([edge_norm] * 3, axis=-1)

    overlay = image * 0.3 + edge_rgb * 0.7

    return overlay

@tf.function
def preprocess_image_tf(path, angle):
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)  # [0,1] float32

    image = image[-256:, :, :]  # Bottom crop
    image = tf.image.resize(image, [64, 128])

    processed = tf_preprocess_lane_features(image)

    return processed, angle


def main():
    # # Process a batch of images
    # num_images = 1
    # fig, axes = plt.subplots(num_images, 2, figsize=(12, 3 * num_images))

    # for i in range(1, num_images + 1):
    #     image_path = f'test_image_{i}.jpg'
    #     img = cv2.imread(image_path)

    #     if img is None:
    #         print(f"Failed to load image: {image_path}")
    #         continue

    #     result = lane_detection_with_hough(img)

    #     # Original image
    #     axes[i - 1, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #     axes[i - 1, 0].set_title(f'Original Image {i}')
    #     axes[i - 1, 0].axis('off')

    #     # Processed output
    #     axes[i - 1, 1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    #     axes[i - 1, 1].set_title(f'Lane Detection Overlay {i}')
    #     axes[i - 1, 1].axis('off')

    # plt.tight_layout()
    # plt.show()
    import cv2

def main():
    image_path = 'test_image_1.jpg'
    img = cv2.imread(image_path)

    if img is None:
        print(f"Failed to load image: {image_path}")
        return

    result = lane_detection_with_hough(img)

    # Resize if needed to ensure same dimensions (just in case)
    if img.shape != result.shape:
        result = cv2.resize(result, (img.shape[1], img.shape[0]))

    concatenated = cv2.vconcat([img, result])
    concatenated_rgb = cv2.cvtColor(concatenated, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 6))
    plt.imshow(concatenated_rgb)
    plt.title("Original vs Lane Detection")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()