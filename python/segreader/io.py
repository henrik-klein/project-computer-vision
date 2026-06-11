import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def save_image(path: str, image: np.ndarray) -> None:
    cv2.imwrite(path, image)


def draw_annotations(
    image: np.ndarray,
    digit_boxes: list,
    decoded_digits: list,
    number_string: str,
) -> np.ndarray:
    out = image.copy()
    for (y1, x1, y2, x2), digit in zip(digit_boxes, decoded_digits):
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_y = max(y1 - 6, 14)
        cv2.putText(out, digit, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(out, number_string, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return out
