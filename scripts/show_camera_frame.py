"""Read a single frame from the USB camera and display it."""

import cv2


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open USB camera")

    try:
        # Discard a few frames to let auto-exposure/focus settle.
        for _ in range(10):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Could not read frame from camera")

        cv2.imshow("Camera Frame", frame)
        cv2.waitKey(0)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
