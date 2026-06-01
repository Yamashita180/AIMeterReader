import cv2
import numpy as np
import math
from pathlib import Path
from ultralytics import YOLO

YOLO_MODEL = "yolo26x.pt"
IMAGE_PATH = "InputImages/FixedCamera.jpg"
OUTPUT_PATH = "OutputImages/result.jpg"

MIN_VALUE = 0.0
MAX_VALUE = 2.0
CLOCK_CLASS_NAME = "clock"

clicked_points = []


def calc_angle(center, point):
    cx, cy = center
    px, py = point
    dx = px - cx
    dy = cy - py
    return math.degrees(math.atan2(dy, dx))


def angle_to_value(angle_deg, min_angle, max_angle):
    angle = angle_deg % 360
    min_a = min_angle % 360
    max_a = max_angle % 360

    total = (min_a - max_a) % 360
    pos = (min_a - angle) % 360

    ratio = pos / total
    ratio = max(0.0, min(1.0, ratio))

    return MIN_VALUE + ratio * (MAX_VALUE - MIN_VALUE)


def mouse_callback(event, x, y, flags, param):
    global clicked_points

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"clicked: {x}, {y}")


def calibrate_angles(meter_img, center):
    """
    画像上で0MPa方向、2MPa方向をクリックして角度を設定する。
    """
    global clicked_points
    clicked_points = []

    display = meter_img.copy()

    cv2.circle(display, center, 5, (255, 0, 0), -1)
    cv2.putText(
        display,
        "Click 0MPa point, then 2MPa point. Press ESC to cancel.",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )

    cv2.namedWindow("calibration")
    cv2.setMouseCallback("calibration", mouse_callback)

    while True:
        temp = display.copy()

        if len(clicked_points) >= 1:
            cv2.circle(temp, clicked_points[0], 5, (0, 255, 0), -1)
            cv2.line(temp, center, clicked_points[0], (0, 255, 0), 2)
            cv2.putText(
                temp,
                "0MPa",
                clicked_points[0],
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        if len(clicked_points) >= 2:
            cv2.circle(temp, clicked_points[1], 5, (0, 0, 255), -1)
            cv2.line(temp, center, clicked_points[1], (0, 0, 255), 2)
            cv2.putText(
                temp,
                "2MPa",
                clicked_points[1],
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        cv2.imshow("calibration", temp)

        key = cv2.waitKey(20) & 0xFF

        if key == 27:
            cv2.destroyWindow("calibration")
            raise RuntimeError("Calibration canceled")

        if len(clicked_points) >= 2:
            break

    cv2.destroyWindow("calibration")

    min_angle = calc_angle(center, clicked_points[0])
    max_angle = calc_angle(center, clicked_points[1])

    print(f"MIN_ANGLE(0MPa) = {min_angle:.1f} deg")
    print(f"MAX_ANGLE(2MPa) = {max_angle:.1f} deg")

    return min_angle, max_angle


def detect_center_circle(meter_img):
    h, w = meter_img.shape[:2]

    hsv = cv2.cvtColor(meter_img, cv2.COLOR_BGR2HSV)

    lower_gold = np.array([10, 40, 80])
    upper_gold = np.array([45, 220, 255])
    mask = cv2.inRange(hsv, lower_gold, upper_gold)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_center = (w // 2, h // 2)
    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue

        (x, y), r = cv2.minEnclosingCircle(cnt)

        if r < min(w, h) * 0.02 or r > min(w, h) * 0.15:
            continue

        dist = math.hypot(x - image_center[0], y - image_center[1])
        candidates.append((dist, int(x), int(y), int(r)))

    if candidates:
        _, cx, cy, r = min(candidates, key=lambda v: v[0])
        return (cx, cy), r, mask

    return (w // 2, h // 2), int(min(w, h) * 0.06), mask


def create_needle_mask(meter_img, center, center_radius):
    h, w = meter_img.shape[:2]

    hsv = cv2.cvtColor(meter_img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(meter_img, cv2.COLOR_BGR2GRAY)

    lower_blue = np.array([80, 10, 30])
    upper_blue = np.array([145, 255, 230])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    mask_dark = cv2.inRange(gray, 20, 170)

    mask = cv2.bitwise_or(mask_blue, mask_dark)

    valid = np.zeros((h, w), dtype=np.uint8)
    outer_radius = int(min(w, h) * 0.43)
    cv2.circle(valid, center, outer_radius, 255, -1)
    cv2.circle(valid, center, int(center_radius * 0.9), 0, -1)

    mask = cv2.bitwise_and(mask, valid)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask


def detect_needle_tip(meter_img, center, center_radius):
    h, w = meter_img.shape[:2]
    cx, cy = center

    mask = create_needle_mask(meter_img, center, center_radius)

    max_r = int(min(w, h) * 0.43)
    start_r = int(center_radius * 1.1)
    end_r = max_r

    best_score = -1
    best_tip = None
    best_angle = None

    for angle_deg in np.arange(-180, 180, 0.5):
        rad = math.radians(angle_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        hit_count = 0
        max_continuous = 0
        continuous = 0
        last_hit_r = None
        gap = 0
        max_gap = 8

        for r in range(start_r, end_r):
            x = int(cx + r * cos_a)
            y = int(cy - r * sin_a)

            if x < 0 or x >= w or y < 0 or y >= h:
                break

            if mask[y, x] > 0:
                hit_count += 1
                continuous += 1
                max_continuous = max(max_continuous, continuous)
                last_hit_r = r
                gap = 0
            else:
                gap += 1
                if gap > max_gap:
                    continuous = 0

        if last_hit_r is None:
            continue

        length = last_hit_r - start_r

        if length < center_radius * 2.0:
            continue

        score = max_continuous * 2.0 + length * 0.5 + hit_count * 0.2

        if score > best_score:
            tip_x = int(cx + last_hit_r * cos_a)
            tip_y = int(cy - last_hit_r * sin_a)
            best_score = score
            best_tip = (tip_x, tip_y)
            best_angle = angle_deg

    return best_tip, mask, best_angle


def main():
    Path("OutputImages").mkdir(parents=True, exist_ok=True)

    model = YOLO(YOLO_MODEL)
    img = cv2.imread(IMAGE_PATH)

    if img is None:
        raise FileNotFoundError(f"画像を読み込めません: {IMAGE_PATH}")

    results = model(img)

    detected_any = False

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            name = result.names[cls]

            if name != CLOCK_CLASS_NAME:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            meter = img[y1:y2, x1:x2].copy()

            center, center_radius, center_mask = detect_center_circle(meter)
            min_angle, max_angle = calibrate_angles(meter, center)

            tip, needle_mask, detected_angle = detect_needle_tip(
                meter,
                center,
                center_radius,
            )

            if tip is None:
                print("針の先端を検出できませんでした")
                continue

            angle = calc_angle(center, tip)
            value = angle_to_value(angle, min_angle, max_angle)

            print(f"confidence={conf:.2f}")
            print(f"center={center}")
            print(f"center_radius={center_radius}")
            print(f"min_angle={min_angle:.1f}")
            print(f"max_angle={max_angle:.1f}")
            print(f"tip={tip}")
            print(f"angle={angle:.1f} deg")
            print(f"value={value:.3f} MPa")

            cv2.circle(meter, center, center_radius, (255, 0, 0), 2)
            cv2.circle(meter, center, 5, (255, 0, 0), -1)

            # 0MPa線
            r = int(min(meter.shape[:2]) * 0.42)
            p0 = (
                int(center[0] + r * math.cos(math.radians(min_angle))),
                int(center[1] - r * math.sin(math.radians(min_angle))),
            )
            cv2.line(meter, center, p0, (0, 255, 0), 1)

            # 2MPa線
            p2 = (
                int(center[0] + r * math.cos(math.radians(max_angle))),
                int(center[1] - r * math.sin(math.radians(max_angle))),
            )
            cv2.line(meter, center, p2, (0, 0, 255), 1)

            # 針
            cv2.circle(meter, tip, 5, (0, 255, 255), -1)
            cv2.line(meter, center, tip, (255, 0, 255), 2)

            cv2.putText(
                meter,
                f"{value:.3f} MPa",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            img[y1:y2, x1:x2] = meter

            cv2.imshow("meter", meter)
            cv2.imshow("center_mask", center_mask)
            cv2.imshow("needle_mask", needle_mask)

            detected_any = True

    if not detected_any:
        print("メーターを検出できませんでした")

    cv2.imwrite(OUTPUT_PATH, img)
    print(f"saved: {OUTPUT_PATH}")

    cv2.imshow("result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()