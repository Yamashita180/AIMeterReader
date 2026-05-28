import cv2
import numpy as np
from ultralytics import YOLO
import math

YOLO_MODEL = "best.pt"
IMAGE_PATH = "test.jpg"

# メーター仕様
MIN_VALUE = 0
MAX_VALUE = 1.0

# 目盛りの角度設定
# 例：左下が0、右下が1.0の圧力計
MIN_ANGLE = 225
MAX_ANGLE = -45


def angle_to_value(angle_deg):
    # 角度を 0〜360 に正規化
    angle = angle_deg % 360
    min_a = MIN_ANGLE % 360
    max_a = MAX_ANGLE % 360

    if max_a < min_a:
        max_a += 360
    if angle < min_a:
        angle += 360

    ratio = (angle - min_a) / (max_a - min_a)
    ratio = max(0.0, min(1.0, ratio))

    return MIN_VALUE + ratio * (MAX_VALUE - MIN_VALUE)


def detect_needle_angle(meter_img):
    gray = cv2.cvtColor(meter_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    h, w = gray.shape
    center = (w // 2, h // 2)

    edges = cv2.Canny(blur, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=int(min(w, h) * 0.25),
        maxLineGap=10
    )

    if lines is None:
        return None

    best_line = None
    best_score = 0

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # 中心に近い線を針候補にする
        d1 = math.hypot(x1 - center[0], y1 - center[1])
        d2 = math.hypot(x2 - center[0], y2 - center[1])
        length = math.hypot(x2 - x1, y2 - y1)

        center_distance = min(d1, d2)
        score = length - center_distance * 0.5

        if score > best_score:
            best_score = score
            best_line = (x1, y1, x2, y2)

    if best_line is None:
        return None

    x1, y1, x2, y2 = best_line

    # 中心から遠い点を針先とみなす
    d1 = math.hypot(x1 - center[0], y1 - center[1])
    d2 = math.hypot(x2 - center[0], y2 - center[1])

    tip = (x1, y1) if d1 > d2 else (x2, y2)

    dx = tip[0] - center[0]
    dy = center[1] - tip[1]  # 画像座標はYが下向きなので反転

    angle = math.degrees(math.atan2(dy, dx))
    return angle, best_line, center, tip


def main():
    model = YOLO(YOLO_MODEL)
    img = cv2.imread(IMAGE_PATH)

    results = model(img)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            meter = img[y1:y2, x1:x2]
            detected = detect_needle_angle(meter)

            if detected is None:
                print("針を検出できませんでした")
                continue

            angle, line, center, tip = detected
            value = angle_to_value(angle)

            print(f"confidence={conf:.2f}")
            print(f"angle={angle:.1f} deg")
            print(f"value={value:.3f}")

            # 可視化
            lx1, ly1, lx2, ly2 = line
            cv2.line(meter, (lx1, ly1), (lx2, ly2), (0, 0, 255), 2)
            cv2.circle(meter, center, 5, (255, 0, 0), -1)
            cv2.circle(meter, tip, 5, (0, 255, 0), -1)

            cv2.imshow("meter", meter)
            cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()