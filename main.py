import time
from pydub import AudioSegment
import simpleaudio as sa
import utils
import numpy as np
import cv2


def click(event, mouse_x, mouse_y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print(f"{mouse_x}, {mouse_y}")
        print()


utils.start_camera()
utils.use_filter()
cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Output', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback('Output', click)

song = AudioSegment.from_mp3("arrow_sound.mp3")
wave_object = sa.WaveObject.from_wave_file('arrow_sound.wav')

x_limits = (110, 615)
y_limits = (235, 460)

# out_depth = cv2.VideoWriter('out_depth_2.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (640, 480))
# out_rgb = cv2.VideoWriter('out_frame_2.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (640, 480))

ball_first_point = None
ball_something_weird_timeout = 0
ball_timeout = 0
ball_seconds_cooldown = 1

animation_points_message = []

stored_hits = []
angles = []
canvas_h = 1080
canvas_w = 1920

target = cv2.imread('target.jpg')
scale = (canvas_h / 1.) / target.shape[0]
target = cv2.resize(target, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

while True:
    ret, frame, depth = utils.get_frames()
    if not ret:
        continue
    canvas = np.zeros((canvas_h, canvas_w, 3), np.uint8)
    canvas[-target.shape[0]:, canvas.shape[1] // 2 - target.shape[1] // 2: canvas.shape[1] // 2 + target.shape[1] // 2] = target

    depth = 255 - cv2.flip(depth, 0)
    depth_roi = depth.copy()[y_limits[0]: y_limits[1], x_limits[0]: x_limits[1]]
    _, mask = cv2.threshold(depth_roi, 210, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(filter(lambda l: cv2.contourArea(l) > 500, contours))
    if len(contours):
        contour_mask = np.zeros_like(mask)
        contour = max(contours, key=lambda l: cv2.contourArea(l))
        (x, y), r = cv2.minEnclosingCircle(contour)
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        d = float(depth_roi[int(cY), int(cX)])
        cv2.circle(frame, (int(x) + x_limits[0], int(y)), int(r), (0, 255, 0), 2)
        if time.time() - ball_timeout > ball_seconds_cooldown:
            if ball_first_point is None or time.time() - ball_something_weird_timeout > 0.5:
                ball_first_point = (x, y, d)
                ball_something_weird_timeout = time.time()
            else:
                ball_second_point = (x, y, d)
                d_end = ball_second_point[2]
                try:
                    slope = (ball_second_point[1] - ball_first_point[1]) / (ball_second_point[0] - ball_first_point[0])
                    x_end = (-ball_first_point[1] + slope * ball_first_point[0]) / slope
                except ZeroDivisionError:
                    continue
                cv2.circle(frame, (x_limits[0] + int(x_end), y_limits[0]), 5, (0, 0, 255), -1)

                d_mapped = 56.54665 * d_end - 12765.29  # 49.58504 * d_end - 10904.68  # 29.58504 * d_end - 6327
                x_mapped = 271.7306 * x_end ** 0.2478501
                if np.iscomplex(x_mapped):
                    x_mapped = np.absolute(x_mapped)
                # print(f"({x_mapped}, {d_mapped})")
                stored_hits.append((int(x_mapped), int(d_mapped)))
                angles.append(np.random.uniform(np.radians(180), np.radians(270)))
                # print(int(x_end), int(d_end))
                # print(euclidean([x_mapped, d_mapped], [canvas_w / 2, canvas_h / 2]))

                animation_points_message.append(utils.PointsMessage([int(x_mapped), int(d_mapped)], (canvas_w / 2, canvas_h / 2)))
                # wave_object.play()

                ball_first_point = None
                ball_timeout = time.time()

    for pt, ang in zip(stored_hits, angles):
        cv2.circle(canvas, pt, 7, (0, 0, 0), -1)
        pt1 = (pt[0] + int(200 * np.cos(ang)), pt[1] + int(200 * np.sin(ang)))
        cv2.arrowedLine(canvas, pt1, pt, (100, 100, 100), 7, cv2.LINE_AA)

    for anim in animation_points_message:
        anim.display(canvas)
        anim.update()
        if anim.is_dead():
            animation_points_message.remove(anim)

    cv2.imshow('Output', canvas)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# out_rgb.release()
# out_depth.release()
cv2.destroyAllWindows()
utils.release()
