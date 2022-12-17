import numpy as np
import cv2 

BLUE = ([104, 5, 5], [108, 255, 255])
RED = ([110, 5, 5], [179, 255, 255])


def add_to_mask(hsv, mask, color):
    lower = np.array(color[0], dtype="uint8")
    upper = np.array(color[1], dtype="uint8")
    mask += cv2.inRange(hsv, lower, upper)


cap = cv2.VideoCapture("./balls.mp4")
FPS = cap.get(cv2.CAP_PROP_FPS)
prev = []
appr_r = 0.0
speeds = {}


def write_speeds(centers, prev):
    _speeds = {}
    if len(centers) > len(prev):
        _speeds[tuple(centers[-1])] = 0
        prev.append(centers[-1])
    for point in centers:   
        min = np.linalg.norm(point - prev[0])
        for prev_point in prev:
            dist = np.linalg.norm(point - prev_point)
            if dist < min:
                min = dist
        if min < 2:
            min = 0
        _speeds[tuple(point)] = min
    return _speeds
        


def display_speeds(_speeds):
    for key, value in _speeds.items():
        cv2.putText(frame, f"Speed = {value*3} px/sec", key,
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255))


while True:
    flag, frame = cap.read()
    if flag:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype = "uint8")
        add_to_mask(hsv, mask, BLUE)
        add_to_mask(hsv, mask, RED)
        mask = cv2.erode(mask, None, iterations=20)
        mask = cv2.dilate(mask, None, iterations=10)

        contours,hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cnt = 0   
        centers = []

        for c in contours:
            (x,y),radius = cv2.minEnclosingCircle(c)
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == 2:
                appr_r = radius
                print(appr_r)
            center = (int(x),int(y))
            radius = int(radius)
            if radius > appr_r - 20 and radius < appr_r + 20:
                cnt += 1
                cv2.circle(frame,center,radius,(0,255,0),2)
                centers.append(np.array(center))

        if cap.get(cv2.CAP_PROP_POS_FRAMES) % (FPS // 3) == 1:
            speeds = write_speeds(centers, prev)
            prev = centers
        display_speeds(speeds)     

        cv2.putText(frame, f"BALLS: {cnt}",(10,60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255))  
        cv2.imshow('video', frame)
        pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
        cv2.waitKey(1000)

    if cv2.waitKey(10) == 27:
        break
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        break

