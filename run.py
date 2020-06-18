import cv2
import datetime
from src.hand_tracker import HandTracker

print (datetime.datetime.now().timestamp())
WINDOW = "Hand Tracking"
PALM_MODEL_PATH = "models/palm_detection_without_custom_op.tflite"
LANDMARK_MODEL_PATH = "models/hand_landmark.tflite"
ANCHORS_PATH = "models/anchors.csv"

POINT_COLOR = (0, 255, 0)
CONNECTION_COLOR = (255, 0, 0)
THICKNESS = 2

DETECTOR_RADIUS = 200
DETECTOR_POSITION = (200, 300) 
DETECTOR_WAITTIME = 2
def runBot():
    pass

cv2.namedWindow(WINDOW)
capture = cv2.VideoCapture(0)

if capture.isOpened():
    hasFrame, frame = capture.read()
else:
    hasFrame = False

#        8   12  16  20
#        |   |   |   |
#        7   11  15  19
#    4   |   |   |   |
#    |   6   10  14  18
#    3   |   |   |   |
#    |   5---9---13--17
#    2    \         /
#     \    \       /
#      1    \     /
#       \    \   /
#        ------0-
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
]

detector = HandTracker(
    PALM_MODEL_PATH,
    LANDMARK_MODEL_PATH,
    ANCHORS_PATH,
    box_shift=0.2,
    box_enlarge=1.3
)

frameSkip = False
prevFrameDetected = False
while hasFrame:
    if frameSkip:
        frameSkip = False
        capture.read()
        continue
    frameSkip = True
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    points, _ = detector(image)
    detected = False
    if points is not None:
        for point in points:
            x, y = point
            cv2.circle(frame, (int(x), int(y)), THICKNESS * 2, POINT_COLOR, THICKNESS)
            if ((DETECTOR_POSITION[0] - x)*(DETECTOR_POSITION[0] - x) + (DETECTOR_POSITION[1] - y)*(DETECTOR_POSITION[1] - y)) < DETECTOR_RADIUS*DETECTOR_RADIUS:
                if prevFrameDetected == False:
                    startTime = datetime.datetime.now().timestamp()
                detected = True

        for connection in connections:
            x0, y0 = points[connection[0]]
            x1, y1 = points[connection[1]]
            cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), CONNECTION_COLOR, THICKNESS)

    prevFrameDetected = detected
    detectorColor = (255, 0, 0)
    if detected:
        detectorColor = (0, 255, 0)
        currentTime = datetime.datetime.now().timestamp()
        if currentTime - startTime >= DETECTOR_WAITTIME:
            detectorColor = (255, 255, 0)
            runBot()

    cv2.circle(frame, DETECTOR_POSITION, DETECTOR_RADIUS, detectorColor, THICKNESS)
    cv2.imshow(WINDOW, frame)
    hasFrame, frame = capture.read()
    key = cv2.waitKey(1)
    if key == 27:
        break


capture.release()
cv2.destroyAllWindows()
