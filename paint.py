import cv2
import numpy as np
import mediapipe as mp
from collections import deque


bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]


smooth_x = deque(maxlen=15)  
smooth_y = deque(maxlen=15)

alpha = 0.3
ema_x = None
ema_y = None

blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0


kernel = np.ones((5,5),np.uint8)

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0

def draw_square_with_white_border(image, center, size, color, border_thickness):
    x, y = center
    half_size = size // 2

    
    cv2.rectangle(image, (x - half_size, y - half_size), (x + half_size, y + half_size), color, -1)

    border_color = (255, 255, 255)
    cv2.rectangle(image, (x - half_size - border_thickness, y - half_size - border_thickness),
                  (x + half_size + border_thickness, y + half_size + border_thickness), border_color, border_thickness)


mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils



cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

ret = True
while ret:
    ret, frame = cap.read()
    
    if not ret or frame is None:
        print("Error: Could not read frame")
        break

    x, y, c = frame.shape

    frame = cv2.flip(frame, 1)
 
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    paintWindow = np.zeros((x, y, 3)) + 0
    paintWindow = cv2.rectangle(paintWindow, (30, 10), (110, 90), (255, 255, 255), -1)
    paintWindow = cv2.rectangle(paintWindow, (160, 10), (240, 90), (255, 0, 0), -1)
    paintWindow = cv2.rectangle(paintWindow, (290, 10), (370, 90), (0, 255, 0), -1)
    paintWindow = cv2.rectangle(paintWindow, (420, 10), (500, 90), (0, 0, 255), -1)
    paintWindow = cv2.rectangle(paintWindow, (545, 10), (625, 90), (0, 255, 255), -1)

    draw_square_with_white_border(paintWindow, (70, 50), 80, (255, 255, 255), 3)
    draw_square_with_white_border(paintWindow, (200, 50), 80, (255, 0, 0), 3)
    draw_square_with_white_border(paintWindow, (330, 50), 80, (0, 255, 0), 3)
    draw_square_with_white_border(paintWindow, (460, 50), 80, (0, 0, 255), 3)
    draw_square_with_white_border(paintWindow, (585, 50), 80, (0, 255, 255), 3)

    cv2.putText(paintWindow, "CLEAR", (45, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "BLUE", (175, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "GREEN", (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "RED", (440, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "YELLOW", (555, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

    frame = cv2.rectangle(frame, (30, 10), (110, 90), (255, 255, 255), -1)
    frame = cv2.rectangle(frame, (160, 10), (240, 90), (255, 0, 0), -1)
    frame = cv2.rectangle(frame, (290, 10), (370 , 90), (0, 255, 0), -1)
    frame = cv2.rectangle(frame, (420 , 10 ), (500, 90), (0, 0, 255), -1)
    frame = cv2.rectangle(frame, (545, 10), (625 , 90), (0, 255, 255), -1)

    draw_square_with_white_border(frame, (70, 50), 80, (255, 255, 255), 3)
    draw_square_with_white_border(frame, (200, 50), 80, (255, 0, 0), 3)
    draw_square_with_white_border(frame, (330, 50), 80, (0, 255, 0), 3)
    draw_square_with_white_border(frame, (460, 50), 80, (0, 0, 255), 3)
    draw_square_with_white_border(frame, (585, 50), 80, (0, 255, 255), 3)

    cv2.putText(frame, "CLEAR", (45, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (175, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (440, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "YELLOW", (555, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
 


    result = hands.process(framergb)


    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
             
                lmx = int(lm.x * y)
                lmy = int(lm.y * x)

                landmarks.append([lmx, lmy])


          
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
        
        # Get forefinger coordinates
        raw_x, raw_y = landmarks[8][0], landmarks[8][1]
        
        # Add to smoothing buffers
        smooth_x.append(raw_x)
        smooth_y.append(raw_y)
        
        # Calculate moving average
        ma_x = int(sum(smooth_x) / len(smooth_x))
        ma_y = int(sum(smooth_y) / len(smooth_y))
        
        # Calculate exponential moving average
        if ema_x is None:
            ema_x = raw_x
            ema_y = raw_y
        else:
            ema_x = alpha * raw_x + (1 - alpha) * ema_x
            ema_y = alpha * raw_y + (1 - alpha) * ema_y
        
        # Combine both smoothing techniques
        smoothed_x = int((ma_x + ema_x) / 2)
        smoothed_y = int((ma_y + ema_y) / 2)
        
        fore_finger = (smoothed_x, smoothed_y)
        center = fore_finger
        thumb = (landmarks[4][0], landmarks[4][1])
        cv2.circle(frame, center, 3, (0,255,0),-1)
        print(center[1]-thumb[1])
        if (thumb[1]-center[1]<30):
            if len(bpoints) > 0:
                bpoints.append(deque(maxlen=512))
                blue_index += 1
            if len(gpoints) > 0:
                gpoints.append(deque(maxlen=512))
                green_index += 1
            if len(rpoints) > 0:
                rpoints.append(deque(maxlen=512))
                red_index += 1
            if len(ypoints) > 0:
                ypoints.append(deque(maxlen=512))
                yellow_index += 1

        elif center[1] <= 90:
            if 30 <= center[0] <= 110:
                bpoints = [deque(maxlen=512)]
                gpoints = [deque(maxlen=512)]
                rpoints = [deque(maxlen=512)]
                ypoints = [deque(maxlen=512)]

                blue_index = 0
                green_index = 0
                red_index = 0
                yellow_index = 0

                paintWindow[100:,:,:] = 0
            elif 160<= center[0] <= 240:
                    colorIndex = 0 # Blue
            elif 290 <= center[0] <= 370:
                    colorIndex = 1 # Green
            elif 420<= center[0] <= 500:
                    colorIndex = 2 # Red
            elif 545 <= center[0] <= 665:
                    colorIndex = 3 # Yellow
        else :
            if colorIndex == 0 and blue_index >= 0:
                bpoints[blue_index].appendleft(center)
            elif colorIndex == 1 and green_index >= 0:
                gpoints[green_index].appendleft(center)
            elif colorIndex == 2 and red_index >= 0:
                rpoints[red_index].appendleft(center)
            elif colorIndex == 3 and yellow_index >= 0:
                ypoints[yellow_index].appendleft(center)

    else:
        if len(bpoints) > 0:
            bpoints.append(deque(maxlen=512))
            blue_index += 1
        if len(gpoints) > 0:
            gpoints.append(deque(maxlen=512))
            green_index += 1
        if len(rpoints) > 0:
            rpoints.append(deque(maxlen=512))
            red_index += 1
        if len(ypoints) > 0:
            ypoints.append(deque(maxlen=512))
            yellow_index += 1

    points = [bpoints, gpoints, rpoints, ypoints]
 
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

    cv2.imshow("Output", frame)
    cv2.imshow("Paint", paintWindow)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
