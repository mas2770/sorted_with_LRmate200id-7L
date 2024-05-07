import cv2
import pyrealsense2
import numpy as np
from realsense_depth import *

dc = DepthCamera()

# cord_centre_cube = {"Blue": [[0,0],[],[],[],0,[],0,[],[],0,0],
#     "Red": [[0,0],[],[],[],0,[],0,[],[],0,0],
#     "Yellow": [[0,0],[],[],[],0,[],0,[],[],0,0],
#     "Orange": [[0,0],[],[],[],0,[],0,[],[],0,0],
#     "Purple": [[0,0],[],[],[],0,[],0,[],[],0,0]
# }
colors = {
    "Blue": (255, 144, 30),
    "Red": (0, 0, 255),
    "Yellow": (0, 255, 255),
    "Orange": (0, 165, 255),
    "Purple": (204, 50, 153)
}
hsv_upper_and_lower = {
    "Blue": [[74, 100, 100], [130, 255, 255]],
    "Red": [[0, 120, 70], [10, 255, 255]],
    "Yellow": [[26, 50, 50], [35, 255, 255]],
    "Orange": [[11, 50, 50], [30, 255, 255]],
    "Purple": [[120, 50, 50], [170, 255, 255]]
}

while(1):

    contours = []
    ret, depth_frame, color_frame = dc.get_frame()
    img = color_frame
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = cv2.blur(hsv, (5, 5))
    
    for key in colors:

        if key != "Blue":
            continue
        lower = np.array(hsv_upper_and_lower[key][0])
        upper = np.array(hsv_upper_and_lower[key][1])
        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours!=[] and contours is not None:
            largest_contour = None
            max_area = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
                    largest_contour = contour
            
            if cv2.contourArea(largest_contour) > 2500:
                rect = cv2.minAreaRect(largest_contour)
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                x, y, w, h = cv2.boundingRect(largest_contour)
                center_x = x + w // 2
                center_y = y + h // 2
                # cord_centre_cube[key][1].append(center_x)
                # cord_centre_cube[key][2].append(center_y)
                # cord_centre_cube[key][5].append(-rect[2])
                # cord_centre_cube[key][7].append(w)
                print("w =" ,w,"h =", h)
                # cord_centre_cube[key][8].append(h)
                cv2.circle(img,(center_x,center_y),5,(0,255,0),-1)
                # try:
                #     depth_centre = depth_frame[640,360]
                #     depth_edge = depth_frame[640,260]
                #     distance_between = math.sqrt((int(pow(depth_edge,2))-int(pow(depth_centre,2))))
                #     transfer_coefficient = distance_between/100
                #     if transfer_coefficient == 0.0:
                #         continue
                #     cord_centre_cube[key][3].append(transfer_coefficient)
                #     # cord_centre_cube[key][4].append(depth_edge)
                # except:
                #     print('Невозможно определить глубину изображения!!!')
                cv2.drawContours(img, [box], -1, colors[key], 7)
                # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
                # cv2.setWindowProperty('img', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow('img',img)
                cv2.waitKey(1)
cv2.destroyAllWindows()