import cv2
import numpy as np
from scipy.interpolate import splprep, splev

# List to store selected points
points = []

pointColor = (255, 255, 255)
lineColor = (255, 0, 255)

# Mouse callback function
def select_points(event, x, y, flags, param):
    global points, image
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(image, (x, y), 5, pointColor, -1)
        cv2.imshow("Path from Points", image)

# Create a blank canvas
image = np.zeros((500, 500, 3), dtype=np.uint8)
cv2.imshow("Path from Points", image)
cv2.setMouseCallback("Path from Points", select_points)

# Main loop
def points_loop():
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('d'):  # Draw path
            break
        if key == ord('u'):  # Undo
            if len(points) > 0:
                points.pop()
                image = np.zeros((500, 500, 3), dtype=np.uint8)
                for point in points:
                    cv2.circle(image, point, 5, pointColor, -1)
                    
                cv2.imshow("Path from Points", image)

# Final drawing
def draw_path():
    if len(points) > 3:
        points = np.array(points)
        tck, u = splprep([points[:, 0], points[:, 1]], s=0)
        u_fine = np.linspace(0, 1, 500)
        x_new, y_new = splev(u_fine, tck)
        for i in range(len(x_new) - 1):
            pt1 = (int(x_new[i]), int(y_new[i]))
            pt2 = (int(x_new[i + 1]), int(y_new[i + 1]))
            cv2.line(image, pt1, pt2, lineColor, 2)
    else:
        print("Need at least 4 points to draw a path")
        points_loop()

points_loop()

draw_path()

cv2.imshow("Path from Points", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
