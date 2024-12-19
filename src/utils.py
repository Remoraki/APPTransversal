import numpy as np
from scipy.interpolate import splprep, splev
import cv2

def set_background_with_texture(canvas, texture):
        canvas_h, canvas_w = canvas.shape[:2]
        texture_h, texture_w = texture.shape[:2]

        for y in range(0, canvas_h, texture_h):
            for x in range(0, canvas_w, texture_w):
                x_end = min(x + texture_w, canvas_w)
                y_end = min(y + texture_h, canvas_h)

                texture_roi = texture[: y_end - y, : x_end - x]

                canvas[y:y_end, x:x_end] = texture_roi

        return canvas
    
def calculate_spline(points):
    k = min(len(points) - 1, 5)
    points = np.array(points)
    tck, u = splprep([points[:, 0], points[:, 1]], s=0, k=k)
    u_fine = np.linspace(0, 1, 500)
    x_new, y_new = splev(u_fine, tck)
    return x_new, y_new

def calculate_lerp_points(pt1, pt2, num_points=100):
    x_new = np.linspace(pt1[0], pt2[0], num_points)
    y_new = np.linspace(pt1[1], pt2[1], num_points)
    return x_new, y_new

def calculate_spline_with_tangents(points):
    k = min(len(points) - 1, 5)
    points = np.array(points)
    tck, u = splprep([points[:, 0], points[:, 1]], s=0, k=k)
    u_fine = np.linspace(0, 1, 500)
    x_new, y_new = splev(u_fine, tck)
    
    dx, dy = splev(u_fine, tck, der=1)
    tangents = np.array([dx, dy]).T
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]  # Normalize tangents
    
    return x_new, y_new, tangents

def draw_line(image, points, color):
    pt1, pt2 = points
    cv2.line(image, pt1, pt2, color, 2)
    
def draw_line_with_width(image, points, color, width):
    half_width = width // 2
    pt1, pt2 = points
    cv2.line(image, pt1, pt2, color, 2)
    
    pt1_left = (pt1[0] - half_width, pt1[1])
    pt1_right = (pt1[0] + half_width, pt1[1])
    pt2_left = (pt2[0] - half_width, pt2[1])
    pt2_right = (pt2[0] + half_width, pt2[1])
    
    cv2.line(image, pt1_left, pt2_left, (255, 0, 0), 2)
    cv2.line(image, pt1_right, pt2_right, (255, 0, 0), 2)
    
def draw_spline(image, points, color):
        x_new, y_new = calculate_spline(points)
        for i in range(len(x_new) - 1):
            pt1 = (int(x_new[i]), int(y_new[i]))
            pt2 = (int(x_new[i + 1]), int(y_new[i + 1]))
            cv2.line(image, pt1, pt2, color, 2)

def draw_spline_with_width(image, points, color, width):
    half_width = width // 2
    x_new, y_new, tangents = calculate_spline_with_tangents(points)

    for i in range(len(x_new) - 1):
        pt1 = np.array([int(x_new[i]), int(y_new[i])])
        pt2 = np.array([int(x_new[i + 1]), int(y_new[i + 1])])

        tangent = tangents[i]

        normal = np.array([-tangent[1], tangent[0]])

        p1_left = (pt1 + half_width * normal).astype(int)
        p2_left = (pt2 + half_width * normal).astype(int)
        p1_right = (pt1 - half_width * normal).astype(int)
        p2_right = (pt2 - half_width * normal).astype(int)

        cv2.line(image, tuple(pt1), tuple(pt2), color, 2)

        cv2.line(image, tuple(p1_left), tuple(p2_left), (255, 0, 0), 2)
        cv2.line(image, tuple(p1_right), tuple(p2_right), (255, 0, 0), 2)


def distance_from_path(contour, x, y):
    return cv2.pointPolygonTest(contour, (x, y), True)