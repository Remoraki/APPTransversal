import cv2
import numpy as np
from shapely.geometry import Polygon
from utils import *
class PathDrawer:
    def __init__(
        self, path_texture, path_texture_mask, grass_texture, 
        height=600, width=800, grid_size=(64, 64), path_width=70,
        points_color=(255, 255, 255), line_color=(255, 0, 255)
    ):
        self.selected_points = []
        self.path_width = path_width
        
        self.points_color = points_color
        self.line_color = line_color
        
        self.image = np.zeros((height, width, 3), dtype=np.uint8)
        self.mask = np.zeros((height, width, 3), dtype=np.uint8)
        
        self.path_texture = path_texture
        self.grid_size = grid_size
        self.grass_texture = cv2.resize(grass_texture, (256, 256))
        self.path_texture_mask = path_texture_mask
        
        self.image = set_background_with_texture(self.image, self.grass_texture)
        self.background = np.copy(self.image)
        
        self.mask = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)
        
        self.draw_grid()
        
        self.spline_drawn = False
        
        nb_cols = width // grid_size[0] + width % grid_size[0]
        nb_rows = height // grid_size[1] + height % grid_size[1]
        
        self.grid = np.zeros((nb_rows, nb_cols), dtype=bool)
    
    def select_points(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_points.append((x, y))
            cv2.circle(self.image, (x, y), 5, self.points_color, -1)
            cv2.imshow("Path from Points", self.image)
            
    def reset_image(self):
        self.image = np.copy(self.background)
        self.draw_grid()
    
    def draw_path(self):
        if len(self.selected_points) == 2:
            draw_line_with_width(
                self.image, 
                self.selected_points, 
                self.line_color,
                self.path_width
            )
            self.spline_drawn = True
        elif len(self.selected_points) > 2:
            draw_spline_with_width(
                self.image, 
                self.selected_points, 
                self.line_color,
                self.path_width
            )
            self.spline_drawn = True
        else:
            print("Need at least 2 points to draw a path")
    
    def toggle_spline(self):
        if self.spline_drawn:
            self.spline_drawn = False
            self.reset_image()
            for point in self.selected_points:
                cv2.circle(self.image, point, 5, self.points_color, -1)
            cv2.imshow("Path from Points", self.image)
        else:
            self.draw_path()
            cv2.imshow("Path from Points", self.image)
            
    def calculate_angle(self, x1, y1, x2, y2, tangent=None):
        if tangent is None:
            return np.degrees(np.arctan2(y2 - y1, x2 - x1))
        else:
            return np.degrees(np.arctan2(tangent[1], tangent[0]))
    
    def calculate_point(self, x, y, angle, distance):
        angle = np.radians(angle)
        x_new = x + distance * np.cos(angle)
        y_new = y + distance * np.sin(angle)
        return x_new, y_new
    
    def draw_path_with_texture(self):
        self.reset_image()
        if len(self.selected_points) == 2:
            x_new, y_new = calculate_lerp_points(self.selected_points[0], self.selected_points[1])
        elif len(self.selected_points) > 2:
            x_new, y_new, tangents = calculate_spline_with_tangents(self.selected_points)
        else:
            print("Need at least 2 points to draw a path")
            return
    
        grid_width, grid_height = self.grid_size
        resized_texture = cv2.resize(self.path_texture, self.grid_size)
        resized_texture_mask = cv2.resize(self.path_texture_mask, self.grid_size)
        
        if len(resized_texture_mask.shape) == 3:
            resized_texture_mask = cv2.cvtColor(resized_texture_mask, cv2.COLOR_BGR2GRAY)
        
        contours, _ = cv2.findContours(resized_texture_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rows, cols = self.image.shape[:2]
        num_rows = rows // grid_height
        num_cols = cols // grid_width
        
        half_path_width = self.path_width // 2
        
        left_boundary = np.zeros((len(x_new), 2))
        right_boundary = np.zeros((len(x_new), 2))
        
        for i, (x, y) in enumerate(zip(x_new, y_new)):
            if i == 0:
                continue
            
            if len(self.selected_points) == 2:
                angle = self.calculate_angle(x_new[i - 1], y_new[i - 1], x, y)
            else:
                angle = self.calculate_angle(x_new[i - 1], y_new[i - 1], x, y, tangents[i - 1])
            
            left_boundary[i] = self.calculate_point(x, y, angle + 90, half_path_width)
            right_boundary[i] = self.calculate_point(x, y, angle - 90, half_path_width)
            
            left_boundary[i] = (int(left_boundary[i][0]), int(left_boundary[i][1]))
            right_boundary[i] = (int(right_boundary[i][0]), int(right_boundary[i][1]))
        
        path_polygon = Polygon(np.concatenate((left_boundary, right_boundary[::-1])))
        
        mask = np.zeros((rows, cols), dtype=np.uint8)
        
        for i in range(num_rows):
            for j in range(num_cols):
                top_left = (j * grid_width, i * grid_height)
                top_right = (j * grid_width + grid_width, i * grid_height)
                
                grid_polygon = Polygon([top_left, top_right, (top_right[0], top_right[1] + grid_height), (top_left[0], top_left[1] + grid_height)])
                
                if path_polygon.intersects(grid_polygon):
                    cv2.fillPoly(mask, [np.array([top_left, top_right, (top_right[0], top_right[1] + grid_height), (top_left[0], top_left[1] + grid_height)])], 255)
        
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask = cv2.bitwise_and(mask, self.image)
        
        filtered_canvas = np.zeros_like(self.image)
        
        for contour in contours:
            contour_points = [tuple(point[0]) for point in contour]
            
            if len(contour_points) >= 3:  # A valid polygon requires at least 3 points
                contour_polygon = Polygon(contour_points)

                if path_polygon.intersects(contour_polygon):
                    cv2.fillPoly(filtered_canvas, [np.array(contour_points)], (255, 255, 255))
                    
        filtered_canvas = cv2.bitwise_and(filtered_canvas, mask)
        self.image = cv2.addWeighted(self.image, 1, filtered_canvas, 1, 0)
        
                
        self.draw_grid()

    def draw_grid(self):
        grid_width, grid_height = self.grid_size
        rows, cols = self.image.shape[:2]
        
        for x in range(0, cols, grid_width):
            cv2.line(self.image, (x, 0), (x, rows), (255, 255, 255), 1)
        
        for y in range(0, rows, grid_height):
            cv2.line(self.image, (0, y), (cols, y), (255, 255, 255), 1)