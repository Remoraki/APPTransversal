import cv2
import numpy as np
from utils import *
class PathDrawer:
    def __init__(
        self, path_texture, grass_texture, 
        height=600, width=800, grid_size=(64, 64), path_width=100,
        points_color=(255, 255, 255), line_color=(255, 0, 255)
    ):
        self.selected_points = []
        self.path_width = path_width
        
        self.points_color = points_color
        self.line_color = line_color
        
        self.image = np.zeros((height, width, 3), dtype=np.uint8)
        
        self.path_texture = path_texture
        self.grid_size = grid_size
        self.grass_texture = cv2.resize(grass_texture, (256, 256))
        
        self.image = set_background_with_texture(self.image, self.grass_texture)
        self.background = np.copy(self.image)
        
        self.spline_drawn = False
    
    def select_points(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_points.append((x, y))
            cv2.circle(self.image, (x, y), 5, self.points_color, -1)
            cv2.imshow("Path from Points", self.image)
            
    def reset_image(self):
        self.image = np.copy(self.background) 
    
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
    
    def draw_path_with_texture(self):
        self.reset_image()
        if len(self.selected_points) == 2:
            x_new, y_new = calculate_lerp_points(self.selected_points[0], self.selected_points[1])
        elif len(self.selected_points) > 2:
            x_new, y_new = calculate_spline(self.selected_points)
        else:
            print("Need at least 2 points to draw a path")
            return
    
        grid_width, grid_height = self.grid_size
        resized_texture = cv2.resize(self.path_texture, self.grid_size)

        # Determine which grid cells the path overlaps
        for x, y in zip(x_new, y_new):
            grid_x = int(x // grid_width)
            grid_y = int(y // grid_height)

            top_left_x = grid_x * grid_width
            top_left_y = grid_y * grid_height

            x_start = max(0, top_left_x)
            y_start = max(0, top_left_y)
            x_end = min(self.image.shape[1], x_start + grid_width)
            y_end = min(self.image.shape[0], y_start + grid_height)

            texture_roi = resized_texture[: y_end - y_start, : x_end - x_start]

            self.image[y_start:y_end, x_start:x_end] = texture_roi
