import cv2
import numpy as np
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
        
        rows, cols = self.image.shape[:2]
        num_rows = rows // grid_height
        num_cols = cols // grid_width
        
        half_path_width = self.path_width // 2

        for x, y in zip(x_new, y_new):
            for dx in range(-half_path_width, half_path_width + 1, grid_width // 8):
                for dy in range(-half_path_width, half_path_width + 1, grid_height // 8):   
                    grid_x = int((x + dx) // grid_width)
                    grid_y = int((y + dy) // grid_height)

                    if 0 <= grid_x <= num_cols and 0 <= grid_y <= num_rows:
                        
                        # Calculate the top-left corner of the grid cell
                        top_left_x = grid_x * grid_width
                        top_left_y = grid_y * grid_height

                        # Clip texture to fit within canvas boundaries
                        x_start = int(max(0, top_left_x))
                        y_start = int(max(0, top_left_y))
                        x_end = int(min(self.image.shape[1], x_start + grid_width))
                        y_end = int(min(self.image.shape[0], y_start + grid_height))

                        texture_roi = resized_texture[: y_end - y_start, : x_end - x_start]

                        self.image[y_start:y_end, x_start:x_end] = texture_roi
                
        self.draw_grid()

    def draw_grid(self):
        grid_width, grid_height = self.grid_size
        rows, cols = self.image.shape[:2]
        
        for x in range(0, cols, grid_width):
            cv2.line(self.image, (x, 0), (x, rows), (255, 255, 255), 1)
        
        for y in range(0, rows, grid_height):
            cv2.line(self.image, (0, y), (cols, y), (255, 255, 255), 1)