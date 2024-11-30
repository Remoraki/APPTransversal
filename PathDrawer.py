import cv2
import numpy as np
from scipy.interpolate import splprep, splev

class PathDrawer:
    def __init__(self, path_texture, grass_texture, height=600, width=800, grid_size=(64, 64), points_color=(255, 255, 255), line_color=(255, 0, 255)):
        self.selected_points = []
        self.points_color = points_color
        self.line_color = line_color
        self.image = np.zeros((height, width, 3), dtype=np.uint8)
        self.path_texture = path_texture
        self.grid_size = grid_size
        self.grass_texture = cv2.resize(grass_texture, (256, 256))
        self.image = self.set_background_with_texture(self.image, self.grass_texture)
        self.background = np.copy(self.image)
        self.spline_drawn = False
    
    def select_points(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_points.append((x, y))
            cv2.circle(self.image, (x, y), 5, self.points_color, -1)
            cv2.imshow("Path from Points", self.image)
            
    def reset_image(self):
        self.image = np.copy(self.background) 
            
    def set_background_with_texture(self, canvas, texture):
        # Get dimensions
        canvas_h, canvas_w = canvas.shape[:2]
        texture_h, texture_w = texture.shape[:2]

        # Tile the texture
        for y in range(0, canvas_h, texture_h):
            for x in range(0, canvas_w, texture_w):
                # Determine the region of interest (ROI) on the canvas
                x_end = min(x + texture_w, canvas_w)
                y_end = min(y + texture_h, canvas_h)

                # Determine the corresponding ROI from the texture
                texture_roi = texture[: y_end - y, : x_end - x]

                # Paste the texture into the canvas
                canvas[y:y_end, x:x_end] = texture_roi

        return canvas
    
    def draw_spline(self):
        if len(self.selected_points) > 3:
            points = np.array(self.selected_points)
            tck, u = splprep([points[:, 0], points[:, 1]], s=0)
            u_fine = np.linspace(0, 1, 500)
            x_new, y_new = splev(u_fine, tck)
            for i in range(len(x_new) - 1):
                pt1 = (int(x_new[i]), int(y_new[i]))
                pt2 = (int(x_new[i + 1]), int(y_new[i + 1]))
                cv2.line(self.image, pt1, pt2, self.line_color, 2)
            self.spline_drawn = True
        else:
            #TODO: add linear and cubic interpolation
            print("Need at least 4 points to draw a path")
    
    def toggle_spline(self):
        if self.spline_drawn:
            self.spline_drawn = False
            self.reset_image()  # Reset the image if the spline is being hidden
            for point in self.selected_points:
                cv2.circle(self.image, point, 5, self.points_color, -1)
            cv2.imshow("Path from Points", self.image)
        else:
            self.draw_spline()  # Redraw the spline if it was hidden
            cv2.imshow("Path from Points", self.image)
    
    def draw_path_with_texture(self):
        if len(self.selected_points) > 3:
            # Create a grid of the specified size
            grid_width, grid_height = self.grid_size
            rows, cols = self.image.shape[:2]
            num_rows = rows // grid_height
            num_cols = cols // grid_width

            # Resize the texture to fit the grid cells
            resized_texture = cv2.resize(self.path_texture, (grid_width, grid_height))

            # Calculate the spline path
            points = np.array(self.selected_points)
            tck, u = splprep([points[:, 0], points[:, 1]], s=0)
            u_fine = np.linspace(0, 1, 500)
            x_new, y_new = splev(u_fine, tck)

            # Determine which grid cells the path overlaps
            for x, y in zip(x_new, y_new):
                grid_x = int(x // grid_width)
                grid_y = int(y // grid_height)

                # Calculate the top-left corner of the grid cell
                top_left_x = grid_x * grid_width
                top_left_y = grid_y * grid_height

                # Clip texture to fit within canvas boundaries
                x_start = max(0, top_left_x)
                y_start = max(0, top_left_y)
                x_end = min(self.image.shape[1], x_start + grid_width)
                y_end = min(self.image.shape[0], y_start + grid_height)

                # Compute the region of interest (ROI) on the texture
                texture_roi = resized_texture[
                    : y_end - y_start, : x_end - x_start
                ]

                # Blend the texture into the canvas
                self.image[y_start:y_end, x_start:x_end] = texture_roi
        else:
            #TODO: add linear and cubic interpolation
            print("Need at least 4 points to draw a path")
