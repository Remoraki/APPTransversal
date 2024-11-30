import cv2
import numpy as np
from scipy.interpolate import splprep, splev

class PathDrawer:
    def __init__(self, path_texture, grid_size=(50, 50), points_color=(255, 255, 255), line_color=(255, 0, 255)):
        self.selected_points = []
        self.points_color = points_color
        self.line_color = line_color
        self.image = np.zeros((500, 500, 3), dtype=np.uint8)
        self.path_texture = path_texture
        self.grid_size = grid_size
    
    def select_points(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_points.append((x, y))
            cv2.circle(self.image, (x, y), 5, self.points_color, -1)
            cv2.imshow("Path from Points", self.image)
    
    def points_loop(self):
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('d'):     # Draw path
                break
            if key == ord('u'):
                if len(self.selected_points) > 0:
                    self.selected_points.pop()
                    self.image = np.zeros((500, 500, 3), dtype=np.uint8)
                    for point in self.selected_points:
                        cv2.circle(self.image, point, 5, self.points_color, -1)
                    cv2.imshow("Path from Points", self.image)
    
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
        else:
            #TODO: add linear and cubic interpolation
            print("Need at least 4 points to draw a path")
            self.points_loop()
            self.draw_spline()
    
    def draw_path_with_texture(self):
        if len(self.selected_points) > 3:
            # Create a grid of the specified size
            grid_width, grid_height = self.grid_size
            rows, cols = self.image.shape[:2]
            num_rows = rows // grid_height
            num_cols = cols // grid_width

            # Resize the texture to fit the grid cells
            resized_texture = cv2.resize(self.path_texture, (grid_width, grid_height))
            
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

                # Draw the texture in the grid cell
                x_start = max(0, top_left_x)
                y_start = max(0, top_left_y)
                x_end = min(self.image.shape[1], x_start + grid_width)
                y_end = min(self.image.shape[0], y_start + grid_height)

                # Blend the texture into the grid cell
                sub_image = self.image[y_start:y_end, x_start:x_end]
                texture_region = resized_texture[: y_end - y_start, : x_end - x_start]
                blended_region = cv2.addWeighted(sub_image, 0.5, texture_region, 0.5, 0)
                self.image[y_start:y_end, x_start:x_end] = blended_region

            cv2.imshow("Path from Points", self.image)
            #self.draw_spline() 

        else:
            #TODO: add linear and cubic interpolation
            print("Need at least 4 points to draw a path")
            self.points_loop()
            self.draw_path()