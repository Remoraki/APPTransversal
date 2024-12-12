import cv2
from Images import ImageLoader, ImagePathLoader, ImageSegmenter
from scipy.interpolate import splprep, splev
import numpy as np


class PathCreator():
    def __init__(self, loader):
        self.loader = loader
        self.nb_of_points = 0
        self.X_input = []
        self.Y_input = []
        self.path = None

    def draw_point(self, x, y, color = (0,0,0)):
        cv2.circle(self.loader.image, (x,y), 5, color, -1)

    def on_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.draw_point(x, y, (255,0,0))
            self.X_input.append(x)
            self.Y_input.append(y)
            self.nb_of_points += 1

    def capture_input(self, nb_of_points):
        cv2.setMouseCallback(self.loader.window_name, self.on_click)
        self.nb_of_points = 0
        self.X_input = []
        self.Y_input = []
        while(self.nb_of_points < nb_of_points):
            self.loader.show()
            if cv2.waitKey(20) & 0xFF == 27:
                break

    def create_path(self, k, n):
        tck, u = splprep(np.stack([self.X_input, self.Y_input]), s=0, k=k)
        new_u = np.linspace(0, 1, n)
        self.path = splev(new_u, tck)
        X = self.path[0]
        Y = self.path[1]
        for i in range(len(X)):
            self.draw_point(round(X[i]), round(Y[i]))

    def draw_bounds(self, d):
        d *= 2
        height, width = self.loader.image.shape[:2]
        y_d = range(height)
        x_d = range(width)
        x_d, y_d = np.meshgrid(x_d, y_d)
        D = np.inf * np.ones_like(x_d)
        x_p = self.path[0]
        y_p = self.path[1]
        for i in range(len(x_p)):
            d_x = x_d - round(x_p[i])
            d_y = y_d - round(y_p[i])
            d_i = np.sqrt(np.square(d_x) + np.square(d_y))
            D = np.minimum(D, d_i)
        mask = (D <= d).astype(int)
        to_draw = np.ones((height, width, 3))
        to_draw[:, :, 0] *= mask
        to_draw[:, :, 1] *= mask 
        to_draw[:, :, 2] *= mask
        self.path_mask = mask
        self.path_sdf = D
        ImageLoader(-1, image=to_draw).show_and_wait()


class PathTexturer():
    def __init__(self, path : PathCreator, loader : ImagePathLoader, segmenter : ImageSegmenter):
        self.loader = loader
        self.path = path
        self.segmenter = segmenter
        self.grid_shape = 0
        self.grid_x = []
        self.grid_y = []
        self.grid_mask = []
        self.texture_points = []
        self.texture_points_coordinates = []

    def sample_grid(self, n):
        h,w = self.loader.image.shape[:2]
        self.n = n
        self.grid_x = np.linspace(0, w-1, n+1)
        self.grid_y = np.linspace(0, h-1, n+1)
        self.loader.resize_road_absolute(int(h/n), int(w/n))

    def draw_on_grid(self):
        self.texture_points = []
        self.texture_points_coordinates = []
        self.grid_mask = np.zeros((self.n + 1, self.n + 1)).astype(bool)
        for i in range(self.n+1):
            x = self.grid_x[i]
            for j in range(self.n+1):
                y = self.grid_y[j]
                if self.path.path_mask[int(y),int(x)]:
                    loader.draw_road(int(y), int(x))
                    self.texture_points.append([int(x),int(y)])
                    self.texture_points_coordinates.append((i,j))
                    self.grid_mask[i,j] = True      


    def get_border_info(self, i, j):
        info = [False, False, False, False]
        n,m = self.path.path_mask.shape
        if j == 0:
            info[0] = True
        elif not (self.grid_mask[i,j-1]):
            info[0] = True
        if i == 0:
            info[1] = True
        elif not (self.grid_mask[i-1,j]):
            info[1] = True
        if i == m-1:
            info[2] = True
        elif not (self.grid_mask[i+1,j]):
            info[2] = True
        if j == n-1:
            info[3] = True
        elif not (self.grid_mask[i,j+1]):
            info[3] = True
        return info
    
    def remove_at(self, x, y, mask):
        indices_x, draw_indices_x, indices_y, draw_indices_y = self.loader.get_draw_coordinates(x, y)
        for i in range(len(draw_indices_x)):
            for j in range(len(draw_indices_y)):
                self.loader.image[draw_indices_y[j], draw_indices_x[i], :] *= (np.invert(mask[indices_y[j], indices_x[i]]))
  
    def remove_outside(self): 
        rh,rw = self.loader.road.shape[:2]
        self.segmenter.resize_image(rh, rw)
        self.segmenter.segment()
        border, up, down, left, right = self.segmenter.get_border()
        for i in range(len(self.texture_points)):
            t_p = self.texture_points[i]
            t_p_c = self.texture_points_coordinates[i]
            border_info = self.get_border_info(t_p_c[0], t_p_c[1])
            border_labels = []
            if border_info[0]:
                border_labels += up
            if border_info[1]:
                border_labels += left
            if border_info[2]:
                border_labels += down
            if border_info[3]:
                border_labels += right
            border_labels = set(border_labels)
            mask = self.segmenter.get_components(border_labels)
            self.remove_at(t_p[0], t_p[1], mask.astype(bool))
                


          

if __name__ == '__main__':
    loader = ImagePathLoader(1, bg=np.zeros((500,500,3)), roadPath= "Textures/texture3.png")

    path = PathCreator(loader=loader)
    path.capture_input(3)
    path.create_path(k=1, n=20)
    path.draw_bounds(d=20)

    segmenter = ImageSegmenter(2, maskPath="Textures/texture3_masque.png")

    texturer = PathTexturer(path, loader, segmenter)
    texturer.sample_grid(20)
    texturer.draw_on_grid()
    loader.show_and_wait()
    texturer.remove_outside()
    loader.show_and_wait()
