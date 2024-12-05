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
        self.grid_shape = 0
        self.grid_x = []
        self.grid_y = []
        self.texture_points = []

    def sample_grid(self, n):
        h,w = self.loader.image.shape[:2]
        self.n = n
        self.grid_x = np.linspace(0, w-1, n+1)
        self.grid_y = np.linspace(0, h-1, n+1)
        self.loader.resize_road_absolute(int(h/n), int(w/n))

    def draw_on_grid(self):
        self.texture_points = []
        for x in self.grid_x:
            for y in self.grid_y:
                if self.path.path_mask[int(y),int(x)]:
                    loader.draw_road(int(y), int(x))
                    self.texture_points.append((x,y))

    
    def remove_outside(self):
        # TO DO
        pass

          

if __name__ == '__main__':
    loader = ImagePathLoader(1, bg=np.zeros((500,500,3)), roadPath= "Textures/texture3.png")

    path = PathCreator(loader=loader)
    path.capture_input(5)
    path.create_path(k=3, n=1000)
    path.draw_bounds(d=20)

    segmenter = ImageSegmenter(2, maskPath="Textures/texture3_masque.png")

    texturer = PathTexturer(path, loader, segmenter)
    texturer.sample_grid(20)
    texturer.draw_on_grid()

    loader.show_and_wait()
