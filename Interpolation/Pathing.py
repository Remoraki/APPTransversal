import cv2
from Images import ImageLoader
from scipy.interpolate import splprep, splev
import numpy as np


class PathCreator():
    def __init__(self, image_path):
        self.loader = ImageLoader(image_path)
        self.nb_of_points = 0
        self.X = []
        self.Y = []

    def draw_point(self, x, y, color = (0,0,0)):
        print("CENTER")
        print(x,y)
        cv2.circle(self.loader.image, (x,y), 5, color, -1)

    def on_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.draw_point(x, y, (255,0,0))
            self.X.append(x)
            self.Y.append(y)
            self.nb_of_points += 1

    def capture_input(self, nb_of_points):
        cv2.setMouseCallback(self.loader.window_name, self.on_click)
        while(self.nb_of_points < nb_of_points):
            self.loader.show()
            if cv2.waitKey(20) & 0xFF == 27:
                break

    def create_path(self, k):
        tck, u = splprep(np.stack([self.X, self.Y]), s=0)
        new_u = np.linspace(0, 1, 100)
        path = splev(new_u, tck)
        X = path[0]
        Y = path[1]
        print(X)
        print(Y)
        for i in range(len(X)):
            self.draw_point(round(X[i]), round(Y[i]))
        self.loader.show_and_wait()


if __name__ == '__main__':
    path = PathCreator('Textures/texture3.png')
    path.capture_input(10)
    path.create_path(3)
