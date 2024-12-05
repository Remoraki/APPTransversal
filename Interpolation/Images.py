import cv2
import numpy as np

class ImageLoader():
    def __init__(self, n, imagePath="", image=None):
        self.imagePath = imagePath
        if image is None:
            self.image = cv2.imread(imagePath)
            if self.image is None:
                print("Image could not be loaded from : " + imagePath)
        else:
            self.image = image
        self.window_name = "my_window" + str(n)
        cv2.namedWindow(self.window_name)
        
    def show(self):
        cv2.imshow(self.window_name, self.image)
        

    def show_and_wait(self):
        self.show()
        cv2.waitKey(0)
        self.clear()

    def clear(self):
        cv2.destroyWindow(self.window_name)

class ImagePathLoader(ImageLoader):
    def __init__(self, n, bgPath="", bg=None, roadPath="", road=None):
        super().__init__(n, bgPath, bg)
        if road is None:
            self.road = cv2.imread(roadPath)
            if self.road is None:
                print("Image could not be loaded from : " + roadPath)

    def show_road(self):
        cv2.imshow(self.window_name, self.road)
        cv2.waitKey(0)

    def resize_road(self, p):
        h,w = self.image.shape[:2]
        new_h = int(h * p) 
        new_w = int(w * p)
        self.road = cv2.resize(self.road, (new_h, new_w))

    def resize_road_absolute(self, h, w):
        self.road = cv2.resize(self.road, (h,w))


    def get_draw_coordinates(self, x, y):
        rh, rw = self.road.shape[:2]
        bgh, bgw = self.image.shape[:2]

        indices_x = np.array(range(rw))
        draw_indices_x = np.array(range(int(x - rw/2), int(x + rw/2)))
        indices_y = np.array(range(rh))
        draw_indices_y = np.array(range(int(y - rh/2), int(y + rh/2)))
        

        
        inside_x = np.logical_and(draw_indices_x >= 0, draw_indices_x < bgw)
        inside_y = np.logical_and(draw_indices_y >= 0, draw_indices_y < bgh)
        indices_x = indices_x[inside_x]
        draw_indices_x = draw_indices_x[inside_x]
        indices_y = indices_y[inside_y]
        draw_indices_y = draw_indices_y[inside_y]

        return indices_x, draw_indices_x, indices_y, draw_indices_y

    def draw_road(self, x, y):
        indices_x, draw_indices_x, indices_y, draw_indices_y = self.get_draw_coordinates(x,y)

        for i in range(len(draw_indices_x)):
            for j in range(len(draw_indices_y)):

                    self.image[draw_indices_x[i], draw_indices_y[j], :] = self.road[indices_x[i], indices_y[j], :] / 255


class ImageSegmenter(ImageLoader):
    def __init__(self, n, maskPath="", mask=None):
        super().__init__(n, imagePath=maskPath, image=mask)
        
        self.mask = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        
        self.num_labels, self.labels = cv2.connectedComponents(self.mask)

    def get_mask_for_label(self, label):
        return (self.labels == label).astype(int)

    def draw_segmentation(self):
        for label in range(self.num_labels):
            component_mask = (self.labels == label).astype(int) 
            n,m = component_mask.shape
            to_draw = np.ones((n,m,3))
            to_draw[:,:,0] = component_mask
            to_draw[:,:,1] = component_mask
            to_draw[:,:,2] = component_mask
            cv2.imshow(self.window_name, to_draw)
            cv2.waitKey(0)
        

def test_loader():
    loader = ImageLoader(0, imagePath='Textures/texture3.png')
    loader.show_and_wait()

def test_path_loader():
    loader = ImagePathLoader(0, bg=np.zeros((500,500,3)), roadPath='Textures/texture3.png')
    loader.show_road()
    loader.resize_road(0.1)
    loader.draw_road(50, 50)
    loader.show_and_wait()

def test_segmenter():
    segmenter = ImageSegmenter(0, maskPath="Textures/texture3_masque.png")
    segmenter.draw_segmentation()

if __name__ == '__main__':
    test_segmenter()