import cv2


class ImageLoader():
    def __init__(self, imagePath):
        self.imagePath = imagePath
        self.image = cv2.imread(imagePath)
        if self.image is None:
            print("Image could not be loaded from : " + imagePath)
        self.window_name = "my_window"
        cv2.namedWindow("my_window")
        

    def show(self):
        cv2.imshow(self.window_name, self.image)

    def show_and_wait(self):
        self.show()
        cv2.waitKey(0)
        self.clear()

    def clear(self):
        cv2.destroyWindow(self.window_name)


if __name__ == '__main__':
    ImageLoader('Textures/texture3.png').show()