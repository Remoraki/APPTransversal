import cv2
from PathDrawer import PathDrawer

POINT_COLOR = (255, 255, 255)
LINE_COLOR = (255, 0, 255)

texture = cv2.imread("Textures/texture3.png")
grass_texture = cv2.imread("Textures/tilable-img_0044-dark.png")

PathDrawer = PathDrawer(texture, grass_texture, points_color=POINT_COLOR, line_color=LINE_COLOR)

cv2.imshow("Path from Points", PathDrawer.image)
cv2.setMouseCallback("Path from Points", PathDrawer.select_points)

PathDrawer.points_loop()

PathDrawer.draw_path_with_texture()

cv2.imshow("Path from Points", PathDrawer.image)
cv2.waitKey(0)
cv2.destroyAllWindows()
