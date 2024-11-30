import cv2
from PathDrawer import PathDrawer

texture = cv2.imread("Textures/texture3.png")
grass_texture = cv2.imread("Textures/tilable-img_0044-dark.png")

path_drawer = PathDrawer(texture, grass_texture)
path_width = 10

cv2.imshow("Path from Points", path_drawer.image)
cv2.setMouseCallback("Path from Points", path_drawer.select_points)

while True:
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('d'):  # Draw the path when 'd' is pressed
        path_drawer.draw_path_with_texture()
        cv2.imshow("Path from Points", path_drawer.image)

    elif key == ord('u'):  # Undo the last point when 'u' is pressed
        if len(path_drawer.selected_points) > 0:
            path_drawer.selected_points.pop()
            path_drawer.reset_image()
            for point in path_drawer.selected_points:
                cv2.circle(path_drawer.image, point, 5, path_drawer.points_color, -1)
            cv2.imshow("Path from Points", path_drawer.image)
            
    elif key == ord('t'):  # Toggle the spline drawing when 't' is pressed
        path_drawer.toggle_spline(path_width)

    elif key == ord('q'):  # Exit the loop when 'q' is pressed
        break

cv2.destroyAllWindows()
