import cv2
import numpy as np
import pygame
import sys 
from scipy.interpolate import griddata,interp1d


class LoadIm():
    def __init__(self, width=800, height=600):
        

        self.width = width
        self.height = height
        self.background = (0,0,0)
        self.cursor = (0,0,255)

        self.screen = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        
        # Liste pour stocker les points
        self.points = [(0,np.random.randint(0,self.height))]
        cv2.namedWindow("Ajouter des points rouges")
        cv2.setMouseCallback("Ajouter des points rouges", self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # Clic gauche
            self.points.append((x, y))
            cv2.circle(self.screen, (x, y), 5, self.cursor, -1) # cercle type curseur


    # def interpol(self):

    #     grid_z = griddata(self.points, method='linear')
    #     return grid_z

    def run(self):
        # Boucle principale
        while True:
            # Afficher l'image
            cv2.imshow("Ajouter des points rouges", self.screen)

            # Attendre une touche
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13:
                # all_points = self.interpol()
                # print(all_points)
                self.points.append((self.width,np.random.randint(0,self.height)))
                points = np.array(self.points)
                fonct = interp1d(points[:,0], points[:,1])


                nb_points = 400
                for elem_x in np.linspace(0,self.width,nb_points):

                    cv2.circle(self.screen, (elem_x, fonct(elem_x)), 5, self.cursor, -1)
            # Quitter si 'q' est pressé
            if key == ord('q'):
                break
            
           
        # Quitter OpenCV
        cv2.destroyAllWindows()



if __name__ == "__main__":
    drawer = LoadIm()
    drawer.run()




#       File "./Interpolation.py", line 56, in run
#     cv2.circle(self.screen, (elem_x, fonct(elem_x)), 5, self.cursor, -1)
# cv2.error: OpenCV(4.10.0) :-1: error: (-5:Bad argument) in function 'circle'
# > Overload resolution failed:
# >  - Can't parse 'center'. Sequence item with index 0 has a wrong type
# >  - Can't parse 'center'. Sequence item with index 0 has a wrong type

# (myenv) nlermusi@forge:~/S9/Transverse/APPTransversal$ 

# The error you're encountering indicates that the arguments being passed to the cv2.circle function are not of the expected type. Specifically, the center argument (which is a tuple representing the coordinates of the circle's center) must contain integers, but it seems that elem_x or the result of fonct(elem_x) is not being converted to an integer.

# Here's how you can fix this issue:

#     Convert Coordinates to Integers: Ensure that both elem_x and the result of fonct(elem_x) are converted to integers before passing them to cv2.circle.

#     Handle NaN Values: If fonct(elem_x) returns NaN (which can happen if the input is out of bounds and you haven't set fill_value properly), you should skip drawing that point.

# Here’s the modified run method with these considerations:

# python

# def run(self):
#     # Boucle principale
#     while True:
#         # Afficher l'image
#         cv2.imshow("Ajouter des points rouges", self.screen)

#         # Attendre une touche
#         key = cv2.waitKey(1) & 0xFF
        
#         if key == 13:
#             points = np.array(self.points)
#             if len(points) < 2:
#                 print("Please add at least two points for interpolation.")
#                 continue
            
#             # Create the interpolation function with fill_value
#             fonct = interp1d(points[:, 0], points[:, 1], bounds_error=False, fill_value="extrapolate")

#             nb_points = 400
#             for elem_x in np.linspace(0, self.width, nb_points):
#                 # Use the interpolation function
#                 y_value = fonct(elem_x)
                
#                 # Check if y_value is valid (not NaN)
#                 if np.isnan(y_value):
#                     continue
                
#                 # Convert to integers for OpenCV
#                 cv2.circle(self.screen, (int(elem_x), int(y_value)), 5, self.cursor, -1)

#         # Quitter si 'q' est pressé
#         if key == ord('q'):
#             break
            
#     # Quitter OpenCV
#     cv2.destroyAllWindows()
