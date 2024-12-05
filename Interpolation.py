import cv2
import numpy as np
from scipy.interpolate import make_interp_spline, splprep,splev
import sys
from chemin import Chemin 

class LoadIm():
    def __init__(self, width=800, height=600, texture_path="Textures/grass.jpg"):
        self.width = width
        self.height = height
        self.background = (0, 0, 0)
        self.cursor = (0, 0, 255)

        self.chemin = Chemin(width, height, 14,13)

        # Charger l'image de texture (background)
        self.background_image = cv2.imread(texture_path)
        if self.background_image is None:
            print("Erreur : impossible de charger l'image de texture.")
            sys.exit()

        # Redimensionner l'image de texture pour qu'elle corresponde à la taille de l'écran
        self.background_image = cv2.resize(self.background_image, (self.width, self.height))

        # Liste pour stocker les points
        self.points = []

        cv2.namedWindow("Ajouter des points rouges")
        cv2.setMouseCallback("Ajouter des points rouges", self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # Clic gauche
            self.points.append((x, y))
            cv2.circle(self.screen, (x, y), 5, self.cursor, -1)  # Cercle type curseur

    def run(self):
        # Boucle principale
        je_peux = True
        while True:
            # Appliquer la texture de fond
            self.screen = self.background_image.copy()

            self.chemin.draw_grid(self.screen)
            # Afficher les points ajoutés sur l'image
            for point in self.points:
                cv2.circle(self.screen, (point[0], point[1]), 5, self.cursor, -1)

            # Afficher l'image
            cv2.imshow("Ajouter des points rouges", self.screen)

            # Attendre une touche
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13 and je_peux:  # Entrée pour activer l'interpolation
                points = np.array(self.points)
                
                # Assurez-vous que vous avez au moins deux points pour l'interpolation
                x, y = points[:, 0], points[:, 1]
    
                nb_points = 700
                # Générer le spline paramétrique
                tck, u = splprep([x, y], s=0)
                new_points = splev(np.linspace(0, 1, nb_points), tck)

                x_new, y_new = new_points[0], new_points[1]
                for elem_x, elem_y in zip(x_new, y_new):
                    # Vérifier que y_value n'est pas NaN
                    if not np.isnan(elem_y):
                        # Convertir les coordonnées en entiers pour OpenCV
                        int_x, int_y = int(elem_x), int(elem_y)
                        if 0 <= int_x < self.width and 0 <= int_y < self.height:
                            print(f"Tracé pour x={elem_x}: y={elem_y}")  # Affichez les valeurs pour le débogage
                            cv2.circle(self.background_image, (int_x, int_y), 5, self.cursor, -1)
                        je_peux = False

            if key == ord('r'):
                self.chemin.apply_texture_to_grid(self.background_image, self.points)

            # Quitter si 'q' est pressé
            if key == ord('q') or key == 27:
                break

        # Quitter OpenCV
        cv2.destroyAllWindows()


if __name__ == "__main__":
    drawer = LoadIm()
    drawer.run()