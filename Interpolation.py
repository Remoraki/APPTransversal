import cv2
import numpy as np
from scipy.interpolate import make_interp_spline
import sys

class LoadIm():
    def __init__(self, width=800, height=600, texture_path="Textures/grass.jpg"):
        self.width = width
        self.height = height
        self.background = (0, 0, 0)
        self.cursor = (0, 0, 255)

        

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
                if len(points) > 1:
                    # Tri des points pour s'assurer que x est strictement croissant
                    sorted_points = points[np.argsort(points[:, 0])]
                    x_sorted = sorted_points[:, 0]
                    y_sorted = sorted_points[:, 1]

                    # Créer la spline
                    spline = make_interp_spline(x_sorted, y_sorted, k=3)  # k=3 pour une spline cubique

                    # Tracer la courbe d'interpolation
                    nb_points = 700
                    x_new = np.linspace(x_sorted[0], x_sorted[-1], nb_points)
                    y_new = spline(x_new)

                    for elem_x, elem_y in zip(x_new, y_new):
                        # Vérifier que y_value n'est pas NaN
                        if not np.isnan(elem_y):
                            # Convertir les coordonnées en entiers pour OpenCV
                            int_x, int_y = int(elem_x), int(elem_y)
                            if 0 <= int_x < self.width and 0 <= int_y < self.height:
                                print(f"Tracé pour x={elem_x}: y={elem_y}")  # Affichez les valeurs pour le débogage
                                cv2.circle(self.background_image, (int_x, int_y), 7, self.cursor, -1)
                            je_peux = False

            # Quitter si 'q' est pressé
            if key == ord('q'):
                break

        # Quitter OpenCV
        cv2.destroyAllWindows()


if __name__ == "__main__":
    drawer = LoadIm()
    drawer.run()