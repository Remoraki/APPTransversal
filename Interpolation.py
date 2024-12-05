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

    def draw_parallel_lines(self, x_new, y_new, thickness):
        # Créer un masque binaire vide (tout noir)
        mask = np.zeros((self.height, self.width), dtype=np.uint8)

        for i in range(len(x_new) - 1):
            x1, y1 = x_new[i], y_new[i]
            x2, y2 = x_new[i + 1], y_new[i + 1]

            dx, dy = x2 - x1, y2 - y1
            length = np.sqrt(dx ** 2 + dy ** 2)
            nx, ny = -dy / length, dx / length

            offset_x = thickness * nx
            offset_y = thickness * ny

            # Points pour la première ligne parallèle
            int_x1a, int_y1a = int(x1 + offset_x), int(y1 + offset_y)
            int_x2a, int_y2a = int(x2 + offset_x), int(y2 + offset_y)

            # Points pour la seconde ligne parallèle
            int_x1b, int_y1b = int(x1 - offset_x), int(y1 - offset_y)
            int_x2b, int_y2b = int(x2 - offset_x), int(y2 - offset_y)

            # Tracer les lignes parallèles sur le masque
            cv2.line(mask, (int_x1a, int_y1a), (int_x2a, int_y2a), 255, thickness)
            cv2.line(mask, (int_x1b, int_y1b), (int_x2b, int_y2b), 255, thickness)

            # Révéler les tuiles touchées avec update_visibility (optionnel)
            self.chemin.update_visibility(int_x1a, int_y1a)
            self.chemin.update_visibility(int_x2a, int_y2a)
            self.chemin.update_visibility(int_x1b, int_y1b)
            self.chemin.update_visibility(int_x2b, int_y2b)

        return mask


    def run(self):
        # Boucle principale
        je_peux = True
        self.screen = self.background_image.copy()
        while True:
            # Appliquer la texture de fond
            

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
                            self.chemin.update_visibility(int_x, int_y)
                            # 30 pixels d'écart 
                            self.draw_parallel_lines(x_new, y_new, 10)
                            self.points.append((int_x, int_y))
                            cv2.circle(self.screen, (int_x, int_y), 3, self.cursor, -1)
                        je_peux = False

            if key == ord('r'):
                self.chemin.apply_texture_to_grid(self.screen)
                for point in self.points:
                    cv2.circle(self.screen, (point[0], point[1]), 5, self.cursor, -1)
                #for points 


            # Quitter si 'q' est pressé
            if key == ord('q') or key == 27:
                break

        # Quitter OpenCV
        cv2.destroyAllWindows()


if __name__ == "__main__":
    drawer = LoadIm()
    drawer.run()