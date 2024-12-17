import cv2
import numpy as np

class Chemin():
    def __init__(self, width, height, nb_blocksx, nb_blocksy, texture_path="Textures/texture7"):
        self.width = width
        self.height = height
        self.nb_blocksx = nb_blocksx
        self.nb_blocksy = nb_blocksy
        self.block_width = width // nb_blocksx
        self.block_height = height // nb_blocksy

        # Charger l'image de texture
        self.chemin = cv2.imread(f"{texture_path}.png")
        if self.chemin is None:
            print("Erreur : impossible de charger l'image de texture.")
            sys.exit()

        # Redimensionner la texture pour qu'elle corresponde à la taille des blocs
        self.chemin = cv2.resize(self.chemin, (self.block_width, self.block_height))

        self.mask = cv2.imread(f"{texture_path}_masque.png", cv2.IMREAD_GRAYSCALE)

        # Créer une grille de visibilité des blocs
        self.visible_blocks = np.zeros((nb_blocksy, nb_blocksx), dtype=bool)

    def draw_grid(self, screen):
        """Dessine une grille avec un nombre défini de blocs en x et y."""
        color = (255, 255, 255)  # Blanc

        # Lignes horizontales
        for i in range(self.nb_blocksy + 1):
            y = i * self.block_height
            cv2.line(screen, (0, y), (self.width, y), color, 1)

        # Lignes verticales
        for j in range(self.nb_blocksx + 1):
            x = j * self.block_width
            cv2.line(screen, (x, 0), (x, self.height), color, 1)

    def apply_texture_to_grid(self, screen):
        """Applique les textures visibles sur la grille."""
        for i in range(self.nb_blocksy):
            for j in range(self.nb_blocksx):
                if self.visible_blocks[i, j]:
                    x_start = j * self.block_width
                    y_start = i * self.block_height
                    screen[y_start:y_start + self.block_height, x_start:x_start + self.block_width] = self.chemin

    def update_visibility(self, x, y):
        """Met à jour la visibilité des blocs en fonction des points de la ligne."""
        j = x // self.block_width
        i = y // self.block_height
        if 0 <= i < self.nb_blocksy and 0 <= j < self.nb_blocksx:
            self.visible_blocks[i, j] = True

    def repeat_texture(self):
        """Répète la tuile et le masque dans une image 3x3."""
        # Répéter la texture dans une grille 3x3
        repeated_texture = np.tile(self.chemin, (3, 3, 1))

        # Répéter le masque dans une grille 3x3
        repeated_mask = np.tile(self.mask, (3, 3))

        return repeated_texture, repeated_mask

    def find_and_merge_contours(self, mask):
        """Trouver les contours dans l'image répétée et fusionner les cailloux répétitifs."""
        # Trouver les contours dans l'image répétée
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Fusionner les contours répétitifs
        unique_contours = set()
        for contour in contours:
            contour_tuple = tuple(map(tuple, contour.reshape(-1, 2)))
            unique_contours.add(contour_tuple)

        return unique_contours

    def display_unique_contours(self, repeated_texture, unique_contours):
        """Afficher les contours uniques dans une fenêtre séparée."""
        # Créer une image vide pour afficher les contours
        contour_image = repeated_texture.copy()

        cv2.imshow("Masque Répété de ma bitye ", contour_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Dessiner les contours sur l'image
        image_color = cv2.cvtColor(contour_image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(image_color, unique_contours, -1, (255, 255, 0), 1)

        # Afficher l'image avec les contours uniques
        cv2.imshow("Contours Uniques", image_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    

    def load_and_process(self):
        """Fonction principale pour charger l'image, trouver les contours et afficher les résultats."""
        # Répéter la texture et le masque dans une grille 3x3
        repeated_texture, repeated_mask = self.repeat_texture()

        # Trouver les contours uniques dans l'image répétée
        #unique_contours = self.find_and_merge_contours(repeated_mask)

        # Afficher l'image avec les contours uniques
       
        contours_de_mort,_ = cv2.findContours(repeated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.display_unique_contours(repeated_mask, contours_de_mort)

        return contours_de_mort


    def find_centered_window(self, repeated_mask, center_tile=(1, 1), tile_size=1):
        """Créer une fenêtre centrée de 2x2 tuiles."""

        milieux = repeated_mask.shape[0] // 2, repeated_mask.shape[1] // 2
        print(milieux , "milieux")
        x_start = (milieux[0] - self.width) 
        y_start = (milieux[1] - self.height) 
        x_end = (milieux[0] + self.width) 
        y_end = (milieux[1] + self.height) 

        # Découper la fenêtre du masque répété
        mask_window = repeated_mask[y_start:y_end, x_start:x_end]

        return mask_window, (x_start, y_start, x_end, y_end)

    def filter_contours_in_window(self, contours, mask_window, window_coords):
        """Filtrer et garder uniquement les contours à l'intérieur de la fenêtre."""
        filtered_contours = []
        x_start, y_start, x_end, y_end = window_coords
        height, width = mask_window.shape

        for contour in contours:
            # Vérifier si tous les points du contour sont à l'intérieur de la fenêtre
            inside_window = True
            for point in contour:
                pt = tuple(point[0])
                if not (x_start <= pt[0] < x_end and y_start <= pt[1] < y_end):
                    inside_window = False
                    break
            
            if inside_window:
                filtered_contours.append(contour)

        return filtered_contours

    def display_filtered_contours(self, repeated_mask, filtered_contours):
        """Afficher les contours filtrés dans la fenêtre."""
        # Créer une image pour afficher les résultats
        result_image = cv2.cvtColor(repeated_mask, cv2.COLOR_GRAY2BGR)  # Convertir en image colorée

        # Dessiner les contours filtrés
        for contour in filtered_contours:
            cv2.drawContours(result_image, [contour], -1, (0, 255, 0), 2)

        # Afficher l'image avec les contours filtrés
        cv2.imshow("Contours dans la fenêtre centrée", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process_centered_window(self):
       
        #la giga fenetre
        contours_de_mort = self.load_and_process()

        repeated_texture, repeated_mask = self.repeat_texture()

        
        mask_window, window_coords = self.find_centered_window(repeated_mask)
        
        print("coord de morts egalement", window_coords)
        # Étape 4 : Filtrer les contours dans la fenêtre
        filtered_contours = self.filter_contours_in_window(contours_de_mort, mask_window, window_coords)

        # Étape 5 : Afficher les contours filtrés
        self.display_filtered_contours(repeated_mask, filtered_contours)
