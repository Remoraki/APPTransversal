import cv2
import numpy as np
import sys

class Chemin:
    def __init__(self, width, height,nb_blocksx,nb_blocksy, texture_path="Textures/texture7"):
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

        # Obtenir les dimensions de la texture
        self.texture_height, self.texture_width = self.chemin.shape[:2]

    def draw_grid(self, screen):
        """Dessine une grille avec un nombre défini de blocs en x et y."""
        color = (255, 255, 255)  # Blanc

        n_blocks_x = self.nb_blocksx
        n_blocks_y = self.nb_blocksy


        # Lignes horizontales
        for i in range(n_blocks_y + 1):
            y = i * self.block_height
            cv2.line(screen, (0, y), (self.width, y), color, 1)

        # Lignes verticales
        for j in range(n_blocks_x + 1):
            x = j * self.block_width
            cv2.line(screen, (x, 0), (x, self.height), color, 1)

        

    def apply_texture(self, screen, block_x, block_y):
        """Applique la texture du chemin sur un bloc spécifique."""
        # Extraire la portion de la texture correspondant au bloc
        texture_roi = cv2.resize(self.chemin, (self.block_width, self.block_height))

        # Déterminer les coordonnées du bloc à appliquer
        x_start = block_x
        y_start = block_y
        x_end = x_start + self.block_width
        y_end = y_start + self.block_height

        if x_end <= screen.shape[1] and y_end <= screen.shape[0]:
            # Appliquer la texture directement sur le bloc
            roi = screen[y_start:y_end, x_start:x_end]
            np.copyto(roi, texture_roi)

    def apply_texture_to_grid(self, screen, points):
        """Applique la texture du chemin sur la grille traversée par le chemin."""
        # Calculer la taille des blocs de la grille
        block_width = self.width // self.block_width
        block_height = self.height // self.block_height

        # Pour chaque point du chemin, appliquer la texture sur le bloc correspondant
        for elem_x, elem_y in points:
            # Vérifier que les coordonnées sont dans les limites de l'écran
            if 0 <= elem_x < self.width and 0 <= elem_y < self.height:
                # Trouver le bloc traversé
                block_x = (elem_x // block_width) * block_width
                block_y = (elem_y // block_height) * block_height

                # Appliquer la texture au bloc
                self.apply_texture(screen, block_x, block_y)
