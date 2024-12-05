import cv2
import sys
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