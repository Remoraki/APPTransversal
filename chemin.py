import cv2
import sys
import numpy as np

class Chemin():
    def __init__(self, texture_path="Textures/texture7"):
        # Charger l'image de texture
        self.chemin = cv2.imread(f"{texture_path}.png")
        if self.chemin is None:
            print("Erreur : impossible de charger l'image de texture.")
            sys.exit()

        # Obtenir les dimensions de la texture
        self.texture_height, self.texture_width = self.chemin.shape[:2]

    def apply_texture(self, screen, x, y):
        x_start = (x // self.texture_width) * self.texture_width
        y_start = (y // self.texture_height) * self.texture_height
        x_end = x_start + self.texture_width
        y_end = y_start + self.texture_height

        if x_end <= screen.shape[1] and y_end <= screen.shape[0]:
            # Appliquer la texture directement
            roi = screen[y_start:y_end, x_start:x_end]
            texture_roi = self.chemin[0:self.texture_height, 0:self.texture_width]
            np.copyto(roi, texture_roi)

