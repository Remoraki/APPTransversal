import cv2
import numpy as np
import sys

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

        # Charger le masque (image segmentée)
        self.mask = cv2.imread(f"{texture_path}_masque.png", cv2.IMREAD_GRAYSCALE)

        # Créer une grille de visibilité des blocs
        self.visible_blocks = np.zeros((nb_blocksy, nb_blocksx), dtype=bool)

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

        # Utiliser un ensemble pour garder seulement les contours uniques
        unique_contours = set()

        for contour in contours:
            # Convertir le contour en tuple pour pouvoir l'ajouter à un ensemble
            contour_tuple = tuple(map(tuple, contour[:, 0, :]))
            unique_contours.add(contour_tuple)

        return unique_contours

    def display_unique_contours(self, repeated_texture, unique_contours):
        """Afficher les contours uniques dans une fenêtre séparée."""
        # Créer une image vide pour afficher les contours
        contour_image = repeated_texture.copy()

        # Dessiner les contours sur l'image
        for contour in unique_contours:
            contour_np = np.array(contour, dtype=np.int32)
            cv2.drawContours(contour_image, [contour_np], -1, (0, 255, 0), 2)

        # Afficher l'image avec les contours uniques
        cv2.imshow("Contours Uniques", contour_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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

    def load_segmented_image(self):
        """Charge l'image segmentée et identifie les cailloux réellement coupés."""
        # Trouver les contours des cailloux
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Associer un ID unique à chaque contour
        self.cailloux = {}
        height, width = self.mask.shape

        for i, contour in enumerate(contours):
            # Vérifier si le caillou est réellement coupé entre deux bords opposés
            touches_left = np.any(contour[:, 0, 0] == 0)
            touches_right = np.any(contour[:, 0, 0] == width - 1)
            touches_top = np.any(contour[:, 0, 1] == 0)
            touches_bottom = np.any(contour[:, 0, 1] == height - 1)

            # Vérifier si le contour est coupé entre des bords opposés
            is_cut = (touches_left and touches_right) or (touches_top and touches_bottom)

            # Si le caillou est coupé, reconstruire son contour
            if is_cut:
                new_contour = contour.copy()
                if touches_left and touches_right:
                    new_contour = np.vstack((new_contour, contour[contour[:, 0, 0] == width - 1]))
                if touches_top and touches_bottom:
                    new_contour = np.vstack((new_contour, contour[contour[:, 0, 1] == height - 1]))
            else:
                new_contour = contour  # Contour intact

            self.cailloux[i] = {
                "contour": new_contour,
                "neighbors": set(),
                "inside_path": False,  # Statut initial
                "is_cut": is_cut,  # Nouveau statut pour les cailloux réellement coupés
            }

        # Créer une image pour afficher les contours
        contour_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Dessiner les contours sur l'image
        for i, caillou in self.cailloux.items():
            color = (0, 255, 0) if not caillou["is_cut"] else (0, 0, 255)  # Rouge pour les cailloux coupés
            cv2.drawContours(contour_image, [caillou["contour"]], -1, color, 2)
            print(f"Caillou {i}: IsCut={caillou['is_cut']}")

        # Afficher l'image avec les contours
        cv2.imshow("Contours des cailloux", contour_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def calculate_neighbors(self, max_distance=1):
        """Calcule les voisins de chaque caillou en fonction de la proximité."""
        for i, caillou_a in self.cailloux.items():
            for j, caillou_b in self.cailloux.items():
                if i == j:
                    continue

                pt = tuple(map(int, caillou_a["contour"][0][0]))
                # Vérifier si les contours sont proches
                distance = cv2.pointPolygonTest(caillou_b["contour"], pt, True)
                if distance < max_distance:
                    self.cailloux[i]["neighbors"].add(j)
                    self.cailloux[j]["neighbors"].add(i)

    def load_and_process(self):
        """Fonction principale pour charger l'image, trouver les contours et afficher les résultats."""
        # Répéter la texture et le masque dans une grille 3x3
        repeated_texture, repeated_mask = self.repeat_texture()

        # Trouver les contours uniques dans l'image répétée
        unique_contours = self.find_and_merge_contours(repeated_mask)

        # Afficher l'image avec les contours uniques
        self.display_unique_contours(repeated_texture, unique_contours)
