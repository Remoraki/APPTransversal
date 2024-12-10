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

    def load_segmented_image(self):
        """Charge l'image segmentée et affiche les contours des cailloux."""
        # Trouver les contours des cailloux
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Associer un ID unique à chaque contour
        self.cailloux = {}
        height, width = self.mask.shape

        for i, contour in enumerate(contours):
            # Vérifier si le caillou touche les bords
            touches_left = np.any(contour[:, 0, 0] == 0)
            touches_right = np.any(contour[:, 0, 0] == width - 1)
            touches_top = np.any(contour[:, 0, 1] == 0)
            touches_bottom = np.any(contour[:, 0, 1] == height - 1)

            # Si le caillou ne touche pas les bords, c'est une entité complète
            if not (touches_left or touches_right or touches_top or touches_bottom):
                self.cailloux[i] = {
                    "contour": contour,
                    "neighbors": set(),
                    "inside_path": False,  # Statut initial
                }
                continue

            # Reconstituer les cailloux qui touchent les bords
            new_contour = contour.copy()
            if touches_left:
                new_contour = np.vstack((new_contour, contour[contour[:, 0, 0] == width - 1]))
            if touches_right:
                new_contour = np.vstack((new_contour, contour[contour[:, 0, 0] == 0]))
            if touches_top:
                new_contour = np.vstack((new_contour, contour[contour[:, 0, 1] == height - 1]))
            if touches_bottom:
                new_contour = np.vstack((new_contour, contour[contour[:, 0, 1] == 0]))

            self.cailloux[i] = {
                "contour": new_contour,
                "neighbors": set(),
                "inside_path": False,  # Statut initial
            }

        # Créer une image pour afficher les contours
        contour_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Dessiner les contours sur l'image
        for i, caillou in self.cailloux.items():
            cv2.drawContours(contour_image, [caillou["contour"]], -1, (0, 255, 0), 2)
            print(f"Caillou {i}: {caillou['contour']}")

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

    def display_neighbors(self):
        """Affiche les voisins de chaque caillou."""
        neighbor_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        for i, caillou in self.cailloux.items():
            # Dessiner le contour du caillou
            cv2.drawContours(neighbor_image, [caillou["contour"]], -1, (0, 255, 0), 2)
            # Dessiner les lignes vers les voisins
            for neighbor in caillou["neighbors"]:
                neighbor_contour = self.cailloux[neighbor]["contour"]
                pt1 = tuple(caillou["contour"][0][0])
                pt2 = tuple(neighbor_contour[0][0])
                cv2.line(neighbor_image, pt1, pt2, (255, 0, 0), 1)
                print(f"Caillou {i} est voisin avec Caillou {neighbor}")

        # Afficher l'image avec les voisins
        cv2.imshow("Voisins des cailloux", neighbor_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()