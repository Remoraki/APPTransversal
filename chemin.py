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
       
        contours,_ = cv2.findContours(repeated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.display_unique_contours(repeated_mask, contours)

        

    def calculate_neighbors(self, max_distance=10):
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

    def get_connected_components(self):
        """Obtient les composants connectés des cailloux."""
        visited = set()
        components = []

        def dfs(caillou_id, component):
            stack = [caillou_id]
            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    component.append(current)
                    stack.extend(self.cailloux[current]["neighbors"])

        for caillou_id in self.cailloux:
            if caillou_id not in visited:
                component = []
                dfs(caillou_id, component)
                components.append(component)

        return components

    def display_connected_components(self):
        """Affiche les composants connectés des cailloux."""
        # Répéter la texture et le masque dans une image 3x3
        repeated_texture, repeated_mask = self.repeat_texture()

        # Trouver les contours uniques dans l'image répétée
        unique_contours = self.find_and_merge_contours(repeated_mask)

        # Créer une image vide pour afficher les contours
        component_image = repeated_texture.copy()
        compona = cv2.cvtColor(component_image, cv2.COLOR_GRAY2BGR)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]

        # Obtenir les composants connectés
        components = self.get_connected_components()

        for idx, component in enumerate(components):
            color = colors[idx % len(colors)]
            for caillou_id in component:
                cv2.drawContours(compona, [self.cailloux[caillou_id]["contour"]], -1, (255,0,0), 2)
                print(f"Composant {idx}: Caillou {caillou_id}")

        # Afficher l'image avec les composants connectés
        cv2.imshow("Composants connectés des cailloux", compona)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



    def find_contours_in_centered_window(self, repeated_mask, center_tile=(1, 1), tile_size=2):
        """Trouver les contours dans une fenêtre centrée de taille 2 tuiles."""
        x_start = (center_tile[0] - tile_size) * self.block_width
        y_start = (center_tile[1] - tile_size) * self.block_height
        x_end = (center_tile[0] + tile_size + 1) * self.block_width
        y_end = (center_tile[1] + tile_size + 1) * self.block_height

        # Découper la fenêtre dans le masque répété
        mask_window = repeated_mask[y_start:y_end, x_start:x_end]

        # Trouver les contours dans la fenêtre découpée
        contours, _ = cv2.findContours(mask_window, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours, mask_window, (x_start, y_start)


    def filter_contours_within_window(self, contours, mask_window):
        """Conserver les contours qui se trouvent dans la fenêtre centrée de taille 2 tuiles."""
        filtered_contours = []
        height, width = mask_window.shape

        for contour in contours:
            # Vérifier si tous les points du contour sont à l'intérieur de la fenêtre
            if np.all((0 <= contour[:, 0, 0]) & (contour[:, 0, 0] < width) &
                    (0 <= contour[:, 0, 1]) & (contour[:, 0, 1] < height)):
                filtered_contours.append(contour)

        return filtered_contours


    def create_neighbors_graph(self, contours):
        """Créer un graphe des voisins basé sur les contours."""
        cailloux_graph = {}
        for i, contour_a in enumerate(contours):
            cailloux_graph[i] = {"contour": contour_a, "neighbors": set()}

        # Ajouter des voisins à partir de la distance entre les cailloux
        for i, caillou_a in cailloux_graph.items():
            for j, caillou_b in cailloux_graph.items():
                if i == j:
                    continue
                # Calculer la distance entre les contours
                distance = cv2.pointPolygonTest(caillou_b["contour"], tuple(caillou_a["contour"][0][0]), True)
                if distance < 10:  # Distance seuil pour déterminer si les cailloux sont voisins
                    cailloux_graph[i]["neighbors"].add(j)
                    cailloux_graph[j]["neighbors"].add(i)

        return cailloux_graph


    def update_connectivity_on_duplicates(self, cailloux_graph):
        """Mettre à jour la connectivité des cailloux en supprimant les doublons."""
        visited = set()
        for caillou_id, caillou in list(cailloux_graph.items()):
            # Si le caillou a déjà été visité, le supprimer du graphe
            if caillou_id in visited:
                del cailloux_graph[caillou_id]
                continue

            visited.add(caillou_id)
            for neighbor_id in caillou["neighbors"]:
                # Si un voisin existe qui est un doublon de taille, fusionner les deux
                if len(cailloux_graph[caillou_id]["contour"]) == len(cailloux_graph[neighbor_id]["contour"]):
                    # Fusionner les contours
                    cailloux_graph[caillou_id]["neighbors"].update(cailloux_graph[neighbor_id]["neighbors"])
                    # Supprimer le caillou dupliqué
                    del cailloux_graph[neighbor_id]

        return cailloux_graph


    def process_cailloux_in_window(self):
        """Traite les cailloux dans une fenêtre centrée avec un rayon défini, en mettant à jour le graphe des voisins et la connectivité."""
        # Étape 1 : Répéter la texture et le masque
        repeated_texture, repeated_mask = self.repeat_texture()

        # Étape 2 : Trouver les contours dans la fenêtre centrée de taille 2 tuiles
        contours, mask_window, offset = self.find_contours_in_centered_window(repeated_mask)

        # Étape 3 : Filtrer les contours qui sont dans la fenêtre
        filtered_contours = self.filter_contours_within_window(contours, mask_window)

        # Étape 4 : Créer un graphe des voisins
        cailloux_graph = self.create_neighbors_graph(filtered_contours)

        # Étape 5 : Mettre à jour la connectivité des cailloux en supprimant les doublons
        updated_cailloux_graph = self.update_connectivity_on_duplicates(cailloux_graph)

        # Afficher les contours restants dans la fenêtre
        result_image = repeated_mask.copy()
        RESULT_rgb = cv2.cvtColor(result_image, cv2.COLOR_GRAY2BGR)
        for contour in filtered_contours:
            cv2.drawContours(RESULT_rgb, [contour + np.array(offset)], -1, (255, 255, 0), 3)

        # Dessiner la grille sur l'image
        self.draw_grid(RESULT_rgb)

        # Afficher l'image finale et le graphe
        cv2.imshow("Contours et Graphe des voisins", RESULT_rgb)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Afficher le mapping dans la console
        print("Mapping des cailloux restants :")
        for caillou_id, caillou in updated_cailloux_graph.items():
            print(f"Caillou {caillou_id}: Contour = {caillou['contour']}, Neighbors = {caillou['neighbors']}")

        return updated_cailloux_graph
