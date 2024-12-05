import cv2
import numpy as np

# Charger le masque de texture
masque = cv2.imread("Textures/texture3_masque.png", cv2.IMREAD_GRAYSCALE)

# Détecter les contours des pierres
contours, _ = cv2.findContours(masque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Créer une image en couleur pour afficher les pierres colorées
colored_image = cv2.cvtColor(masque, cv2.COLOR_GRAY2BGR)

# Générer des couleurs aléatoires pour chaque pierre
colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(len(contours))]

# Colorer chaque pierre avec une couleur différente
for i, contour in enumerate(contours):
    cv2.drawContours(colored_image, [contour], -1, colors[i], thickness=cv2.FILLED)

# Afficher l'image colorée
cv2.imshow("Colored Stones", colored_image)
cv2.waitKey(0)
cv2.destroyAllWindows()