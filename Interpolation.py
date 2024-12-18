import numpy as np
from scipy.interpolate import BSpline, make_interp_spline
import matplotlib.pyplot as plt
import cv2
from matplotlib.image import imread
from scipy.ndimage import binary_erosion, binary_dilation
from scipy import signal
from skimage.morphology import disk

def interpolate(selected_x: np.ndarray, selected_y: np.ndarray, nbInterpolatedPoints: int) -> np.ndarray:

    xx = np.linspace(selected_x[0], selected_x[-1], nbInterpolatedPoints)
    bspl = make_interp_spline(selected_x, selected_y, k=2)

    return np.round(xx).astype(int), np.round(bspl(xx)).astype(int)

def get_mask(x_interpolated,y_interpolated,width,shape):

    mask = np.zeros(shape)

    to_delete = mask.copy()

    circle = disk(width)

    for i in range(x_interpolated.shape[0]):
    
        x = x_interpolated[i]
        y = y_interpolated[i]

        p = 2
        to_delete[y-p:y+p+1,x-p:x+p+1] = 1

        y_min = max(0, y - width)
        y_max = min(shape[0], y + width + 1)
        x_min = max(0, x - width)
        x_max = min(shape[1], x + width + 1)
        
        cropped_circle = circle[:y_max - y_min, :x_max - x_min]
        
        mask[y_min:y_max, x_min:x_max] = np.maximum(mask[y_min:y_max, x_min:x_max], cropped_circle)
        

    plt.imshow(to_delete,cmap='gray')
    plt.show()

    return mask

def get_boundaries(mask):


    kernel = np.array([[ -1, 0, +1],
                        [-1, 0, +1],
                        [-1, 0, +1]])
    col_derivative = signal.convolve2d(mask, kernel, boundary='symm', mode='same')

    row_derivative = signal.convolve2d(mask, kernel.T, boundary='symm', mode='same')

    boundaries = np.abs(col_derivative + row_derivative) > 0

    return boundaries.astype(int)

def translate_matrix(matrix, vector):
    # Déterminer les dimensions de la matrice
    rows, cols = matrix.shape

    # Initialiser une matrice remplie de zéros de même taille
    translated = np.zeros_like(matrix)

    # Déterminer le décalage
    dx, dy = vector

    # Indices sources et cibles après translation
    src_x = slice(max(-dx, 0), rows - max(dx, 0))
    src_y = slice(max(-dy, 0), cols - max(dy, 0))

    tgt_x = slice(max(dx, 0), rows - max(-dx, 0))
    tgt_y = slice(max(dy, 0), cols - max(-dy, 0))

    # Effectuer la translation
    translated[tgt_x, tgt_y] = matrix[src_x, src_y]

    return translated

def clip_boundaries(boundaries,path):
    
    # Pour supprimer les pixels noir du chemin sur la frontière de celui-ci
    boundaries = boundaries * path

    non_nul_indices = np.argwhere(boundaries)

    coord_i = non_nul_indices[:,0]
    coord_j = non_nul_indices[:,1]

    print(len(coord_i))

    for k in range(len(coord_i)):
        print(str(k)+'/'+str(len(coord_i)))
        path = clip_auxiliaire(path,coord_i[k],coord_j[k])

    return path

# def clip_auxiliaire(path,coord_i,coord_j):

#     neigh_size = 1

#     if np.array_equal(path[coord_i-neigh_size:coord_i+neigh_size+1,coord_j-neigh_size:coord_j+neigh_size+1],np.zeros((3,3))):
#         return path
#     else :
#         path[coord_i-neigh_size:coord_i+neigh_size+1,coord_j-neigh_size:coord_j+neigh_size+1] = np.zeros((2*neigh_size+1,2*neigh_size+1))

#         plt.imshow(path[coord_i-20:coord_i+20+1,coord_j-20:coord_j+20+1],cmap='gray')
#         plt.title('new path with i='+str(coord_i)+' j='+str(coord_j))
#         plt.show()

#         return clip_auxiliaire(path,coord_i-1,coord_j-1) * clip_auxiliaire(path,coord_i-1,coord_j) * clip_auxiliaire(path,coord_i-1,coord_j+1) * clip_auxiliaire(path,coord_i,coord_j-1) * clip_auxiliaire(path,coord_i,coord_j) * clip_auxiliaire(path,coord_i,coord_j+1) * clip_auxiliaire(path,coord_i+1,coord_j-1) * clip_auxiliaire(path,coord_i+1,coord_j) * clip_auxiliaire(path,coord_i+1,coord_j+1)

def clip_auxiliaire(path, coord_i, coord_j, neigh_size=2):
    sub_path = path[
        max(0, coord_i - 1):min(path.shape[0], coord_i + 1 + 1),
        max(0, coord_j - 1):min(path.shape[1], coord_j + 1 + 1)
    ]

    if np.array_equal(sub_path, np.zeros_like(sub_path)):
        return path

    i_min = max(0, coord_i - neigh_size)
    i_max = min(path.shape[0], coord_i + neigh_size + 1)
    j_min = max(0, coord_j - neigh_size)
    j_max = min(path.shape[1], coord_j + neigh_size + 1)
    path[i_min:i_max, j_min:j_max] = 0

    plt.imshow(path[max(0, coord_i-20):min(path.shape[0], coord_i+20+1),
                    max(0, coord_j-20):min(path.shape[1], coord_j+20+1)],
               cmap='gray')
    plt.title(f'new path with i={coord_i}, j={coord_j}')
    plt.show()

    for di in [-neigh_size , neigh_size]: #range(-neigh_size, neigh_size + 1)
        for dj in [-neigh_size , neigh_size]: #range(-neigh_size, neigh_size + 1)
            ni, nj = coord_i + di, coord_j + dj
            if 0 <= ni < path.shape[0] and 0 <= nj < path.shape[1]:
                path = path * clip_auxiliaire(path, ni, nj, neigh_size)

    return path


# def clip_auxiliaire(path, coord_i, coord_j):
#     """
#     Recursively remove boundary components from the binary path matrix.
#     Each recursive call computes the intersection of results from neighboring cells.
#     """
#     # Check for out-of-bounds indices
#     if coord_i < 0 or coord_i >= path.shape[0] or coord_j < 0 or coord_j >= path.shape[1]:
#         return np.ones_like(path)  # Return a matrix of ones (identity for intersection)

#     # Check if the current pixel is already black
#     if path[coord_i, coord_j] == 0:
#         return np.ones_like(path)  # Return identity matrix (no change for intersection)

#     # Create a copy of the path to modify
#     result = np.copy(path)

#     neigh_size = 1

#     # Blacken the 3x3 neighborhood around the current pixel
#     i_min = max(0, coord_i - neigh_size)
#     i_max = min(path.shape[0], coord_i + neigh_size + 1)
#     j_min = max(0, coord_j - neigh_size)
#     j_max = min(path.shape[1], coord_j + neigh_size + 1)
#     result[i_min:i_max, j_min:j_max] = 0

#     # Compute the intersection of results from neighbors
#     neighborhood = list(range(-neigh_size,neigh_size+1))
#     for di in neighborhood:
#         for dj in neighborhood:
#             if di != 0 or dj != 0:  # Skip the center pixel

#                 plt.imshow(path[coord_i-20:coord_i+20+1,coord_j-20:coord_j+20+1],cmap='gray')
#                 plt.title('new path with i='+str(coord_i)+' j='+str(coord_j))
#                 plt.show()

#                 result = result * clip_auxiliaire(path, coord_i + di, coord_j + dj)

#     return result

# def clip_auxiliaire(path, coord_i, coord_j):
#     """
#     Recursively remove boundary components from the binary path matrix.
#     Each recursive call computes the intersection of results from neighboring cells.
#     """
#     # Check for out-of-bounds indices
#     if coord_i < 0 or coord_i >= path.shape[0] or coord_j < 0 or coord_j >= path.shape[1]:
#         return path  # Return a matrix of ones (identity for intersection)

#     # Check if the current pixel is already black
#     if path[coord_i, coord_j] == 0:
#         return path  # Return identity matrix (no change for intersection)

#     # Create a copy of the path to modify
#     # result = np.copy(path)
    

#     # Blacken the 3x3 neighborhood around the current pixel
#     i_min = max(0, coord_i - 1)
#     i_max = min(path.shape[0], coord_i + 1 + 1)
#     j_min = max(0, coord_j - 1)
#     j_max = min(path.shape[1], coord_j + 1 + 1)
#     path[i_min:i_max, j_min:j_max] = 0

#     # Compute the intersection of results from neighbors
#     for di in [-1, 0, 1]:
#         for dj in [-1, 0, 1]:
#             if di != 0 or dj != 0:  # Skip the center pixel

#                 plt.imshow(path[coord_i+di-20:coord_i+di+20+1,coord_j+dj-20:coord_j+dj+20+1],cmap='gray')
#                 plt.title('new path with i='+str(coord_i + di)+' j='+str(coord_j + dj))
#                 plt.show()

#                 path = path * clip_auxiliaire(path, coord_i + di, coord_j + dj)

#     return path


def click_event(event, selected_x, selected_y, flags, params): 

    if event == cv2.EVENT_LBUTTONDOWN: 

        print(selected_x,selected_y)
  
        x.append( selected_x )
        y.append( selected_y )

if __name__ == "__main__":

    """ Pour tester la récupération des coordonnées en lesquel l'utilisateur clique """

    # grass = imread('Textures/grass.png')
    # print("gg ",grass.shape)

    # plt.imshow(grass[:512,:512,:])
    # plt.show()

    x = []
    y = []

    nb_textures = 3

    num_texture = 'texture11'

    texture = imread('Textures/'+num_texture+'_masque.png')
    texture = texture[:,:,0]
    print(texture.shape)

    img = np.zeros((texture.shape[0]*nb_textures,texture.shape[1]*nb_textures))
    cv2.imshow('image', img)

    cv2.setMouseCallback('image', click_event) 

    cv2.waitKey(0) 

    print(x)
    print(y)

    """ Pour tester la fonction interpolate """

    x_array = np.array(x)
    y_array = np.array(y)

    print(x_array)
    print(y_array)
    
    x_interpolated,y_interpolated = interpolate(x_array,y_array,100)

    fig, ax = plt.subplots()

    ax.plot(x_interpolated, y_interpolated, label="Spline Interpolée")
    ax.scatter(x, y, color='red', label="Points d'origine")

    ax.legend()
    plt.show()

    # On construit la matrice constituée de la texture choisie et répétée

    big_texture = np.tile(texture, (nb_textures, nb_textures))

    plt.imshow(big_texture,cmap='gray')
    plt.show()

    # On construit, en partant de l'interpolation, le masque du chemin

    width = 130

    mask = get_mask(x_interpolated,y_interpolated,width=width,shape=big_texture.shape)

    plt.imshow(mask,cmap='gray')
    plt.show()

    path = mask * big_texture

    plt.imshow(path,cmap='gray')
    plt.show()

    boundaries = get_boundaries(mask)

    vectors = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    for vector in vectors:
        translated_boundarie = translate_matrix(boundaries, vector)
        boundaries = boundaries + translated_boundarie

    boundaries = boundaries > 0
    boundaries.astype(int)


    plt.imshow(boundaries,cmap='gray')
    plt.title('boundaries')
    plt.show()

    # new_path = clip_boundaries(boundaries,path)

    plt.imshow(path + boundaries,cmap='gray')
    plt.show()

    # Calcul des composantes connexes
    path = path.astype(np.uint8)
    analysis = cv2.connectedComponentsWithStats(path,4,cv2.CV_32S)
    (totalLabels, label_ids, values, centroid) = analysis

    print(centroid.shape)
    print(label_ids.shape)
    print(values.shape)
    print(totalLabels)

    components_on_boundarie = np.zeros(path.shape, dtype="uint8")

    l = []

    for i in range(1,centroid.shape[0]):

        print(i,"/",centroid.shape[0])

        connected_component = (label_ids == i).astype("uint8")
        # area = values[i, cv2.CC_STAT_AREA]
        
        connected_component = connected_component.astype(np.uint8)
        boundaries = boundaries.astype(np.uint8)

        # component_inter_path = cv2.bitwise_and(connected_component, boundaries)
        
        # pixels_on_boundarie = np.sum(component_inter_path)
        # pixels_on_connected_component = np.sum(connected_component)
        
        # is_on_boundarie = ( pixels_on_boundarie > 0 )

        if True:

            # percent = pixels_on_boundarie/pixels_on_connected_component

            x = np.linspace(0, path.shape[1], path.shape[1])
            y = np.linspace(0, path.shape[0], path.shape[0])
            X, Y = np.meshgrid(x, y)

            distances_to_centroid = np.sqrt((centroid[i,0] - X)**2 + (centroid[i,1] - Y)**2)

            distances_to_centroid = np.round(distances_to_centroid,3)

            distances_to_centroid = distances_to_centroid * boundaries + 1e8 * ( 1.0 - boundaries )

            min_distance_to_centroid = np.min(distances_to_centroid)

            epsilon = 40.0

            if min_distance_to_centroid < epsilon:

                # l.append(min_distance_to_centroid)
                # print(l)

                # plt.imshow(boundaries + connected_component,cmap='gray')
                # plt.title('component and boundarie, min_distance_to_centroid = '+str(min_distance_to_centroid))
                # plt.show()

                components_on_boundarie = cv2.bitwise_or(components_on_boundarie, connected_component)

                # plt.imshow(components_on_boundarie,cmap='gray')
                # plt.title('component and boundarie')
                # plt.show()

    plt.imshow(components_on_boundarie,cmap='gray')
    plt.title('components on boundarie')
    plt.show()

    cliped_path = path * (1 - components_on_boundarie)

    plt.imshow(cliped_path,cmap='gray')
    plt.title('components on boundarie')
    plt.show()

    texture_colored = imread('Textures/'+num_texture+'.png')
    big_texture_colored = np.tile(texture_colored, (nb_textures, nb_textures, 1))

    grass = imread('Textures/grass.png')
    grass = grass[:512,:512,:]
    grass = np.tile(grass, (nb_textures, nb_textures, 1))
    # grass = grass[:512*nb_textures, :512*nb_textures, :]  # Redimensionner si nécessaire

    # S'assurer que grass a 3 canaux RGB
    grass_rgb = grass[:, :, :3]  # Conserver uniquement les trois canaux (R, G, B)

    # Étendre cliped_path pour les 3 canaux
    cliped_path = np.expand_dims(cliped_path, axis=2)  # (H, W, 1)
    cliped_path = np.repeat(cliped_path, 3, axis=2)    # (H, W, 3)

    # Combinaison des textures avec clipping des valeurs
    cliped_path_colored = big_texture_colored[:, :, :3] * cliped_path + grass_rgb * (1 - cliped_path)

    plt.imshow(cliped_path_colored)
    plt.title('new_path')
    plt.show()

