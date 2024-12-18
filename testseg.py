import cv2
import numpy as np

def calcul_qualite(m_ref, clustered_image):
    TP = FP = TN = FN = 0
    TP_2 = FP_2 = TN_2 = FN_2 = 0

    for i in range(m_ref.shape[0]):
        for j in range(m_ref.shape[1]):
            if clustered_image[i, j] == 255 and m_ref[i, j] == 255:
                TP += 1
                FP_2 += 1
            elif clustered_image[i, j] == 255 and m_ref[i, j] == 0:
                FP += 1
                TP_2 += 1
            elif clustered_image[i, j] == 0 and m_ref[i, j] == 255:
                FN += 1
                TN_2 += 1
            else:
                TN += 1
                FN_2 += 1

    DSC1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
    DSC2 = 2 * TP_2 / (2 * TP_2 + FP_2 + FN_2) if (2 * TP_2 + FP_2 + FN_2) > 0 else 0

    if DSC1 > DSC2:
        DSC = DSC1
        P = TP / (TP + FP) if (TP + FP) > 0 else 0
        S = TP / (TP + FN) if (TP + FN) > 0 else 0
    else:
        DSC = DSC2
        P = TP_2 / (TP_2 + FP_2) if (TP_2 + FP_2) > 0 else 0
        S = TP_2 / (TP_2 + FN_2) if (TP_2 + FN_2) > 0 else 0

    print(f"P = {P}")
    print(f"S = {S}")
    print(f"DSC = {DSC}\n")


m_ref = cv2.imread("Textures/texture8_masque.png", cv2.IMREAD_GRAYSCALE)
clustered_image = cv2.imread("seg/full_segmentation8.png", cv2.IMREAD_GRAYSCALE)

calcul_qualite(m_ref, clustered_image)