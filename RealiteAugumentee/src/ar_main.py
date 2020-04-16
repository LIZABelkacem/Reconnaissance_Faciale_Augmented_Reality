import argparse
import sys, pygame
sys.path.append('../include/')
import cv2
import numpy as np
import math
import os
from objloader_simple import *

#   le nombre min de points de ressemblance
#   pour valider la reconnaissance du motif
MIN_MATCHES = 10  


def main():

    #   initialisation de la matrice d'homographie
    homography = None 

    #   matrice de calibration de la caméra, les webcams ont une matrice trés semblable en générale
    camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
    
    #   création d'un detecteur de points ORB
    orb = cv2.ORB_create()
    
    #   création d'un matcheur BF (Brut Force) basé sur la distance de hamming  
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    #   chargement de la réference image à matcher sur le flux vidéo
    dir_name = os.getcwd()
    model = cv2.imread(os.path.join(dir_name, '../reference/1.jpg'), 0)        #   remplacer '1.jpg' par la réference désiréé
    
    #   calcule du point clé du model et de ses descripteurs
    kp_model, des_model = orb.detectAndCompute(model, None)
    
    #   chargement du fichier .obj à afficher sur le flux vidéo
    obj = OBJ(os.path.join(dir_name, '../models/ghilas.obj'), swapyz=True)     #   remplacer 'ghilas.obj' par 'liza.obj' ou l'obj désiréé
    
    # initialisation de la capture vidéo
    cap = cv2.VideoCapture(0)
    cap.set(3, 700) # largeur de la fenetre vidéo
    cap.set(4, 700) # hauteur de la fenetre vidéo

    while True:
        ret, frame = cap.read()
        if not ret:
            print ("[INFO] impossible de capturer la vidéo")
            return 

        #   trouve et affiche les points clés sur la frame
        kp_frame, des_frame = orb.detectAndCompute(frame, None)

        #   matcher les descripteurs de la frame avec ceux de la réf
        matches = bf.match(des_model, des_frame)
        
        #   les ordonner selons leurs distance, 
        #   plus la distance est petite, mieux est le match
        matches = sorted(matches, key=lambda x: x.distance)

        #   calcule la matrice d'homographie si il y'a suffisament de matchs
        if len(matches) > MIN_MATCHES:

            #   differencier entre les points de la source et ceux de la destination
            src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            #   calcule de la matrice d'homographie
            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if args.rectangle:

                #   affiche un rectangle autour du model matché sur la frame
                h, w = model.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                
                #   projection des coins vers la frame
                dst = cv2.perspectiveTransform(pts, homography)
                
                #   connection avec les lignes  
                frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)  
            
            #   si une homographie valide est trouvée, faire le rendu du model
            if homography is not None:
                try:

                    #   obtenir une matrice de projection 3D depuis la matrice 
                    #   d'homographie et les parameters de la caméra
                    projection = projection_matrix(camera_parameters, homography)  
                    
                    # projection du model
                    frame = render(frame, obj, projection, model, False)
                except:
                    pass
            
            #   afficher les N premièrs matches.
            if args.matches:
                frame = cv2.drawMatches(model, kp_model, frame, kp_frame, matches[:10], 0, flags=2)


        else:
            print ("[INFO] pas suffisament de matchs trouvés: %d/%d" % (len(matches), MIN_MATCHES))


        #   afficahge du résultat
        frame = cv2.flip( frame, 1)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0

#   fonction qui fait le rendu d'un model .obj dans la frame actuelle:
def render(img, obj, projection, model, color=False):

    vertices = obj.vertices
    scale_matrix = np.eye(3) * -250
    h, w = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)

        # on décalle le model pour qu'il soit affiché au milieu du cadre et non pas a coté
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:

            #   choix de la couleur du model en (B, G, R)
            cv2.fillConvexPoly(img, imgpts, (55, 57, 251))
    return img

#   fonction qui calcule la matrice de projection 3D à partir 
#   de la martice de calibration de la caméra et de l'homographie estimée
def projection_matrix(camera_parameters, homography):

    #   calcule de la rotation selon l'axe des x, y ainsi que la translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]

    #   normalisation des vecteurs
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l

    #   calcule de la base orthonormale
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)

    #   enfin, le calcule de la matrice de projection 3D
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)


#   construire les argument pour les parser dans le terminal:
parser = argparse.ArgumentParser()

parser.add_argument('-r','--rectangle', help = 'affiche un cadre autour de la réf sur la frame', action = 'store_true')
parser.add_argument('-mk','--model_keypoints', help = 'afficher les points clés sur la réf', action = 'store_true')
parser.add_argument('-fk','--frame_keypoints', help = 'afficher les points clés sur la frame', action = 'store_true')
parser.add_argument('-ma','--matches', help = 'afficher le match entre la réf et la frame', action = 'store_true')

args = parser.parse_args()

if __name__ == '__main__':
    main()