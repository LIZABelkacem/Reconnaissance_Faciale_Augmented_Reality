from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

# 	construire les argument pour les parser dans le terminal:
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,					#	chemin vers la dataset
	help="chemin vers la dataset")
ap.add_argument("-e", "--embeddings", required=True,				#	chemin vers les features du visage "embeddings"
	help="chemin vers les features du visage, embeddings")
ap.add_argument("-d", "--detector", required=True,					#	chemin vers le detecteur de visage de OpenCV
	help="chemin vers le detecteur de visage de OpenCV")
ap.add_argument("-m", "--embedding-model", required=True,			#	chemin vers le model deep learning en série de OpenCV
	help="chemin vers le model deep learning en série de OpenCV")
ap.add_argument("-c", "--confidence", type=float, default=0.5,		#	seuil pour filtrer et réduire les détections faibles
	help="seuil pour filtrer et réduire les détections faibles")
args = vars(ap.parse_args())

# 	charger le detecteur de visage en série du disque
print("[INFO] chargement du détecteur de visages...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# 	charger le model deep learning en série de OpenCV du disque
print("[INFO] chargement du reconnaisseur de visages...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# 	charger le chemin des images depuis le dataset
print("[INFO] quantification des visages...")
imagePaths = list(paths.list_images(args["dataset"]))

# 	initialisation des embeddings et noms reconnus
knownEmbeddings = []
knownNames = []

# 	initialisation du nombre total de visages traités
total = 0

for (i, imagePath) in enumerate(imagePaths):

	# 	extraction du nom de la personne depuis le chemin
	print("[INFO] traitement de l'image {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	#	chargement de l'image, mise à l'échelle
	#	et calcule de sa dimension
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=600)
	(h, w) = image.shape[:2]

	# 	création d'un spot depuis l'image pour faciliter le traitement
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(image, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# 	utilisation de model de deep learning de OpenCV
	#	pour la reconnaissance des visages
	detector.setInput(imageBlob)
	detections = detector.forward()

	# 	on vérifie qu'au moins un visage est détecté
	if len(detections) > 0:

		# 	on cherche le visage avec le plus grand cardre
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]

		#	on vérifie qu'on depasse bien seuil parsé au debut
		if confidence > args["confidence"]:

			#	calcule du (x, y) du cadre pour le visage
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			face = image[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			#	un petit test pour s'assurer que le cadre (visage) 
			#	est suffisament grand
			if fW < 20 or fH < 20:
				continue

			# 	création d'un spot pour le cadre cette fois
			# 	puis on le fais passer dans le model d'embedding
			#	pour obtenir les caracteristique du visage
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			#	ajout du nom de la personne + les caracteristique 
			# 	à leurs listes respectives
			knownNames.append(name)
			knownEmbeddings.append(vec.flatten())
			total += 1

# on enregistre les noms et les caracteristiques sur le disque
print("[INFO] enregistrement de {} caracteristiques...".format(total))
data = {"embeddings": knownEmbeddings, "names": knownNames}
f = open(args["embeddings"], "wb")
f.write(pickle.dumps(data))
f.close()