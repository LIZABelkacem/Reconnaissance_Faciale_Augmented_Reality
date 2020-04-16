UTILISATION :

OBJ VIEWER :

aller dans le dossier 'src' ouvrir un terminal et lancer la commande: 

./0-objViiewer.sh


le programe devrait afficher le premier model, appuyez sur 'esc' pour fermer
il devrait ensuite afficher le deuxième model, appuyez sur 'esc' pour fermer



AR MAIN :

aller dans le dossier 'reference', copier une image pour la pointer a la caméra
à l'aide d'un smartphone, ou l'imprimer directement.

de preference pointer d'abord la reference devant la caméra 

aller dans le dossier 'src', ouvrir un terminal et lancer la commande: 

./1-arMain.sh

pour afficher un autre .obj modifier la ligne 39 du fichier ar_main.py en choisissant 
le fichier .obj désiré (qui se trouve dans le dossier 'model')



REMARQUE :

- assurez-vous d'installer toutes les librairies importées pour le bon fonctionnement
du programme

- si le fichier .sh ne s'exécute pas, pensez à modifier les droit d'accées grace à la commande

chmod +x nomDuFichier.sh
