# AIFproject

Ce dépôt est dédié au projet du cours AIF à l'INSA Toulouse 2024-2025.

Ce projet utilise Docker Compose pour gérer les conteneurs. Les étapes pour l'exécuter sont explicitées plus bas.

## Prérequis

Avant de commencer, assurez-vous que **Docker** et **Docker Compose** sont installés et que Docker est en cours d'exécution sur votre machine.

## Installation et exécution

Ouvrez un terminal et exécutez les commandes suivantes :

1. **Récupération du projet**
    ```sh
    git clone https://github.com/Cirrusfloccus31/AIFproject.git 

3. **Se placer dans le répertoire du projet**
    ```sh
    cd AIFproject 

4. **Construire les images Docker**  
    ```sh
    docker-compose build

5. **Démarrez les conteneurs** 
    ```sh 
    docker-compose up 

Si des problèmes de permission apparaissent utilisez **sudo**.

5. **Utiliser l'application**

Aller à l'adresse [localhost:7860](http://localhost:7860/), vous arrivez alors sur la page de l'application. Les trois onglets correspondent respectivement aux parties 1, 2 et 3.

Vous pouvez utiliser l' image `a_movie_poster.jpg` contenue dans le dossier images pour tester les parties 1 et 2. L'image `not_a_movie_poster.jpg` est faite pour tester la partie 4 (même onglet que la partie 1). Pour tester la partie 3 vous pouvez utiliser le scénario suivant (celui de Toy Story) : Led by Woody, Andy's toys live happily in his room until Andy's birthday brings Buzz Lightyear onto the scene. Afraid of losing his place in Andy's heart, Woody plots against Buzz. But when circumstances separate Buzz and Woody from their owner, the duo eventually learns to put aside their differences. Les inférences pour la partie 1 (premier onglet) peuvent être un peu longues.

6. **Arrêter et supprimer les conteneurs quand vous avez fini d'utiliser l'application**
    ```sh
    docker-compose down 

## Contributeurs

Paul Lacotte, Ariadna Perelló-Achfari, Hugo Germain
    
