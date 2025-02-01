# AIFproject

Ce dépôt est dédié au projet du cours AIF à l'INSA Toulouse 2024-2025.

Ce projet utilise Docker Compose pour gérer les conteneurs. Les étapes pour l'exécuter sont explicitées plus bas.

## Prérequis

Avant de commencer, assurez-vous que **Docker** et **Docker Compose** sont installés et que Docker est en cours d'exécution sur votre machine.

Puis ouvrer un terminal et cloner le projet :
    ```sh
    git clone https://github.com/Cirrusfloccus31/AIFproject.git 

Enfin placer vous dans le répertoire du projet :
    ```sh
    cd AIFproject 

## Installation et exécution

1. **Construire les images Docker**  
Exécutez la commande suivante :  
    ```sh
    docker-compose build

2. **Démarrez les conteneurs** 
Lancez la commande suivante 
    ```sh 
    docker-compose up 

Si des problèmes de permission apparaissent utilisez **sudo**.

3. **Utiliser l'application**

Aller à l'adresse [localhost:7860](http://localhost:7860/), vous arrivez alors sur la page de l'application. Les trois onglets correspondent respectivement aux parties 1, 2 et 3.

Vous pouvez utiliser l' image `a_movie_poster.jpg` contenues dans le dossier images pour tester les parties 1 et 2. L'image `not_a_movie_poster.jpg` est faite pour tester la partie 4 (même onglet que la partie 1). Pour tester la partie 3 vous pouvez utiliser le scénario suivant (celui de Toy Story) : Led by Woody, Andy's toys live happily in his room until Andy's birthday brings Buzz Lightyear onto the scene. Afraid of losing his place in Andy's heart, Woody plots against Buzz. But when circumstances separate Buzz and Woody from their owner, the duo eventually learns to put aside their differences. Les inférences pour la partie 1 (premier onglet) peuvent être un peu longues.

4. **Arrêtez et supprimer les conteneurs**
    ```sh
    docker-compose down 

## Contributeurs

Paul Lacotte, Ariadna Perelló-Achfari, Hugo Germain
    