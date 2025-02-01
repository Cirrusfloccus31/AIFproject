# AIFproject

Ce projet utilise Docker Compose pour gérer les conteneurs. Les étapes pour l'exécuter sont explicitées plus bas.

## Prérequis

Avant de commencer, assurez-vous que **Docker** et **Docker Compose** sont installés et que Docker est en cours d'exécution sur votre machine.

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

3. **Arrêtez et supprimer les conteneurs**
    ```sh
    docker-compose down 

    