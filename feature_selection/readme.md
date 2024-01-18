
# Feature selection

L'objectif de la feature selection est de choisir les features les plus importantes ou les plus informatives pour construire un modèle prédictif. En éliminant les features moins pertinentes, on peut simplifier le modèle, améliorer sa performance, réduire le temps d'apprentisage (fit) et éviter l'overfitting.


En résumé, la feature selection consiste à choisir judicieusement les features les plus utiles pour construire un modèle prédictif, en éliminant celles qui peuvent être redondantes, moins informatives ou qui ajoutent du bruit au modèle.

Docs:
- [sklearn](https://scikit-learn.org/stable/modules/feature_selection.html)
- [Machine Learnia](https://www.youtube.com/watch?v=T4nZDuakYlU)

## Les différentes méthodes

-  **La régularisation**

	-  *L1 (Lasso)* il va faire tendre les paramètres vers 0
	-  *L2 (Ridge)* rend les valeurs des paramètres plus homogènes et moins grands.

```from sklearn.linear_model import Lasso, Ridge```

La régularisation est particulièrement utile lorsque le nombre de caractéristiques est élevé par rapport à la taille de l'ensemble de données, car elle peut aider à éviter l'overfitting dans de telles situations.

  

-  **Filtrage des caractéristiques (Filter Methods)**
	- *VarianceTreshold* cette méthode s'applique sur la variance de chaque features. Il supprimera les features qui ne varient pas ou très peu en fonction d'un seuil.  
  ```from sklearn.feature_selection import VarianceThreshold```
	- *SelectKBest* L'objectif principal de `SelectKBest` est de sélectionner les "k" meilleures features d'un ensemble de données en utilisant divers tests statistiques. Elle est souvent utilisée pour effectuer une première étape de prétraitement des données en éliminant les features moins importantes et en se concentrant sur un sous-ensemble plus restreint de features.  
  ```from sklearn.feature_selection import SelectKBest```

-  **Emballage des caractéristiques (Wrapper Methods)**