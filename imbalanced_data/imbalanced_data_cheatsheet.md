# Imbalanced Data CheatSheet

# Métriques
## Diagnostiquer/Comparer des Modèles de Classification Déséquilibrée
- **Diagnostiquer** un modèle: **Courbe ROC**
- **Comparer** des modèles: **$F_1$ score** ou **AUC**

## Gérer le Compromis Precision/Recall
- Courbe precision-recall
- $F$-scores
- $F$-scores ajustés $F_{\beta}$
    - $\beta$ s'articule autour de 1,
    - Plus $\beta$ est inférieur à 1, plus on favorise la **precision**,
    - Plus $\beta$ est supérieur à 1, plus on favorise le **recall**.
    Des valeurs classiques sont $F_0$, $F_1$, $F_2$.

# Stratification
Avoir des *train set* et *validation set* aussi représentatifs que possible du *train set*, dans un *hold-out* ou une *cross-validation*.

## Hold-Out
### Stratification Simple
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = 0.2,
    random_state = 0,
    stratify = X['feature']  # ou y
)
```
### Stratification Multiple (features catégorielles)
```python
# Création d'une nouvelle colonne juxtaposant les modalités
X['feat1_feat2'] = X.feat1 + "_" + X.feat2
# Hold-out
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = 0.2,
    random_state = 0,
    stratify = X['feat1_feat2']
)
# Nettoyage de la nouvelle colonne
X_train = X_train.drop(columns="feat1_feat2")
X_test = X_test.drop(columns="feat1_feat2")
```
## Cross-Validation
### Stratification (target)
```python
from sklearn.model_selection import cross_validate, StratifiedKFold
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
cv_results = cross_validate(model,
                            X, y,
                            # Passer à cv l'objet kfold
                            cv = kfold,
                            n_jobs = -1)  # Parallélisation
```
### Stratification (feature)
```python
# Variante de ce qui précède
cv_results = cross_validate(model,
                            X, y,
                            # Passer en second paramètre le nom de la feature
                            cv = kfold.split(X, X['feature']),
                            n_jobs = -1)  # Parallélisation
```
# Pondération
## `compute_class_weight`
```python
from sklearn.utils.class_weight import compute_class_weight
# Définir les classes et leur affecter des poids
classes = y_train.unique()
class_weights = compute_class_weight('balanced',
                                     classes = classes,
                                     y = y_train)
# Utiliser ces poids dans l'instanciation du classifier
model = LogisticRegression(class_weight = {k: class_weights[k]
                                            for k in classes})
model.fit(X_train, y_train)
```
## `compute_sample_weight`
```python
from sklearn.utils.class_weight import compute_sample_weight
# Calculer les poids des observations
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
# Utiliser les poids lors du fit
model = RandomForestClassifier(random_state=0)
model.fit(X_train, y_train, sample_weight=sample_weights)
```

# Echantillonnage
## Oversampling
```python
from imblearn.over_sampling import RandomOverSampler
# Instanciation et application d'une stratégie
oversampler = RandomOverSampler(sampling_strategy='minority', random_state=0)
# OU variante en renseignant le ratio souhaité après resampling
oversampler = RandomOverSampler(sampling_strategy=0.5, random_state=0)
# fit et resampling
X_train_os, y_train_os = oversampler.fit_resample(X_train, y_train)
```
## Undersampler
Fonctionnement similaire
## S.M.O.T.E.
### Implémentation
```python
from imblearn.over_sampling import SMOTE
oversampler = SMOTE(random_state=0)
X_train_smote, y_train_smote = oversampler.fit_resample(X_train, y_train)
```

