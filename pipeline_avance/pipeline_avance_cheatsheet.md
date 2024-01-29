# Pipeline Avancé CheatSheet

# Tips de Débugging
- `pipeline.set_output(transform='pandas')` force la sortie du pipeline (sans estimateur) sous forme de DataFrame.
- Les pipelines issus de `make_pipeline` sont similaires aux listes et peuvent donc être **indexés**, **slicés** pour inspection.

Par ordre alphabétique de transformer

# `ColumnTransformer`
- `remainder='passthrough'` ne pas appliquer de traitement aux colonnes restantes.
- `verbose_feature_names_out=False` ne pas préfixer les noms des features par le nom du transformer appliqué.

# `OneHotEncoder`
- `drop='if_binary'` pour encoder une variable binaire.
- `drop='first'` pour encoder une variable nominale à $N$ modalités sur $N-1$ colonnes.
- `handle_unkown='ignore'` ne lève pas d'erreur sur une nouvelle modalité non rencontrée pendant un précédent entraînement. L'encode avec des zéros.
- `sparse_ouput=False` empêche la sortie d'être une *sparse matrix*, ce qui est utile en mode inspection/debug, quand vous souhaitez afficher des DataFrames en sortie.

# `OrdinalEncoder`
Exemple des tailles de vêtements.

```python
# Lister les modalités par ordre croissant
categories = [["XS", "S", "M", "L", "XL"]]  # ATTENTION: liste de listes!

ord_enc = OrdinalEncoder(categories = categories,
                         # Stratégie en cas de modalité inconnue dans le val ou le test set
                         handle_unknown = "use_encoded_value",
                         # Valeur assignée par défaut à ces modalités inconnues
                         unknown_value = -1)
```
