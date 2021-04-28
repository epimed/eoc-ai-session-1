 # Exercice 1 : Decision Trees et Random Forest


```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier
```


```python
def calculate_accuracy(classifier, X_train, X_test, y_train, y_test):
    classifier.fit(X_train, y_train)
    y_pred_train = classifier.predict(X_train)
    y_pred_test = classifier.predict(X_test)
    accuracy_train = metrics.accuracy_score(y_train, y_pred_train)
    accuracy_test = metrics.accuracy_score(y_test, y_pred_test)
    print('Train accuracy:', '{:.3f}'.format(accuracy_train), 'Test accuracy:', '{:.3f}'.format(accuracy_test))
    return accuracy_train, accuracy_test, classifier
```


```python
random_state = 42
data = pd.read_csv('../data/colon_cancer.csv', sep=';', index_col='id_sample')
y = data['tissue_status']
```

## Question 1. Construire la matrice **X** avec toutes les variables disponibles dans le dataset

Indice : vous pouvez utiliser la méthode `select_dtypes('number')` ou la méthode `drop(columns=['tissue_status'])`


```python
# X = data... # à compléter
```

## Question 2. Créer un dataset d'entrainement de 3/4 et un dataset de test de 1/4


```python
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=..., random_state=random_state, stratify=y) # test_size à compléter
```

Quelles sont les dimensions des matrices X_train et X_test ? Combien d'échantillons contient chaque dataset ?

Indice : On peut utiliser l'attribut `shape` du dataframe.


```python
# print('Train:', X_train..., 'Test:', X_test...) # à compléter
```

## Question 3. Créer, entrainer et visualiser un arbre de décision (decision tree) de profondeur 2 pour la totalité des gènes


```python
classifier = DecisionTreeClassifier(max_depth=2, random_state=random_state, criterion='entropy')
accuracy_train, accuracy_test, trained_classifier = calculate_accuracy(classifier, X_train, X_test, y_train, y_test)
```


```python
# plot_tree(..., feature_names=...,  class_names=y.unique(), precision=2, filled=True) # options à compléter
```

Quels sont les meilleurs gènes prédicteurs indentifiés par l'algorithme ?

## Question 4. Calculer la matrice de confusion

La **matrice de confusion** est une matrice qui mesure la qualité d'un système de classification. Chaque ligne correspond à une classe réelle, chaque colonne correspond à une classe estimée. La cellule ligne L, colonne C contient le nombre d'éléments de la classe réelle L qui ont été estimés comme appartenant à la classe C. Un des intérêts de la matrice de confusion est qu'elle montre rapidement si un système de classification parvient à classifier correctement.


```python
metrics.plot_confusion_matrix(trained_classifier, X_test, y_test)  # à exécuter
```

Est-ce que le modèle prédit aussi bien les échantillons normaux que tumoraux ? 

## Question 5. Créer un modèle de Random Forest avec 20 arbres et estimer ses métriques

La méthode de Random Forest crée une série d'arbres de décsion (une forêt). Chaque arbre prend en compte une partie des features tirés au hasard. La résultat final est établi par un vote entre tous les arbres (la majorité gagne). Pour entrainer un modèle de Random Forest, utilisez la classe **RandomForestClassifier** de **scikit-learn**. La profondeur de chaque arbre est définie par l'option *max_depth*. Le nombre total d'arbres dans la forêt est controlé par l'option *n_estimators*.


```python
# classifier = RandomForestClassifier(max_depth=2, n_estimators=..., random_state=random_state, criterion='entropy') # n_estimators à compléter

accuracy_train, accuracy_test, trained_classifier = calculate_accuracy(classifier, X_train, X_test, y_train, y_test)
metrics.plot_confusion_matrix(trained_classifier, X_test, y_test)
```

Quel modèle est plus performant : Decision Tree ou Random Forest ?

## Question 6. Quelles autres métriques peuvent être utilisées pour un problème de classification ?

Pour répondre à cette question, regardez les métriques standards proposées dans **scikit-learn** : https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics

Affichez la **courbe ROC** pour le modèle de Random Forest.


```python
# metrics.plot_roc_curve(estimator=..., X=..., y=...) # options à compléter
```

Affichez un **rapport** avec plusieurs métriques pour le modèle de Random Forest.


```python
# report = metrics.classification_report(y_true=..., y_pred=..., target_names=y.unique()) # options à compléter
# print(report) # à exécuter
```
