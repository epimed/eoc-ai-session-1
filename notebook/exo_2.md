# Exercice 2 : Régression Logistique (LR) et Support Vector Machine (SVM)

Reprise du code déjà réalisé


```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
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
X = data.select_dtypes('number')
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=random_state, stratify=y)
print('Train:', X_train.shape, 'Test:', X_test.shape)
```

## Question 1. Appliquer une normalisation centrée-réduite aux données

Pour faciliter la convergence des algorithmes de machine learning, il est fortément conseillé de normaliser les données. Une approche standard est la normalisation centrée-réduite qui soustrait la moyenne et divise par l'écart-type les valeurs d'expression.

**Attention !** Le calcul de la moyenne $\mu$ et l'écart-type $\sigma$ doit être réalisé **uniquement sur le dataset d'entrainement**. Le dataset de test ne doit pas être utilisé dans le calcul. Il sera normalisé en utilisant les valeurs $\mu$ et $\sigma$ calculées sur le dataset d'entrainement.

La librairie **scikit-learn** contient un certain nombre de *scalers* qui permettent de réaliser facilement cette normalisation. Ainsi, le scaler `StandardScaler` permet de centrer-réduire les données.


```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler() # crée un scaler
scaler.fit(X_train) # calcule mu et sigma sur X_train uniquement
```


```python
# Décommenter et exécuter les lignes ci-dessous pour afficher mu et sigma pour chaque gène
# print('Mean mu', scaler.mean_)
# print('Std sigma', scaler.scale_)
```

Appliquer les valeurs $\mu$ et $\sigma$ calculées pour X_train aux deux datasets : **X_train** et **X_test**. Pour cela, il faut utiliser la méthode `transform` du *scaler*. 


```python
# A exécuter
X_train_scaled = scaler.transform(X_train) # numpy object
X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns) # conversion en pandas DataFrame

X_test_scaled = scaler.transform(X_test) # numpy object
X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns) # conversion en pandas DataFrame
```

Après normalisation, la moyenne des valeurs d'expression pour chaque gène dans le dataset **X_train_scaled** doit être égale 0. Vérifiez que c'est réellement le cas. Pour cela, calculez la moyenne à l'aide de la méthode `mean()` et affichez quelques premières valeurs avec `head()`.


```python
# X_train_scaled... # à compléter 
```

Après normalisation, l'écart-type doit être égal à 1. Vérifiez-le en calculant l'écart-type du dataset **X_train_scaled** à l'aide de la méthode `std()`. Affichez quelques premières valeurs avec `head()`.


```python
# X_train_scaled... # à compléter
```

## Question 2. Créer un modèle de régression logistique (LR)

La **régresssion logistique** utilise une fonction analytique, appelée fonction **logistique** ou fonction **sigmoïde**, qui a une forme caractéristique en **S**. En optimisant les coefficients de cette courbe (*max* de vraisemblance ou *min* de cross-entropie), elle permet d'estimer la probabilité pour un échantillon d'appartenir à telle ou telle classe. Par exemple, *tumoral* versus *normal*. 

<img src="logistic_regression.png" alt="Logistic regression" width="600" aling="center">

Créez le modèle de LR en utilisant les données normalisées et calculer la métrique *accuracy*.


```python
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=random_state, penalty='none')

# accuracy_train, accuracy_test, trained_classifier = calculate_accuracy(classifier, ... , ... , y_train, y_test) # à compléter
```

Affichez la matrice de confusion.


```python
# metrics.plot_confusion_matrix(trained_classifier, ... , ...) # à compléter
```

Est-ce que ce modèle est plus performant que Decision Tree ou Random Forest de l'exercice 1 ?

## Question 3. Evaluer l'impact de chaque gène dans le modèle LR

Après la phase d'entrainement, il est possible de connaître les paramètres $\beta$ du modèle obtenu. Il sont disponibles dans l'attribut `coef_`. Plus le coefficient $\beta$ est grand (en valeur absolue), plus l'impact du gène correspondant est important dans le modèle.

Affichez les coefficients $\beta$ pour quelques premiers gènes :


```python
coefficients = pd.DataFrame(trained_classifier.coef_[0], index=X_train_scaled.columns, columns=['beta'])
coefficients.head()
```

Présentez les coefficients $\beta$ sous forme d'un *barplot*, du plus petit au plus grand.


```python
coefficients = coefficients.sort_values(by='beta')
coefficients.plot.bar(figsize=(15, 3))
```

Quels gènes impactent fortement le modèle ?

## Question 4. Analyser la corrélation entre les meilleurs prédicteurs proposés par le modèle

Identifiez N meilleurs gènes qui impactent le plus fortement le modèle LR.


```python
n_features = 3
coefficients['abs_beta'] = coefficients['beta'].abs() # calcul des valeurs absolues des beta
coefficients = coefficients.sort_values(by='abs_beta', ascending=False) # tri par valeurs absolues
top_features = list(coefficients.head(n_features).index) # liste de n meilleurs features
print('Top features LR:', top_features)
```

Affichez un paiplot pour ces gènes, pour estimer qualitativement leur corrélation. Utilisez la librairie *seaborn* de Python.


```python
%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(data[[*top_features, 'tissue_status']], hue='tissue_status')
```

Est-ce que les gènes sont corrélés ? Est-ce qu'il y a un intérêt de les utiliser ensemble pour l'apprentissage du modèle ?

## Question 5. Calculer les performances du modèle LR en utilisant 1, 2, ... n top features 

Qu'est-ce que fait le code ci-dessous ? Exécutez-le et expliquez le résultat.


```python
for i in range(len(top_features)):
    selected_features = top_features[0:i+1]
    print(selected_features)
    accuracy_train, accuracy_test, trained_classifier = calculate_accuracy(classifier, X_train_scaled[selected_features], X_test_scaled[selected_features], y_train, y_test)
```

## Question 6. Créer un modèle linéaire de Support Vector Machine (SVM)

Le **modèle SVM** cherche à définir une frontière entre deux ou plusieurs classes d'échantillons, en maximisant la marge entre cette frontière et les échantillons les plus proches (vecteurs de support).

<img src="svm.png" alt="Support Vector Machine" width="600" aling="center">

Créez le modèle SVM en utilisant les données normalisées, calculez la métrique *accuracy* et la matrice de confusion.


```python
from sklearn.svm import LinearSVC

classifier = LinearSVC(random_state=random_state)

# accuracy_train, accuracy_test, trained_classifier = calculate_accuracy(...) # à compléter
# metrics.plot_confusion_matrix(...) # à compléter
accuracy_train, accuracy_test, trained_classifier = calculate_accuracy(classifier, X_train_scaled, X_test_scaled, y_train, y_test)
metrics.plot_confusion_matrix(trained_classifier, X_test_scaled, y_test)
```

## Question 7. Evaluer l'impact de chaque gène dans le modèle SVM


```python
coefficients = pd.DataFrame(trained_classifier.coef_[0], index=X_train_scaled.columns, columns=['beta'])

coefficients = coefficients.sort_values(by='beta')
coefficients.plot.bar(figsize=(15, 3))
```


```python
coefficients['abs_beta'] = coefficients['beta'].abs()
coefficients = coefficients.sort_values(by='abs_beta', ascending=False)
top_features = list(coefficients.head(n_features).index)
print('Top features SVM:', top_features)
```

Comparez le résultat avec LR. 

## Question 8. Etude de cas

L'hôpital *AI-Hospital* a mis au point un nouvel outil diagnostique basé sur les niveaux d'expression d'un panel de 3 gènes. Cet outil a donné les mesures suivantes pour un nouveau patient à l'hôpital :


```python
new_patient = {'RNF43': 4.68, 'SLC7A5': 4.10, 'DAO': 7.59} # dict
```

### Question principale : Est-ce que ce patient est atteint d'un cancer du colon ?

Composez un dataset de test **X_test** qui contient un seul échantillon - le nouveau patient.


```python
X_test = pd.DataFrame([new_patient], index=['new_patient'])
X_test
```


```python
panel = X_test.columns # liste de gènes dans le panel
print(panel)
```

Utilisez la totalité de données disponibles en tant que dataset d'entrainement **X_train**.


```python
X_train = data[panel]
y_train = y
```

Entrainez une Régression Logistique. Faites une prédiction pour le patient.


```python
# Code à écrire soi-même
```

### Question bonus : Quelle est la probabilité prédite pour ce patient d'avoir un cancer du colon ? 

Indice : Utilisez la méthode `predict_proba()` du modèle à la place de `predict()` pour obtenir la probabilité de prédiction.


```python
# Code à écrire soi-même
```
