# Projet Python pour la Data Science

## Sujet: 
En France métropolitaine, la température a augmenté de 1,8°C dans la décennie 2011-2019 par rapport à la moyenne 1901-1930, dont 1,0°C par rapport à la moyenne sur 1976-2005. Ce réchauffement peut avoir plusieurs incidences : augmentation des inondations, des orages, des vagues de chaleur, des sécheresses, des incendies et diminution de l’enneigement. Ces événements ont un impact durable sur les flux touristiques en France métropolitaine ainsi que sur les infrastructures touristiques : de nouveaux territoires gagnent en attractivité, les habitudes des vacanciers commencent à changer, redessinant progressivement les saisons touristiques.

**Risque par risque, la vulnérabilité du tourisme face au climat**
![Risque par risque, la vulnérabilité du tourisme face au climat](https://www.adaptation-changement-climatique.gouv.fr/sites/cracc/files/inline-images/catastrophes%20climatiques.png)

*Tableau extrait du guide ADEME ‘Opérateurs et territoires touristiques : s'adapter pour faire face au changement climatique’*

En nous appuyant sur la fréquentation des hébergements touristiques et sur les données climatologiques, nous cherchons à représenter l’impact du réchauffement climatique sur les flux touristiques au niveau départemental. 
Dans ce projet, nous utilisons la température comme indicateur du réchauffement climatique, par souci de concision, même si d'autres variables auraient pu être pertinentes (sécheresse, humidité). Nous distinguons également le tourisme d’été et le tourisme d’hiver entre 2011 et 2019 afin de mieux représenter la répartition des flux touristiques par département en France métropolitaine.

## Problématique: 
Comment les variations de température entre 2011 et 2019 affectent-elles les flux touristiques d'été et d'hiver, mais également la répartition spatiale des séjours en France métropolitaine? Dans quelle mesure les données disponibles permettent-elles d’identifier ces effets ?

## Modèle utilisé:
### Cadre théorique : désaisonnalisation
Pourquoi désaisonnaliser avant les tests de stationnarité ?
Les tests de racine unitaire (ADF, PP, KPSS) reposent sur des hypothèses asymptotiques qui sont violées en présence de :
saisonnalité déterministe non traitée, ruptures périodiques régulières (mensuelles ici). Une saisonnalité non corrigée peut conduire à : une fausse non-stationnarité, une surestimation de l’ordre d’intégration,
des conclusions erronées sur la cointégration.

Approche retenue : saisonnalité déterministe
Nous utilisons une approche classique et valide économétriquement :

Yt = µ + 𝛿1 * D1 + ... + 𝛿12 * D11 + ε où ε~BB(0,sd), 𝐷𝑚 : sont des dummies mensuelles,
Décision :
si au moins un coefficient saisonnier est significatif alors saisonnalité présente
sinon alors pas de correction nécessaire

### Cadre théorique : test de Dickey-Fuller augmenté (ADF)
Problématique de la stationnarité
En économétrie des séries temporelles, une série non stationnaire pose trois problèmes majeurs :
risque de régression fallacieuse, lois asymptotiques non standards, tests de significativité invalides
Une série est stationnaire si : sa moyenne est constante, sa variance est finie et constante, sa structure d’autocorrélation est stable dans le temps

H₀ : la série possède une racine unitaire (non stationnaire) VS H₁ : la série est stationnaire

Décision :
si p-value < α → rejet de H₀ → série stationnaire
sinon on calcule la serie differncier(Yt - Yt-1) puis on refait le test.

l'ordre d'integration est le nombre de foi que l'on à du differencier la serie pour que celle -ci devienne stattionnaire

### Cadre théorique :modèle ARDL 

Le modèle ARDL (AutoRegressive Distributed Lag) est adapté lorsque : Les variables sont intégrées d’ordre différent (I(0) et I(1)), et que l'on souhaite distinguer effets de court terme et relation de long terme.

Forme générale :

Yt = µ + 𝛿1 * D1 + ... + 𝛿12 * D11 + a1 * Yt-1 + ... + ap * Yt-p + b1 * X't + ... + bq * X't-p + ε ou ε~BB(0,sd)

Yt: flux touristique à la periode t

𝑋t : variables climatiques à la periode t

𝛽 : effet du climat sur le flux touristique

### Modèle de prediction :XGBOOST

XGBoost est un algorithme de gradient boosting sur arbres de décision qui construit un modèle prédictif comme une somme séquentielle d’arbres faibles, chaque nouvel arbre corrigeant les erreurs des précédents par descente de gradient.
Sa spécificité réside dans une fonction objectif régularisée et l’utilisation d’une approximation de Taylor d’ordre 2, ce qui lui confère une forte performance prédictive, au prix d’une interprétabilité limitée et sans vocation causale.

## Navigation au sein du projet

Il suffit d'exécuter successivement les cellules du rapport : [rapport.ipynb](https://github.com/Manonchvllr/Project/blob/main/rapport.ipynb)

## Données utilisées: 

• [INSTITUT NATIONAL DE LA STATISTIQUE ET DES ÉTUDES ÉCONOMIQUES (INSEE)](https://www.data.gouv.fr/api/1/datasets/r/1129fd80-2564-452c-86d4-9e36e7cca4a5). Fréquentation des hébergements touristiques, data.gouv.fr, 2025

• [MÉTÉO-FRANCE](https://www.data.gouv.fr/datasets/donnees-climatologiques-de-base-mensuelles/). Données climatologiques de base – mensuelles, data.gouv.fr, 2025.
