# -*- coding: utf-8 -*-
"""
Created on Sat Jan  25 21:25:07 2020

@authors: yassine_ABDOU
"""

# =============================================================================
# Package
# =============================================================================

# Importation des packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import random

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from missingpy import MissForest
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from statsmodels.graphics.mosaicplot import mosaic 

# Vecteur contenant des couleurs
pal_col = ["#007bff", "#DC3545", "#e8ba11", "#990707", "#063f0e", "#39a9db"]

# fixer la graine
random.seed(1994)

# =============================================================================
# Jeu de données
# =============================================================================
# Importation de la table de données
df_housing = pd.read_csv('./Data/data_housing.csv', sep=';',decimal='.', header = 0)

# supprimer la variable 'Id'
df_housing.drop(columns= 'Id', inplace =  True)

# Info() nous donne le nombre de données manquantes par variable et le type de la variable
#df_housing.info()

# 10 premières lignes de notre table de données
df_housing.head(10)

# Dimension de la table de données.
df_housing.shape

# Transformer les variables qualitatives (coder en numérique) en variable catégorielles
var_a_transf = ['MSSubClass', 'Fireplaces', 'OverallQual', 'OverallCond', 'FullBath', 'HalfBath', 'KitchenAbvGr', 'MoSold']
for var in var_a_transf:
    df_housing[var] = df_housing[var].astype('category')

# Création d'une copie de notre jeu de données.
df_housing_copy=df_housing.copy()

# =============================================================================
# Données manquantes (Comptage et visualisation)
# =============================================================================
# Les variables avec des NA's
col_with_NA = df_housing_copy.columns[df_housing_copy.isnull().any()]

# Data contient seulement les variables avec des NA's
df_with_colNA = df_housing_copy[pd.array(col_with_NA)]

# Pourcentage des NA's par variable
per_NA_per_col = 100*(df_with_colNA.isnull().sum())/len(df_housing_copy)
per_NA_per_col

# Représentation graphique du pourcentage des NA's par colonne
per_NA_per_col.sort_values(ascending=False,inplace=True)
sns.set_style("whitegrid") # Theme du graphe
per_NA_per_col.plot.bar(color= pal_col[1])
plt.title("Pourcentage des NA's par colonne")
plt.xlabel("Variables")
plt.ylabel("Pourcentage")
plt.figure(figsize = (100,100))
# =============================================================================
# Création d'une nouvelle variable "Class_prix"
# =============================================================================

# Valeurs de SalePrice à 20%, 50% et 80% de la polutation
np.quantile(df_housing_copy["SalePrice"],0.2)
np.quantile(df_housing_copy["SalePrice"],0.5)
np.quantile(df_housing_copy["SalePrice"],0.8)

# Création de la variable 'Class_prix'
Class_prix = []
for prix in df_housing_copy.SalePrice:
    if prix<np.quantile(df_housing_copy["SalePrice"],0.20):
        classe = 'Classe0'
    elif (prix<np.quantile(df_housing_copy["SalePrice"],0.50))&(prix>=np.quantile(df_housing_copy["SalePrice"],0.20)):
        classe = 'Classe1'
    elif (prix<np.quantile(df_housing_copy["SalePrice"],0.80))&(prix>=np.quantile(df_housing_copy["SalePrice"],0.50)):
        classe = 'Classe2'
    else:
        classe = 'Classe3'
    Class_prix.append(classe)
    
df_housing_copy["Class_prix"] = Class_prix

# Supprimer la variable "Id"
df_housing_copy.drop(columns= 'SalePrice', inplace =  True)

# compter le nombre de lignes dans chaque classe 
Frq=df_housing_copy.groupby(['Class_prix'])['Class_prix'].count()
print('Les fréquences sont : \n', Frq)

# Barplot des fréquences : 
Frq.sort_values(ascending=False,inplace=True)
sns.set_style("whitegrid")
Frq.plot.bar(color= pal_col[5])
plt.title("Nombre d'individus par classe")
plt.xlabel("Classe")
plt.ylabel("Nombre")
plt.figure(figsize = (100,100)) 

# =============================================================================
# Statistique descriptive
# =============================================================================

# Graphe de corrélation (spearman)
corr = df_housing_copy.corr('spearman')
# Génère un masque pour le triangle supérieur
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Configurer la figure matplotlib
f, ax = plt.subplots(figsize=(11, 9))
# Générer une palette de couleurs divergente personnalisée
cmap = sns.diverging_palette(50, 10, as_cmap=True)
# Dessinez la carte thermique avec le masque et le rapport d'aspect correct
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,annot=True)

# Les boxplots
# Année de construction en fonction des classes de prix
sns.set_style("whitegrid")
sns.boxplot(y="YearBuilt",x="Class_prix", data = df_housing_copy, order=["Classe0", "Classe1", "Classe2", "Classe3"], palette = pal_col)
plt.title("Année de construction en fonction des classes de prix")
plt.xlabel("Classe")
plt.ylabel("Année de construction")

# Superficie du garage en fonction des classes de prix
sns.set_style("whitegrid")
sns.boxplot(y="GarageArea",x="Class_prix", data = df_housing_copy,order=["Classe0", "Classe1", "Classe2", "Classe3"], palette = pal_col)
plt.title("Superficie du garage en fonction des classes de prix")
plt.xlabel("Classe")
plt.ylabel("Superficie du garage")

# Nombre de chambre (Sans salles de bains) en fonction des classes de prix
sns.set_style("whitegrid")
sns.boxplot(y="TotRmsAbvGrd",x="Class_prix", data = df_housing_copy,order=["Classe0", "Classe1", "Classe2", "Classe3"], palette = pal_col)
plt.title("Nombre de chambre (Sans salles de bains) en fonction des classes de prix")
plt.xlabel("Classe")
plt.ylabel("Nombre de chambre (Sans salles de bains)")

# classification générale de zonage en fonction des classes de prix
mosaic(df_housing_copy,["Class_prix","MSZoning"],gap=0.3)
# =============================================================================
# Imputation des données manquantes (Première méthode-Mode/Médiane)
# =============================================================================

# transformer la table des pourcentages des NA's en DataFrame.
df_per_NA_per_col = per_NA_per_col.reset_index().rename(columns={"index": "Variable", 0: "pourcentage"}).sort_values(by = 'pourcentage')
df_per_NA_per_col_sup50 = df_per_NA_per_col.loc[df_per_NA_per_col.pourcentage > 50]
df_per_NA_per_col_inf50 = df_per_NA_per_col.loc[df_per_NA_per_col.pourcentage <= 50]

# Suppression des variables avec plus de 50% des NA's
df_housing_copy.drop(columns= df_per_NA_per_col_sup50.Variable, inplace = True)

# Data contenant que les variables qualitatives : 
var_qualitative = df_housing_copy.select_dtypes(exclude=['float', 'integer'])

# Data contenant que les variables quantitatives : 
var_quantitative = df_housing_copy.select_dtypes(['float', 'integer'])

# Construire une base pour les var quali et une pour les quant
Df_varQual_withNA = df_housing_copy[df_per_NA_per_col_inf50.Variable].select_dtypes(exclude=["number","bool_"])
Df_varQuant_withNA = df_housing_copy[df_per_NA_per_col_inf50.Variable].select_dtypes(exclude=["object_","bool_"])

# Faire une copie de notre base "df_housing_copy()"
df_housing_impute1=df_housing_copy.copy()

# Imputer les var_quali par le mode et les var_quant par la médiane
# Variables Quantitatives
for var in list(Df_varQuant_withNA.columns):
    df_housing_impute1.loc[:,var]=df_housing_impute1.loc[:,var].fillna(df_housing_impute1.loc[:,var].median())

# Variables Qualitatives
for var in list(Df_varQual_withNA.columns):
    df_housing_impute1.loc[:,var]=df_housing_impute1.loc[:,var].fillna(df_housing_impute1.loc[:,var].mode())

# Vérification de l'imputation
df_housing_impute1.isnull().sum()

# Indixation des variables qualitatives
for var in list(var_qualitative.columns):
    df_housing_impute1[var] = pd.Series(df_housing_impute1[var], dtype="category").cat.codes

# =============================================================================
# Selection des variables
# =============================================================================
# Data sans variable à expliquer
X=df_housing_impute1.loc[:, df_housing_impute1.columns != "Class_prix"]

# Variable à expliquer
y=df_housing_impute1.Class_prix

# Création d'une instance de la classe
lr = LogisticRegression()

# Algorithme de séléction des variables
selecteur = RFE(estimator=lr)

# Lancement de la recherche 
sol = selecteur.fit(X,y)

# Nombre de variables séléctonnées
print(sol.n_features_) 

# liste des variables séléctionnées
print(sol.support_)

# ordre de suppression (Les variables restantes sont indexées 1)
print(sol.ranking_)

# Réduction de la base aux variables séléctionnées
X = X.loc[:,sol.support_]

# Dimension de la nouvelles base
print(X.shape)

# Division de la base de données en base de test et base d'entrainement
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=1994)

# Dimension des nouvelles table (train et test)
print(X_train.shape,X_test.shape)

# =============================================================================
# Prédiction de notre variables d'intérêt (Avec première méthode d'imputation)
# =============================================================================

###___________________ KNeighbors ___________________###
# Premier modèle (Modèle par défaut)
model_KNN = KNeighborsClassifier()
model_KNN.fit(X_train, y_train)
print("Le plus proche voisin (Knn): \nTrain : {0:f} \nTest  : {1:f}".format(model_KNN.score(X_train, y_train),model_KNN.score(X_test, y_test)))

# Meilleur modèle
k_range = list([1,2,3,4,5,6,7,8,9,10,11,17,30,45,60]) # list des valeurs de K
k_scores = []

# Validation croisée pour chaque valeurs de K
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())

data = pd.DataFrame(k_scores)

# Précision du KNN en fonction des valeurs de K
plt.plot(k_range, k_scores, 'bo')
plt.plot(k_range, k_scores, '--')
plt.title("Précision de la validation croisée en fonction des valeurs de K pour KNN")
plt.xlabel('Valeurs de K')
plt.ylabel('Précision')

knn = KNeighborsClassifier(n_neighbors=17)
knn.fit(X_train,y_train)
print("Knn Optimal : \nTrain : {0:f} \nTest  : {1:f}".format(knn.score(X_train, y_train),knn.score(X_test, y_test)))

# Matrice de confusion et Trace
confusion_matrix(y_test, knn.predict(X_test))
np.trace(confusion_matrix(y_test, knn.predict(X_test)))

###___________________ Random forest ___________________###
# Premier modèle (Modèle par défaut)
model_RF = RandomForestClassifier()
model_RF.fit(X_train, y_train)
print("Random Forest \nTrain : {0:f} \nTest  : {1:f}".format(model_RF.score(X_train, y_train),model_RF.score(X_test, y_test)))

# Importance des variables - RF
ar_RF = np.array([X_test.columns,model_RF.feature_importances_])
Var_importance_RF = pd.DataFrame(np.transpose(ar_RF), columns = ['Variable','Importance'])
Importance_sup003_RF=Var_importance_RF[Var_importance_RF["Importance"]>0.03]
fig, ax = plt.subplots()
ax.barh( np.arange(Importance_sup003_RF.shape[0]), Importance_sup003_RF.Importance.sort_values(), align='center',color=pal_col[5])
ax.set_yticks(np.arange(Importance_sup003_RF.shape[0]))
ax.set_yticklabels(Importance_sup003_RF.Variable)
ax.set_xlabel('Performance')
ax.set_title('Importance des variables - Modèle Random Forest')

# Meilleur modèle
Prm=[{"n_estimators":list(range(300,1000,50))}]
model_RF_opt= GridSearchCV(RandomForestClassifier(),Prm)
model_RF_opt.fit(X_train, y_train)
model_RF_opt.best_params_
print("Random Forest \nTrain : {0:f} \nTest  : {1:f}".format(model_RF_opt.score(X_train, y_train),model_RF_opt.score(X_test, y_test)))

# Matrice de confusion et Trace
confusion_matrix(y_test, model_RF_opt.predict(X_test))
np.trace(confusion_matrix(y_test, model_RF_opt.predict(X_test)))

###___________________ ADABoost ___________________###
# Premier modèle (Modèle par défaut)
model_ADA = AdaBoostClassifier()
model_ADA.fit(X_train, y_train)
print("Le modèle ADAboost \nTrain : {0:f} \nTest  : {1:f}".format(model_ADA.score(X_train, y_train),model_ADA.score(X_test, y_test)))

# Modème optimal
param=[{"n_estimators":list(range(10,200,5))}]
model_ADA_opt= GridSearchCV(AdaBoostClassifier(),param)
model_ADA_opt.fit(X_train, y_train)
model_ADA_opt.best_params_
print("Le modèle ADABoost optimal \nTrain : {0:f} \nTest  : {1:f}".format(model_ADA_opt.score(X_train, y_train),model_ADA_opt.score(X_test, y_test)))

# Matrice de confusion et Trace
confusion_matrix(y_test, model_ADA_opt.predict(X_test))
np.trace(confusion_matrix(y_test, model_ADA_opt.predict(X_test)))

###___________________ Gradient Boosting ___________________###
# Premier modèle (Modèle par défaut)
model_GBT = GradientBoostingClassifier()
model_GBT.fit(X_train, y_train)
print("Le modèle GBT : \nTrain : {0:f} \nTest   : {1:f}".format(model_GBT.score(X_train, y_train),model_GBT.score(X_test, y_test)))

# Importance des variables - Gradient Boosting
ar_GBT = np.array([X_test.columns,model_GBT.feature_importances_])
Var_importance_GBT = pd.DataFrame(np.transpose(ar_GBT), columns = ['Variable','Importance'])
Importance_sup003_GBT=Var_importance_GBT[Var_importance_GBT["Importance"]>0.03]
fig, ax = plt.subplots()
ax.barh( np.arange(Importance_sup003_GBT.shape[0]), Importance_sup003_GBT.Importance.sort_values(), align='center',color=pal_col[5])
ax.set_yticks(np.arange(Importance_sup003_GBT.shape[0]))
ax.set_yticklabels(Importance_sup003_GBT.Variable)
ax.set_xlabel('Performance')
ax.set_title('Importance des variables - Gradient Boosting')

# Modèle optimal
param=[{"n_estimators":list(range(10, 500, 50))}]
model_GBT_opt= GridSearchCV(GradientBoostingClassifier(),param)
model_GBT_opt.fit(X_train, y_train)
model_GBT_opt.best_params_
print("Le modèle GBT optimal : \nTrain : {0:f} \nTest   : {1:f}".format(model_GBT_opt.score(X_train, y_train),model_GBT_opt.score(X_test, y_test)))

# Matrice de confusion et trace
confusion_matrix(y_test, model_GBT_opt.predict(X_test))
np.trace(confusion_matrix(y_test, model_GBT_opt.predict(X_test)))

# =============================================================================
# Imputation des données manquantes (Deuxième méthode-MissForest)
# =============================================================================

# Recopier notre base initiale (Afin de l'utiliser pour la deuxième méthode d'imputation)
df_housing_impute2=df_housing_copy.copy()

# Indixation des variables qualitatives
for var in list(var_qualitative.columns):
    df_housing_impute2[var] = pd.Series(df_housing_impute2[var], dtype="category").cat.codes
    for i in range(0,len(df_housing)):
        if df_housing_impute2[var][i] == -1:
            df_housing_impute2[var][i] = np.nan

# imputation par MissForest
imputer = MissForest(missing_values = np.nan)
Ny_imputed = imputer.fit_transform(df_housing_impute2)
df_housing_impute2=pd.DataFrame(Ny_imputed, columns=df_housing_impute2.columns.values.tolist())
df_housing_impute2.isnull().sum()

# =============================================================================
# Sélection des variables
# =============================================================================
# Data sans variable à expliquer
X2=df_housing_impute2.loc[:, df_housing_impute2.columns != "Class_prix"]

# Variable à expliquer
y2=df_housing_impute2.Class_prix

# Création d'une instance de la classe
lr1 = LogisticRegression()

# Algorithme de sélection de variables
selecteur1 = RFE(estimator=lr1)

# Lancement de la recherche 
sol1 = selecteur1.fit(X2,y2)

# Nombre de variables séléctonnées
print(sol1.n_features_) 

# Réduction de la base aux variables séléctionnées
X2 = X2.loc[:,sol1.support_]

# Dimension de la nouvelles base
print(X2.shape)

# division de la table
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size = 0.3, random_state=1994)

# Dimension des nouvelles tables (train et test)
print(X_train2.shape,X_test2.shape)

# =============================================================================
# Prédiction de notre variables d'intérêt (Avec deuxième méthode d'imputation)
# =============================================================================

###___________________ KNeighbors ___________________###
# Premier modèle (Modèle par défaut)
model_KNN2 = KNeighborsClassifier()
model_KNN2.fit(X_train2, y_train2)
print("Knn: \nTrain : {0:f} \nTest  : {1:f}".format(model_KNN2.score(X_train2, y_train2),model_KNN2.score(X_test2, y_test2)))

# Meilleur modèle
# list des valeurs de K
k_range = list([1,2,3,4,5,6,7,8,9,10,11,17,30,45,60])
k_scores2 = []

# Validation croisée pour chaque valeurs de K
for k in k_range:
    knn2 = KNeighborsClassifier(n_neighbors=k)
    scores2 = cross_val_score(knn2, X2, y2, cv=10, scoring='accuracy')
    k_scores2.append(scores2.mean())
    
pd.DataFrame(k_scores2).max()

# Précision du KNN en fonction des valeurs de K
sns.set_style("whitegrid")
plt.plot(k_range, k_scores2, 'bo')
plt.plot(k_range, k_scores2, '--')
plt.title("Précision de la validation croisée en fonction des valeurs de K pour KNN")
plt.xlabel('Valeurs de K pour le KNN')
plt.ylabel('Précision ')

knn2 = KNeighborsClassifier(n_neighbors=17)
knn2.fit(X_train2,y_train2)
print("Knn Optimal : \nTrain : {0:f} \nTest  : {1:f}".format(knn2.score(X_train2, y_train2),knn2.score(X_test2, y_test2)))

# Matrice de confusion et trace
confusion_matrix(y_test2, knn2.predict(X_test2))
np.trace(confusion_matrix(y_test2, knn2.predict(X_test2)))

###___________________ Random forest ___________________###
# Premier modèle (Modèle par défaut)
model_RF2 = RandomForestClassifier()
model_RF2.fit(X_train2, y_train2)
print("Random Forest \nTrain : {0:f} \nTest  : {1:f}".format(model_RF2.score(X_train2, y_train2),model_RF2.score(X_test2, y_test2)))

# Importance des variables - RF
ar2 = np.array([X_test2.columns,model_RF2.feature_importances_])
Var_importance2 = pd.DataFrame(np.transpose(ar2), columns = ['Variable','Importance'])
Importance_sup0012=Var_importance2[Var_importance2["Importance"]>0.03]
fig, ax = plt.subplots()
ax.barh( np.arange(Importance_sup0012.shape[0]), Importance_sup0012.Importance.sort_values(), align='center',color=pal_col[5])
ax.set_yticks(np.arange(Importance_sup0012.shape[0]))
ax.set_yticklabels(Importance_sup0012.Variable)
ax.set_xlabel('Performance')
ax.set_title('Importance des variables - Modèle Random Forest')

# Meilleur modèle
Prm=[{"n_estimators":list(range(300,1000,50))}]
model_RF_opt2= GridSearchCV(RandomForestClassifier(),Prm)
model_RF_opt2.fit(X_train2, y_train2)
model_RF_opt2.best_params_
print("Random Forest \nTrain : {0:f} \nTest  : {1:f}".format(model_RF_opt2.score(X_train2, y_train2),model_RF_opt2.score(X_test2, y_test2)))

# Matrice de confusion et trace
confusion_matrix(y_test2, model_RF_opt2.predict(X_test2))
np.trace(confusion_matrix(y_test2, model_RF_opt2.predict(X_test2)))

###___________________ ADABoost ___________________###
# Premier modèle (Modèle par défaut)
model_ADA2 = AdaBoostClassifier()
model_ADA2.fit(X_train2, y_train2)
print("Le modèle ADAboost \nTrain : {0:f} \nTest  : {1:f}".format(model_ADA2.score(X_train2, y_train2),model_ADA2.score(X_test2, y_test2)))

# Modème optimal
param=[{"n_estimators":list(range(10,200,5))}]
model_ADA_opt2= GridSearchCV(AdaBoostClassifier(),param)
model_ADA_opt2.fit(X_train2, y_train2)
model_ADA_opt2.best_params_
print("Le modèle ADABoost optimal \nTrain : {0:f} \nTest  : {1:f}".format(model_ADA_opt2.score(X_train2, y_train2),model_ADA_opt2.score(X_test2, y_test2)))

# Matrice de confusion et trace
confusion_matrix(y_test2, model_ADA_opt2.predict(X_test2))
np.trace(confusion_matrix(y_test2, model_ADA_opt2.predict(X_test2)))

###___________________ Gradient Boosting ___________________###
# Premier modèle (Modèle par défaut)
model_GBT2 = GradientBoostingClassifier()
model_GBT2.fit(X_train2, y_train2)
print("Le modèle GBT : \nTrain : {0:f} \nTest   : {1:f}".format(model_GBT2.score(X_train2, y_train2),model_GBT2.score(X_test2, y_test2)))

# Importance des variables - Gradient Boosting
ar_GBT2 = np.array([X_test2.columns,model_GBT2.feature_importances_])
Var_importance_GBT2 = pd.DataFrame(np.transpose(ar_GBT2), columns = ['Variable','Importance'])
Importance_sup003_GBT2=Var_importance_GBT2[Var_importance_GBT2["Importance"]>0.03]
fig, ax = plt.subplots()
ax.barh( np.arange(Importance_sup003_GBT2.shape[0]), Importance_sup003_GBT2.Importance.sort_values(), align='center',color=pal_col[5])
ax.set_yticks(np.arange(Importance_sup003_GBT2.shape[0]))
ax.set_yticklabels(Importance_sup003_GBT2.Variable)
ax.set_xlabel('Performance')
ax.set_title('Importance des variables - Gradient Boosting')

# Modème optimal
param=[{"n_estimators":list(range(10, 500, 50))}]
model_GBT_opt2= GridSearchCV(GradientBoostingClassifier(),param)
model_GBT_opt2.fit(X_train2, y_train2)
model_GBT_opt2.best_params_
print("Le modèle GBT optimal : \nTrain : {0:f} \nTest   : {1:f}".format(model_GBT_opt2.score(X_train2, y_train2),model_GBT_opt2.score(X_test2, y_test2)))

# Matrice de confusion et trace
confusion_matrix(y_test2, model_GBT_opt2.predict(X_test2))
np.trace(confusion_matrix(y_test2, model_GBT_opt2.predict(X_test2)))    