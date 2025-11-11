import pandas as pd
import numpy as np
from fonctions import recup_url, filtre_data

cols = [
    'FREQ', 
    'GEO', 
    'GEO_OBJECT',
    'TERRTYPO',
    'TOUR_MEASURE',
    'TOUR_RESID',
    'CONF_STATUS',
    'DECIMALS',
    'OBS_STATUS',
    'OBS_STATUS_FR',
    'UNIT_MULT',
    'TIME_PERIOD',
    'OBS_VALUE'
]

df = recup_url.url_to_df(url = "https://www.data.gouv.fr/api/1/datasets/r/1129fd80-2564-452c-86d4-9e36e7cca4a5",
                         cols_a_conserver=cols,
                         type_zip="zip",
                         plusieurs_fichiers=True)

# url = "https://www.data.gouv.fr/api/1/datasets/r/1129fd80-2564-452c-86d4-9e36e7cca4a5"
# dossier = requests.get(url)

# # Ouvrir le contenu ZIP en mémoire
# z = zipfile.ZipFile(io.BytesIO(dossier.content))

# # Extraire tous les fichiers
# z.extractall("donnees_zip")


# fichiers = [f for f in os.listdir("donnees_zip") if f.endswith(".csv")]
# df = pd.read_csv(os.path.join("donnees_zip", fichiers[1]), sep = ';', usecols=colonnes_a_conserver)


# on applique le facteur d'échelle 
df["OBS_VALUE_CORR"] = df["OBS_VALUE"] * (10 ** df["UNIT_MULT"])

# on met le bon nombre de décimales
df.loc[df["OBS_VALUE_CORR"].notna(), "OBS_VALUE_CORR"] = df.loc[df["OBS_VALUE_CORR"].notna()].apply(
    lambda x: round(x["OBS_VALUE_CORR"], int(x["DECIMALS"])),
    axis=1
)

# on garde seulement la valeur observée qui est corrigée
df = df.drop(columns=["DECIMALS", "UNIT_MULT", "OBS_VALUE"])

# on choisit le nombre d'arrivée comme indicateur
df = df.loc[df['TOUR_MEASURE'].isin(["ARR"])]

df = df.loc[df['OBS_STATUS'].isin(["A", "P"])]
# on exclut les valeurs manquantes (O), A= Normale (définitive/validée), P= Valeur provisoire
# en faisant: print(df["OBS_STATUS"].value_counts(dropna=False))
# on obtient OBS_STATUS; A 69627; Name: count, dtype: int64
# il n'y a donc pas de P, on va pouvoir supprimer la colonne OBS_STATUS
df = df.drop(columns=["OBS_STATUS"])

# on remarque que certaines valeurs définitives sont marquées Prov sous OBS_STATUS_FR
# A = valeur correcte du point de vue technique,
# mais OBS_STATUS_FR = "PROV" = pas encore consolidée statistiquement.
# en faisant: print(df["CONF_STATUS"].value_counts(dropna=False))
# on obtient CONF_STATUS; F 69627; Name: count, dtype: int64
# il n'y a donc que des observations diffusables, on va pouvoir supprimer la colonne CONF_STATUS
df = df.drop(columns=["CONF_STATUS","TOUR_MEASURE","OBS_STATUS_FR"])

# On prend les données mensuelles et on supprime la colonne FREQ
df = df.loc[df['FREQ'].isin(["M"])]
df = df.drop("FREQ", axis = 1)

# On définit les années que l'on veut garder
# on crée une variable ne contenant que l'annee et une autre le mois
df['AAAA'] = df['TIME_PERIOD'].astype(str).str[:4].astype(int)
df['MM']= df['TIME_PERIOD'].astype(str).str[5:7].astype(int)
df = df.drop("TIME_PERIOD", axis = 1)

#on filtre les données sur nos mois d'interet
df = filtre_data.filtre_annee_mois(df)

# On prend les données de departement et on supprime c
df = df.loc[df['GEO_OBJECT'].isin(["DEP"])]
df = df.drop("GEO_OBJECT", axis = 1)

# on supprime TERRTYPO car tout est identique
df = df.drop("TERRTYPO", axis = 1)

#changer le nom de la colonne GEO en DEP pour la fusion
col = df.columns.tolist()
col[0] = "DEP"
df.columns = col

# on exclut les DOM TOM
df = df.loc[df['DEP'].str.len() == 2]

# la corse est codée 20 dans l'autre fichier, on modifie
df['DEP'] = df['DEP'].replace(['2A', '2B'],['20', '20'])

# la variable "TOUR_RESID" donne l'origine du touriste : on filtre sur total (on ne distingue pas pour l'instant)
df = df.loc[df['TOUR_RESID'].isin(["_T"])]

# on somme les arrivées par année et mois
df = df.groupby(['AAAA','MM', 'DEP'])["OBS_VALUE_CORR"].sum()

# on remet année et mois (devenues index) en variables normales
df = df.reset_index() 

base_touri = df

# A CHANGER 
base_touri.to_csv("~/Project/Data/data_tourisme.csv")