import pandas as pd
import numpy as np
from fonctions import recup_url, filtre_data

cols = [
    'ACTIVITY',
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

import os
import s3fs
os.environ["AWS_ACCESS_KEY_ID"] = 'USJHGHP393O2D500IZ5A'
os.environ["AWS_SECRET_ACCESS_KEY"] = 'QmI8KqTNYkNqnnxyUxCvwlbszEJSji1jtQ4HkgoA'
os.environ["AWS_SESSION_TOKEN"] = 'eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3NLZXkiOiJVU0pIR0hQMzkzTzJENTAwSVo1QSIsImFsbG93ZWQtb3JpZ2lucyI6WyIqIl0sImF1ZCI6WyJtaW5pby1kYXRhbm9kZSIsIm9ueXhpYSIsImFjY291bnQiXSwiYXV0aF90aW1lIjoxNzY1ODg0MDA3LCJhenAiOiJvbnl4aWEiLCJlbWFpbCI6Im1hbm9uX2NoZXZhbGxpZXJAZW5zYWUuZnIiLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiZXhwIjoxNzY2NDkzNzU4LCJmYW1pbHlfbmFtZSI6IkNoZXZhbGxpZXIiLCJnaXZlbl9uYW1lIjoiTWFub24iLCJncm91cHMiOlsiVVNFUl9PTllYSUEiXSwiaWF0IjoxNzY1ODg4OTU4LCJpc3MiOiJodHRwczovL2F1dGgubGFiLnNzcGNsb3VkLmZyL2F1dGgvcmVhbG1zL3NzcGNsb3VkIiwianRpIjoib25ydHJ0OjdjZTYyMmE5LTQ0ZGUtNDU0NC05NDhkLTZlNDk5Y2E2ZjM2YSIsImxvY2FsZSI6ImZyIiwibmFtZSI6Ik1hbm9uIENoZXZhbGxpZXIiLCJwb2xpY3kiOiJzdHNvbmx5IiwicHJlZmVycmVkX3VzZXJuYW1lIjoibWFjaGV2YWxsaWVyIiwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iLCJkZWZhdWx0LXJvbGVzLXNzcGNsb3VkIl19LCJyZXNvdXJjZV9hY2Nlc3MiOnsiYWNjb3VudCI6eyJyb2xlcyI6WyJtYW5hZ2UtYWNjb3VudCIsIm1hbmFnZS1hY2NvdW50LWxpbmtzIiwidmlldy1wcm9maWxlIl19fSwicm9sZXMiOlsib2ZmbGluZV9hY2Nlc3MiLCJ1bWFfYXV0aG9yaXphdGlvbiIsImRlZmF1bHQtcm9sZXMtc3NwY2xvdWQiXSwic2NvcGUiOiJvcGVuaWQgcHJvZmlsZSBncm91cHMgZW1haWwiLCJzaWQiOiIwNjhkMmUwNS02MTA5LWViODMtNjM4My02YWYxZTA3NmZhMWIiLCJzdWIiOiJmMGQxN2VkNC01OTJmLTQyMzAtYjI5YS00MjMxZjQ4ZDc1MzMiLCJ0eXAiOiJCZWFyZXIifQ.96BQIQHOyfzykDva4dvciDFvLIHwdfHMX3OS0Fw5JDwtCF0c7J_AWybFnbG26YBaeZk2t_is6Oaa0pMCOTxCyQ'
os.environ["AWS_DEFAULT_REGION"] = 'us-east-1'
fs = s3fs.S3FileSystem(
    client_kwargs={'endpoint_url': 'https://'+'minio.lab.sspcloud.fr'},
    key = os.environ["AWS_ACCESS_KEY_ID"], 
    secret = os.environ["AWS_SECRET_ACCESS_KEY"], 
    token = os.environ["AWS_SESSION_TOKEN"])


df = pd.read_csv("s3://machevallier/DS_TOUR_FREQ_data.csv", sep=';', usecols=cols)


#df = recup_url.url_to_df(url = "https://www.data.gouv.fr/datasets/frequentation-des-hebergements-touristiques/#/resources/1129fd80-2564-452c-86d4-9e36e7cca4a5",
                         #cols_a_conserver=cols,
                         #type_zip="zip",
                         #plusieurs_fichiers=True)

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
df = df.loc[df['TOUR_MEASURE'].isin(["NUI"])]



df['ACTIVITY'] = df['ACTIVITY'].replace(['I553','I551'],['CAMPING', 'HOTEL'])

df = df[df["ACTIVITY"] == "CAMPING"]

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

departement_noms = {
    "01": "Ain", "02": "Aisne", "03": "Allier", "04": "Alpes-de-Haute-Provence",
    "05": "Hautes-Alpes", "06": "Alpes-Maritimes", "07": "Ardèche", "08": "Ardennes",
    "09": "Ariège", "10": "Aube", "11": "Aude", "12": "Aveyron", "13": "Bouches-du-Rhône",
    "14": "Calvados", "15": "Cantal", "16": "Charente", "17": "Charente-Maritime",
    "18": "Cher", "19": "Corrèze", "21": "Côte-d'Or", "22": "Côtes-d'Armor",
    "23": "Creuse", "24": "Dordogne", "25": "Doubs", "26": "Drôme", "27": "Eure",
    "28": "Eure-et-Loir", "29": "Finistère", "2A": "Corse", "2B": "Corse",
    "30": "Gard", "31": "Haute-Garonne", "32": "Gers", "33": "Gironde", "34": "Hérault",
    "35": "Ille-et-Vilaine", "36": "Indre", "37": "Indre-et-Loire", "38": "Isère",
    "39": "Jura", "40": "Landes", "41": "Loir-et-Cher", "42": "Loire", "43": "Haute-Loire",
    "44": "Loire-Atlantique", "45": "Loiret", "46": "Lot", "47": "Lot-et-Garonne",
    "48": "Lozère", "49": "Maine-et-Loire", "50": "Manche", "51": "Marne",
    "52": "Haute-Marne", "53": "Mayenne", "54": "Meurthe-et-Moselle", "55": "Meuse",
    "56": "Morbihan", "57": "Moselle", "58": "Nièvre", "59": "Nord", "60": "Oise",
    "61": "Orne", "62": "Pas-de-Calais", "63": "Puy-de-Dôme", "64": "Pyrénées-Atlantiques",
    "65": "Hautes-Pyrénées", "66": "Pyrénées-Orientales", "67": "Bas-Rhin",
    "68": "Haut-Rhin", "69": "Rhône", "70": "Haute-Saône", "71": "Saône-et-Loire",
    "72": "Sarthe", "73": "Savoie", "74": "Haute-Savoie", "75": "Paris",
    "76": "Seine-Maritime", "77": "Seine-et-Marne", "78": "Yvelines", "79": "Deux-Sèvres",
    "80": "Somme", "81": "Tarn", "82": "Tarn-et-Garonne", "83": "Var", "84": "Vaucluse",
    "85": "Vendée", "86": "Vienne", "87": "Haute-Vienne", "88": "Vosges", "89": "Yonne",
    "90": "Territoire de Belfort", "91": "Essonne", "92": "Hauts-de-Seine",
    "93": "Seine-Saint-Denis", "94": "Val-de-Marne", "95": "Val-d'Oise",
    "971": "Guadeloupe", "972": "Martinique", "973": "Guyane", "974": "La Réunion",
    "976": "Mayotte"
}

df.insert(
    1,  # position (0 = première colonne, 1 = deuxième, etc.)
    "nom_departement",  # nom de la nouvelle colonne
    df["GEO"].map(departement_noms)
)

#changer le nom de la colonne GEO en DEP pour la fusion
col = df.columns.tolist()
col[2] = "DEP"
df.columns = col

# on exclut les DOM TOM
df = df.loc[df['DEP'].str.len() == 2]

# la corse est codée 20 dans l'autre fichier, on modifie
df['DEP'] = df['DEP'].replace(['2A', '2B'],['20', '20'])

# la variable "TOUR_RESID" donne l'origine du touriste : on filtre sur total (on ne distingue pas pour l'instant)
df = df.loc[df['TOUR_RESID'].isin(["_T"])]

print(df["ACTIVITY"].value_counts(dropna=False))
# Aucun camping n'est présent dans notre sélection

print(df.head(20))

# on somme les arrivées par année et mois
df = df.groupby(['AAAA','MM', 'DEP'])["OBS_VALUE_CORR"].sum()

# on remet année et mois (devenues index) en variables normales
df = df.reset_index() 

base_touri = df

# A CHANGER 
#base_touri.to_csv("~/Project/Data/data_tourisme.csv")
