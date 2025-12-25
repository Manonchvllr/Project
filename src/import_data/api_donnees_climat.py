# les données climatiques sont volumineuses (un fichier par département)
# pour ne pas les télécharger à la main, on crée un programme pour les importer automatiquement 
# en sélectionnant uniquement les variables d'intérêt

import pandas as pd
import numpy as np
from fonctions import recup_url, filtre_data

cols = ['NOM_USUEL', 
        "AAAAMM",
        "TM", 
        "TX",
        "RR", 
        "UMM", 
        "FFM", 
        "TXMIN", 
        "NBJTX0", 
        "NBJTX25", 
        "NBJTX30", 
        "NBJTX35",
        "NBJNEIG", 
        "NBJSOLNG"]

cols_indic = cols[2:len(cols)] 


# pour chaque département, on va procéder de la même façon
# on crée donc une fonction qui prend le département comme argument

def agreg_dpt(DEP):
    df_filtre = recup_url.url_to_df(url = "https://object.files.data.gouv.fr/meteofrance/data/synchro_ftp/BASE/MENS/MENSQ_" + DEP + "_previous-1950-2023.csv.gz",
                        cols_a_conserver=cols,
                        type_zip="gz",
                        plusieurs_fichiers=False)
# on crée une variable ne contenant que l'annee 
    df_filtre['AAAA'] = df_filtre['AAAAMM'].astype(str).str[:4].astype(int)
    df_filtre['MM']= df_filtre['AAAAMM'].astype(str).str[-2:].astype(int)
# on sélectionne nos mois et années d'intérêt
    df_filtre = filtre_data.filtre_annee_mois(df_filtre)
# on calcule la moyenne départementale pour toutes les variables
    df_filtre['TX_num']=df_filtre.TX.astype("float64")
    df_filtre['TXMIN_num']=df_filtre.TXMIN.astype("float64")
    df_filtre = df_filtre.groupby(['AAAA','MM'])[cols_indic].mean()
    df_filtre['DEP'] = DEP
    return(df_filtre)

# on prépare une boucle pour importer un à un les fichiers départements
# et concaténer les outputs de la fonction agreg_dpt
liste_dep = list(np.arange(2,96))
df = agreg_dpt("01")
for i in liste_dep:
    num = f'{i:02}'
    df = pd.concat([df, agreg_dpt(num)])

# on remet année et mois (devenues index) en variables normales
df = df.reset_index() 

df["DEP"] = df.DEP.astype(str)
df["DEP"] = df["DEP"].str.zfill(2)


# on crée deux departements différents pour la Corse pour pouvoir cartographier ensuite
new_rows = df.loc[df['DEP']=="20"].copy()
new_rows1 = new_rows.copy()
new_rows["DEP"] = "2A"
new_rows1["DEP"] = "2B"
df = df._append([new_rows,new_rows1])
df = df.loc[df["DEP"] != "20"]

# saison (été ou hiver)
conditions1 = [
    (df['MM'] <= 3) | (df['MM'] == 12),
    (df['MM'] >= 6) & (df['MM'] <= 9)
    ]

values1 = ['hiver', 'été']

df['saison'] = np.select(conditions1, values1, default='Other')

# période (avant ou après 2015)
conditions2 = [
    (df['AAAA'] <= 2015),
    (df['AAAA'] > 2015)
    ]

values2 = ['avant_2015', 'apres_2015']

df['periode'] = np.select(conditions2, values2, default='Other')

base_temp = df
base_temp.head()


# On veut ajouter la nouvelle base de données, que l'on vient de créer dans dossier Data
from pathlib import Path

# dossier racine du projet = 3 niveaux au-dessus de ce fichier
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    PROJECT_ROOT = Path.cwd().parents[1]

data_dir = PROJECT_ROOT / "Data"
data_dir.mkdir(exist_ok=True)

output_path = data_dir / "data_climat.csv"
base_temp.to_csv(output_path, index=False)

print("Fichier climat sauvegardé dans :", output_path)