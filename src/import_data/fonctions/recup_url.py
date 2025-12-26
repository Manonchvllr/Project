import pandas as pd
import numpy as np
import requests 
import gzip
import zipfile
import io
import os


def url_to_df(url, cols_a_conserver, type_zip, plusieurs_fichiers: bool):
    # Cas fichier .gz (Météo France : un seul fichier CSV compressé)
    if type_zip == "gz":
        file = gzip.open(io.BytesIO(requests.get(url).content))
        df = pd.read_csv(file, sep=";", usecols=cols_a_conserver)
        return df

    # Cas ZIP (data.gouv : plusieurs CSV possibles dans l'archive)
    elif type_zip == "zip":
        z = zipfile.ZipFile(io.BytesIO(requests.get(url).content))

        # Cas où on ne veut qu'un seul fichier : on prend le premier CSV
        if not plusieurs_fichiers:
            noms_csv = [n for n in z.namelist() if n.endswith(".csv")]
            if not noms_csv:
                raise ValueError("Aucun fichier CSV trouvé dans le ZIP.")
            with z.open(noms_csv[0]) as f:
                df = pd.read_csv(f, sep=";", usecols=cols_a_conserver)
            return df

        # plusieurs_fichiers == True : on cherche tous les CSV compatibles
        frames = []
        for name in z.namelist():
            if not name.endswith(".csv"):
                continue
            with z.open(name) as f:
                df_tmp = pd.read_csv(f, sep=";")
            # On garde uniquement les fichiers qui ont bien toutes les colonnes demandées
            if not set(cols_a_conserver).issubset(df_tmp.columns):
                # fichier de métadonnées ou autre structure → on ignore
                continue
            frames.append(df_tmp[cols_a_conserver])

        if not frames:
            raise ValueError(
                "Aucun CSV du ZIP ne contient toutes les colonnes demandées : "
                f"{cols_a_conserver}"
            )

        df = pd.concat(frames, ignore_index=True)
        return df

    else:
        raise ValueError(f"type_zip inconnu : {type_zip}")



# url = "https://object.files.data.gouv.fr/meteofrance/data/synchro_ftp/BASE/MENS/MENSQ_" + DEP + "_previous-1950-2023.csv.gz"



# url = "https://www.data.gouv.fr/api/1/datasets/r/1129fd80-2564-452c-86d4-9e36e7cca4a5"
# dossier = requests.get(url)

# # Ouvrir le contenu ZIP en mémoire
# z = zipfile.ZipFile(io.BytesIO(dossier.content))

# # Extraire tous les fichiers
# z.extractall("donnees_zip")


# fichiers = [f for f in os.listdir("donnees_zip") if f.endswith(".csv")]
# df = pd.read_csv(os.path.join("donnees_zip", fichiers[1]), sep = ';', usecols=colonnes_a_conserver)
