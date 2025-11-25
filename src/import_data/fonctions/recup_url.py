import pandas as pd
import numpy as np
import requests 
import gzip
import zipfile
import io
import os


def url_to_df(url, cols_a_conserver, type_zip, plusieurs_fichiers:bool):
    if type_zip == "gz":
        file = gzip.open(io.BytesIO(requests.get(url).content))
    elif type_zip == "zip":
        file=zipfile.ZipFile(io.BytesIO(requests.get(url).content))
    if plusieurs_fichiers==True:
        file.extractall("donnees_zip")
        fichiers = [f for f in os.listdir("donnees_zip") if f.endswith(".csv")]
        file = os.path.join("donnees_zip", fichiers[1])
    df=pd.read_csv(file, sep=";", usecols=cols_a_conserver)
    return(df)



# url = "https://object.files.data.gouv.fr/meteofrance/data/synchro_ftp/BASE/MENS/MENSQ_" + DEP + "_previous-1950-2023.csv.gz"



# url = "https://www.data.gouv.fr/api/1/datasets/r/1129fd80-2564-452c-86d4-9e36e7cca4a5"
# dossier = requests.get(url)

# # Ouvrir le contenu ZIP en mémoire
# z = zipfile.ZipFile(io.BytesIO(dossier.content))

# # Extraire tous les fichiers
# z.extractall("donnees_zip")


# fichiers = [f for f in os.listdir("donnees_zip") if f.endswith(".csv")]
# df = pd.read_csv(os.path.join("donnees_zip", fichiers[1]), sep = ';', usecols=colonnes_a_conserver)
