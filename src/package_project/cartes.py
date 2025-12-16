from package_project import indicateurs_climat
import geopandas as gpd
from cartiflette import carti_download
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

def remove_leading_zeros(num):
    return num.lstrip('0')



def donnee_carte(data):
# Charger les départements français depuis une source publique
    france = carti_download(
        values = ["France"],
        crs = 4326,
        borders = "DEPARTEMENT",
        vectorfile_format="geojson",
        simplification=50,
        filter_by="FRANCE_ENTIERE",
        source="EXPRESS-COG-CARTO-TERRITOIRE",
        year=2022)
    data["DEP"] = data.DEP.astype(str)
    france["INSEE_DEP"] = france.INSEE_DEP.astype(str)
    france["INSEE_DEP"] = france["INSEE_DEP"].str.lstrip('0').fillna(value='0')
    carte = france.merge(data, left_on="INSEE_DEP", right_on="DEP", how="inner")
    return(carte)



def mise_en_forme_carte(data, annees, mois, indicateur, titre_carte, plotting, evolution):

    if evolution == False:
        titre_axe="Nombre de jours"
        carte=donnee_carte(indicateurs_climat.nbj_par_an(data, annees, mois, indicateur))

        indic=indicateur
    elif evolution == True:
        titre_axe="Taux de variation"
        if all(item in [6,7,8,9] for item in mois):
            saison = ["été"]         #  rouge
        elif all(item in [1,2,3,12] for item in mois):
            saison= ['hiver']
        indic= "evol_2015"
        carte=donnee_carte(indicateurs_climat.nbj_evol_2015(data, saison, indicateur))
        cols="bwr"          # bleu blanc rouge

    if carte[indic].min() < 0:
        center = 0
        cols="bwr"
    else:
        center = (carte[indic].max() - carte[indic].min())/2
        if all(item in [6,7,8,9] for item in mois):
            cols='OrRd'         #  rouge
        elif all(item in [1,2,3,12] for item in mois):
            cols='Blues'

    norme = TwoSlopeNorm(vmin=carte[indic].min(),
            vmax=carte[indic].max(),
            vcenter=center) 
    
    carte.plot(
         column=indic,
         cmap=cols,
         norm=norme,           # <-- centrage sur 0
         linewidth=0.8,
         ax=plotting,
         edgecolor="0.8",
         legend=True,
         legend_kwds={
             "shrink": 0.3,
             "label": titre_axe,
             "orientation": "vertical"
         }
     )
    
    plotting.set_title(titre_carte, fontsize=15)
    plotting.axis("off")