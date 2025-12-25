from src.package_project import indicateurs_climat
import geopandas as gpd
from cartiflette import carti_download
import matplotlib.pyplot as plt
import matplotlib as mcolors
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

# Define your custom color palette
col_palette_mixte = [
   '#154D71', # last color
   '#FFF0C4',
   '#8C1007'  # first color
]

# Create a ListedColormap
custom_cmap_mixte = LinearSegmentedColormap.from_list("custom_gradient", col_palette_mixte)

# fonctions
def hex_to_rgb_norm(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4))

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
    france["INSEE_DEP"] = france.INSEE_DEP.astype(str)
    #france["INSEE_DEP"] = france["INSEE_DEP"].str.lstrip('0').fillna(value='0')
    carte = france.merge(data, left_on="INSEE_DEP", right_on="DEP", how="inner")
    return(carte)



def mise_en_forme_carte(carte_prete, annees, mois, indicateur, titre_carte, plotting, evolution):

    if evolution == False:
        titre_axe="Nombre de jours"
        carte=carte_prete

        indic=indicateur
    elif evolution == True:
        titre_axe="Taux de variation"
        if all(item in [6,7,8,9] for item in mois):
            saison = ["été"]         #  rouge
        elif all(item in [1,2,3,12] for item in mois):
            saison= ['hiver']
        indic= "evol_2015"
        carte=carte_prete
        cols=custom_cmap_mixte         # bleu blanc rouge

    if carte[indic].min() < 0:
        center = 0
        cols=custom_cmap_mixte
    else:
        center = (carte[indic].max() - carte[indic].min())/2
        if all(item in [6,7,8,9] for item in mois):
            cols='OrRd'      #  rouge
        elif all(item in [1,2,3,12] for item in mois):
            cols='Purples'

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