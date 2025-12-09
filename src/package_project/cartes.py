import geopandas as gpd
from cartiflette import carti_download
import matplotlib.pyplot as plt

def carte(data, indicateur, titre_axe, titre_carte):
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
    carte = france.merge(data, left_on="INSEE_DEP", right_on="DEP", how="inner")
    # Afficher la carte avec les noms des départements
    fig, ax = plt.subplots(figsize=(10, 10))
    carte.plot(column=indicateur, cmap='OrRd', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True,
            legend_kwds={
            "shrink": 0.5,       # réduit la taille de la barre
            "label": titre_axe,
            "orientation": "vertical"
        })
    # # Ajouter les noms des départements
    for idx, row in carte.iterrows():
        centroid = row['geometry'].centroid
        ax.text(centroid.x, centroid.y, row['LIBELLE_DEPARTEMENT'], fontsize=8, ha='center', va='center')
        ax.set_title(titre_carte, fontsize=15)
        ax.axis('off')
    plt.tight_layout()
    plt.show()