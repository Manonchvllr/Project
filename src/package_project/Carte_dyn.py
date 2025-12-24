##################################################################################################################"
# Importation des packages
##############################################################################################################"###

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
import pandas as pd
import geopandas as gpd
import time
from IPython.display import HTML, display
from IPython.display import Image
from shapely.errors import TopologicalError



###############################################################################################################
# Importation des donnnée
###############################################################################################################
def Base_carte():
    base = pd.read_csv("base.csv")
    base = base.groupby(["DEP", "AAAA", "saison"])["TM"].mean().reset_index()
    base["DEP"] = base["DEP"].astype(str).str.zfill(2)
    return base



##############################################################################################################
# Chargement de la carte et verification
##############################################################################################################
def carte():
    url = "https://france-geojson.gregoiredavid.fr/repo/departements.geojson"
    gdf = gpd.read_file(url).rename(columns={"code": "DEP"}).to_crs(epsg=3857)
    gdf["DEP"] = gdf["DEP"].astype(str).str.zfill(2)
    return gdf


def clean_geometry(g):
    try:
        return g.buffer(0)
    except TopologicalError:
        print("il y a un probleme")
        return None

def verification():
    """
    Description:
    """
    gdf = carte()
    gdf1 = gdf
    gdf["geometry"] = gdf["geometry"].apply(clean_geometry)
    
    # détection des géométries invalides
    n_invalid = (~gdf.geometry.is_valid | gdf.geometry.isna()).sum()
    if n_invalid > 0:
        print(f"{n_invalid} géométries invalides supprimées")
        
    gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid]
    gdf["DEP"] = gdf["DEP"].astype(str).str.zfill(2)
    return gdf


######################################################################################################################
# creaction du gif
#######################################################################################################################

def gif(saison):
    """
    Description:
    """

    # definition du nom du gif
    nom_fichier = f"cart_pour_les {saison}.gif"

    # chargement des données et filtrage selon la saison
    base = Base_carte()
    gdf = carte()
    base = base[base["saison"] == saison]

    # Frames 
    frames = (
        base[["AAAA", "saison"]]
        .drop_duplicates()
        .sort_values("AAAA")
        .itertuples(index=False, name=None))
    frames = list(frames)

    # Pré-merge
    merged_frames = {}
    for anne, moi in frames:
        df = base.loc[
            (base["AAAA"] == anne) & (base["saison"] == moi),
            ["DEP", "TM"]]
        
        merged = gdf.merge(df, on="DEP", how="inner")
        merged_frames[(anne, moi)] = merged if not merged.empty else None

    # Figure
    fig, ax = plt.subplots(figsize=(10, 10), facecolor="white", layout="constrained")
    ax.axis("off")

    # dimenssionnemt des cartes
    minx, miny, maxx, maxy = gdf.total_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal")

    # Uniformisation des col bars
    vmin, vmax = base["TM"].min(), base["TM"].max()
    cmap = mpl.cm.get_cmap("coolwarm")
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04).set_label("Température (°C)")

    # Fonction update locale
    def update(i):
        anne, moi = frames[i]
    
        ax.clear()
        ax.axis("off")
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_aspect("equal")
    
        ax.set_title(f"Température — Année {anne}, Saison {moi}", fontsize=15)
    
        carte = merged_frames.get((anne, moi))
        if carte is None:
            return []
    
        carte.plot(
            column="TM",
            cmap=cmap,
            norm=norm,
            linewidth=0.6,
            ax=ax,
            edgecolor="0.6",
            legend=False
        )
    
        # Labels centrés
        for _, row in carte.iterrows():
            c = row.geometry.centroid
            ax.text(
                c.x, c.y,
                str(row.get("nom", "")),
                fontsize=7,
                ha="center",
                va="center",
                color="black"
            )
    
        return ax.collections + ax.texts



    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=6000,
        blit=False)

    ani.save(nom_fichier, writer="pillow", fps=1)

    plt.close(fig)  # évite l'affichage statique de la figure

    time.sleep(2)
    Image(filename = nom_fichier)


    




