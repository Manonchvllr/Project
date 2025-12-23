import pandas
import numpy as np

# moyenne temperature max quotidienne 
def temp_moy(data, annees, mois, var_temp):
    data = data.loc[data['AAAA'].isin(annees)].loc[data['MM'].isin(mois)].groupby("DEP")[var_temp].mean()
    return(data)

#  nombre de jours d'une caractéristique climatique (au dessus/ en dessous d'une certaine température, neige, sécheresse...) sur toute la période
def nbj_par_an(data, annees, mois, var_climat):
    data = pandas.DataFrame(data.loc[data['AAAA'].isin(annees)].loc[data['MM'].isin(mois)].groupby(["DEP","periode", "AAAA"])[var_climat].sum())
    data = data.reset_index() 
    return(data)

#  nombre moyen de jours au dessus/ en dessous d'une certaine température par année
def nbj_evol_2015(data, saison, var_climat):
    data = pandas.DataFrame(data.loc[data['saison'].isin([saison])].groupby(["DEP","periode"])[var_climat].mean())
    data = data.reset_index()
    data = data.pivot(index="DEP", columns="periode", values=var_climat)
    data = data.reset_index(names=["DEP"])
    # data["evol_2015"] = if data.iloc["avant_2015"==0](data.apres_2015 - data.avant_2015)/data.avant_2015
    data["evol_2015"] = np.where(data['avant_2015']!= 0, 
                                 (data.apres_2015 - data.avant_2015)/data.avant_2015, 
                                 np.nan)
    return(data)


