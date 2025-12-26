import pandas
import numpy as np

# repartition arrivees touristiques
def repartition_arrivees(data, annees, mois, par_groupe):
    data = pandas.DataFrame(data.loc[data['AAAA'].isin(annees)].loc[data['MM'].isin(mois)].loc[~data["DEP"].isin([75,77,78,91,92,93,94,95])].groupby(par_groupe)["OBS_VALUE_CORR"].sum())
    data = data.reset_index()
    data["part_tourisme"] = ((data.OBS_VALUE_CORR / data.OBS_VALUE_CORR.sum()))*100
    return(data)


# évolution avant et après 2015
def evol_arrivees(data, mois):
    data = pandas.DataFrame(data.loc[data['MM'].isin(mois)].groupby(["DEP", "periode", "AAAA"])["OBS_VALUE_CORR"].sum())
    data = data.groupby(["DEP","periode"]).mean()
    data = data.reset_index()
    data = data.pivot(index="DEP", columns="periode", values="OBS_VALUE_CORR")
    data = data.reset_index(names=["DEP"])
    data["evol_2015"] = np.where(data['avant_2015']!= 0, 
                                 (data.apres_2015 - data.avant_2015)/data.avant_2015, 
                                 np.nan)
    return(data)