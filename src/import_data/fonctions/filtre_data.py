# on sélectionne nos mois et années d'intérêt
def filtre_annee_mois(df):
    df = df[df.AAAA.isin(list(range(2011, 2023)))]
    df = df[df.MM.isin([1,2,3,6,7,8,9,12])]
    return(df)