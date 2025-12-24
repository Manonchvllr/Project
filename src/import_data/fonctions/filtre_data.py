# on sélectionne nos mois et années d'intérêt
def filtre_annee_mois(df):
    df = df[df.AAAA.isin(list(range(2011, 2023)))]
    df = df[df.MM.isin([x for x in range(1,13)])]
    return(df)