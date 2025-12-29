#######################################################################################################################
# IMPORTATION DES PACKAGES
######################################################################################################################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import time
import warnings
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ardl import ARDL
from IPython.display import clear_output
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_breusch_godfrey as BG
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.diagnostic import het_breuschpagan as BP
from statsmodels.tsa.stattools import grangercausalitytests as GC
from statsmodels.tsa.api import UECM
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# lien pour une ressource presentant les ARDL
# https://fr.scribd.com/presentation/458866122/ARDL

warnings.filterwarnings("ignore")

#######################################################################################################################
# Fonction pour les données
######################################################################################################################
def Donne(dep, path="base.csv", verbose=True):
    """
    OBJECTIF
    --------
    Préparer une base de données mensuelle propre et exploitable
    pour l’analyse économétrique d’un département donné.

    ÉTAPES MÉTHODOLOGIQUES
    ---------------------
    1. Filtrage départemental
    2. Construction d’un index temporel mensuel
    3. Restriction de l’échantillon avant 2020 (crise sanitaire)
    4. Mise à fréquence mensuelle explicite
    5. Interpolation temporelle des mois manquants
    6. Transformation logarithmique du flux touristique

    PARAMÈTRES
    ----------
    dep : int ou str
        Numéro du département
    path : str
        Chemin vers le fichier CSV
    verbose : bool
        Affichage des informations descriptives

    SORTIES
    -------
    Departement : DataFrame
        Série temporelle mensuelle prête pour l’analyse
    donne_pres : dict
        Dictionnaire des mois observés par année
    """

    # =====================================================
    # 0. Chargement et filtrage
    # =====================================================
    base = pd.read_csv(path)

    dep = str(dep)
    dep = dep.zfill(2)
    data = base.loc[
        base["DEP"] == dep,
        ["AAAA", "MM", "TM", "OBS_VALUE_CORR", "NBJTX30", "NBJNEIG"]
    ].copy()

    # =====================================================
    # 1. Information sur la présence des données
    # =====================================================
    donne_pres = (
        data.groupby("AAAA")["MM"]
        .apply(lambda x: sorted(int(m) for m in x.unique()))
        .to_dict()
    )

    # =====================================================
    # 2. Construction de la date mensuelle
    # =====================================================
    data["Date"] = pd.to_datetime(
        data["AAAA"].astype(str) + "-" + data["MM"].astype(str)
    )

    data = (
        data
        .groupby("Date")[["TM", "NBJTX30", "NBJNEIG", "OBS_VALUE_CORR"]]
        .mean()
        .sort_index()
    )
    data1 = data.copy() # Garder l'enssembles des données pour une comparaison avec la prediction
    # =====================================================
    # 3. Restriction avant la crise sanitaire
    # =====================================================
    data = data.loc[data.index < "2020-01-01"]

    # =====================================================
    # 4. Mise à fréquence mensuelle explicite
    # =====================================================
    data = data.asfreq("MS")
    data1 = data1.asfreq("MS")
    # =====================================================
    # 5. Interpolation temporelle
    # =====================================================
    data = data.interpolate(method="time")
    data1 = data1.interpolate(method="time")
    # =====================================================
    # 6. Transformation logarithmique
    # =====================================================
    data["OBS_VALUE_CORR"] = np.log(data["OBS_VALUE_CORR"])
    data1["OBS_VALUE_CORR"] = np.log(data1["OBS_VALUE_CORR"])

    return data, donne_pres , data1

##########################################################################################################
# Graphique personnaliser pour l'evolution des series 
#########################################################################################################
def Graphique(variable, Departement):
    """
    DESCRIPTION: je trace la courbe d'evolution de la variable passer en argument
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(Departement[variable], label = f"{variable}")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"{variable}")
    ax.set_title(f"Evolution de {variable}")
    ax.legend()
    plt.show()

#######################################################################################################"
# CORRELOGRAMME
#######################################################################################################
def Correlogramme(variable, Departement):
    """
    DESCRIPTION: pour la variable passer en argument et en utilisant les Données du departement 
    je fais les graphes de correlogrammes afin d'apprecier...
    """
    var = Departement[f"{variable}"]
    if Departement is None:
        var = variable
        
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    plot_acf(
        var.dropna(),
        lags=12,
        ax=axes[0]
    )
    plot_pacf(
        var.dropna(),
        lags=12,
        ax=axes[1],
        method="ywm"
    )
    axes[0].set_title(f"ACF – {variable}")
    axes[1].set_title(f"PACF – {variable}")
    plt.tight_layout()
    plt.show()

#######################################################################################################"
# DESAISONALISATION
#######################################################################################################
def Desaisonnalisation(var, Departement, alpha=0.05, verbose=True):
    """
    OBJECTIF
    --------
    Identifier et corriger une éventuelle saisonnalité déterministe
    (mensuelle) d’une série temporelle avant les tests de stationnarité.

    APPROCHE THEORIQUE
    ------------------
    La saisonnalité est modélisée comme déterministe via des variables
    indicatrices mensuelles :

        y_t = μ + Σ δ_m D_{m,t} + ε_t

    Si au moins un coefficient saisonnier est significatif au seuil α,
    la série est désaisonnalisée à l’aide des résidus du modèle.

    PARAMÈTRES
    ----------
    var : str
        Nom de la variable à analyser
    Departement : DataFrame
        Données temporelles avec index de type datetime mensuel
    alpha : float
        Seuil de significativité statistique
    verbose : bool
        Affichage détaillé des résultats

    SORTIES
    -------
    serie_corrigee : Series
        Série désaisonnalisée ou brute
    statut : str
        Indication qualitative sur la transformation appliquée
    """

    # =====================================================
    # 0. Préparation (sans modifier le DataFrame original)
    # =====================================================
    data = Departement.copy()
    data["month"] = data.index.month.astype("category")

    # =====================================================
    # 1. Régression sur dummies mensuelles
    # =====================================================
    modele = sm.OLS.from_formula(
        f"{var} ~ C(month)",
        data=data
    ).fit(cov_type="HC1")

    if verbose:
        print("===================================")
        print("TEST DE SAISONNALITÉ DÉTERMINISTE")
        print("===================================")
        print(modele.summary())

    # =====================================================
    # 2. Test de significativité conjointe
    # =====================================================
    pvals = modele.pvalues[modele.model.exog_names[1:]]
    saisonnalite = (pvals < alpha).any()

    # =====================================================
    # 3. Décision et transformation
    # =====================================================
    if saisonnalite:

        if verbose:
            print("-----------------------------------")
            print("CONCLUSION")
            print("Présence d’une saisonnalité déterministe significative.")
            print("La série est désaisonnalisée via les résidus du modèle.")
            print("-----------------------------------")

        serie_corrigee = modele.resid
        statut = "Série corrigée de la saisonnalité déterministe"

    else:

        if verbose:
            print("-----------------------------------")
            print("CONCLUSION")
            print("Aucune saisonnalité déterministe détectée.")
            print("La série est conservée en niveau.")
            print("-----------------------------------")

        serie_corrigee = data[var]
        statut = "Série non transformée (pas de saisonnalité)"

    return serie_corrigee, statut
    

####################################################################################################
# TEST DE DICKEY FULLER: Stationnarite
####################################################################################################
def Dickey_fuller(
    var,
    Departement,
    regression="c",
    alpha=0.05,
    max_diff=5,
    verbose=True
):
    """
    OBJECTIF
    --------
    Tester la stationnarité d’une série temporelle via la procédure
    de Dickey-Fuller Augmentée (ADF) et déterminer son ordre d’intégration.

    APPROCHE ECONOMETRIQUE
    ---------------------
    - Désaisonnalisation préalable de la série
    - Test ADF avec sélection automatique du nombre de retards (AIC)
    - Différenciation itérative jusqu’à stationnarité

    PARAMÈTRES
    ----------
    var : str
        Nom de la variable à tester
    Departement : DataFrame
        Base de données temporelle
    regression : {"c", "ct", "n"}
        Spécification déterministe :
        - "c"  : constante
        - "ct" : constante + tendance
        - "n"  : aucun terme déterministe
    alpha : float
        Seuil de significativité
    max_diff : int
        Nombre maximal de différenciations
    verbose : bool
        Affichage détaillé des résultats intermédiaires

    SORTIES
    -------
    ordre_I : int
        Ordre d’intégration de la série
    statut  : str
        Indication qualitative de stationnarité
    """

    # =====================================================
    # 0. Préparation de la série
    # =====================================================
    serie, statut = Desaisonnalisation(var, Departement)
    nb_diff = 0

    if verbose:
        print("===================================")
        print(f"TEST ADF – Variable : {var}")
        print(f"Spécification déterministe : {regression}")
        print("===================================")

    # =====================================================
    # 1. Boucle de test ADF
    # =====================================================
    while nb_diff <= max_diff:

        resultat = adfuller(
            serie.dropna(),
            regression=regression,
            autolag="AIC"
        )

        adf_stat, p_value, lags, nobs = (
            resultat[0],
            resultat[1],
            resultat[2],
            resultat[3]
        )

        if verbose:
            print(f"D = {nb_diff} | ADF = {adf_stat:.3f} | p-value = {p_value:.4f}")

        # =================================================
        # 2. Décision statistique
        # =================================================
        if p_value < alpha:

            if verbose:
                print("-----------------------------------")
                print("CONCLUSION DU TEST ADF")
                print("-----------------------------------")
                print(f"Série stationnaire après {nb_diff} différenciation(s)")
                print(f"ADF statistic : {adf_stat:.3f}")
                print(f"Lags utilisés : {lags}")
                print(f"Observations  : {nobs}")
                print("-----------------------------------")

            return nb_diff, statut

        # =================================================
        # 3. Différenciation
        # =================================================
        serie = serie.diff()
        nb_diff += 1

    # =====================================================
    # 4. Cas non stationnaire
    # =====================================================
    if verbose:
        print("-----------------------------------")
        print("ATTENTION")
        print("La série reste non stationnaire")
        print(f"après {max_diff} différenciations.")
        print("-----------------------------------")

    return nb_diff, statut
###############################################################################################
# CALCULE DU POIDS DU CLIMAT DANS LA PREDICTION DU FLUX TOURISTIQUE
##############################################################################################
def modele(Departement, base):
    """
    OBJECTIF
    --------
    Estimer un modèle ARDL afin d’évaluer l’impact du changement climatique
    sur les flux touristiques départementaux.

    APPROCHE ECONOMETRIQUE
    ----------------------
    - Modèle ARDL saisonnier
    - Sélection du modèle par critères d’information (AIC, BIC)
    - Validation par diagnostic des résidus
    - Analyse de causalité et de cointégration (UECM, Bounds test)

    SORTIES
    -------
    - commentaire : interprétation économique finale
    - code        : diagnostic synthétique ('bon', 'mauvais', 'incertain')
    """

    # =====================================================
    # 0. Définition des variables
    # =====================================================
    y = Departement["OBS_VALUE_CORR"]  # Flux touristique
    X = Departement[["TM", "NBJTX30", "NBJNEIG"]]  # Variables climatiques

    print("# Sélection du modèle ARDL valide et optimal")

    # =====================================================
    # 1. Recherche du modèle valide
    # =====================================================
    pmax, qmax = 12, 12
    criteres = []

    for p in range(pmax + 1):
        for q in range(qmax + 1):
            try:
                ardl = ARDL(
                    endog=y,
                    lags=p,
                    exog=X,
                    order=q,
                    period=12,
                    seasonal=True,
                    trend="c",
                    missing="drop"
                )

                res = ardl.fit(cov_type="HC1")

                # Test de bruit blanc (Ljung-Box)
                lb = acorr_ljungbox(res.resid, lags=[12, 15, 18], return_df=True)

                if lb.lb_pvalue.iloc[0] > 0.05:
                    criteres.append((p, q, res.aic, res.bic))

            except Exception:
                continue

    # =====================================================
    # 2. Absence de modèle valide
    # =====================================================
    if len(criteres) == 0:
        commentaire = (
            "Aucun modèle ARDL valide n’a été identifié. "
            "Les propriétés statistiques des données ne permettent pas "
            "de conclure de manière fiable."
        )
        return commentaire, "pas de modèle"

    # =====================================================
    # 3. Sélection du modèle optimal (BIC)
    # =====================================================
    p, q, _, _ = min(criteres, key=lambda x: x[3])
    print(f"Modèle retenu : ARDL({p},{q})")

    ardl_final = ARDL(
        endog=y,
        lags=p,
        exog=X,
        order=q,
        period=12,
        seasonal=True,
        trend="c",
        missing="drop"
    )

    res = ardl_final.fit(cov_type="HC1")

    # =====================================================
    # 4. Présentation des résultats
    # =====================================================
    print("===================================")
    print("Résultats de l’estimation ARDL")
    print(res.summary())
    print("===================================")

    print("Diagnostic des résidus")
    res.plot_diagnostics(lags=12, figsize=(8, 6))
    plt.show()

    # =====================================================
    # 5. Analyse de causalité climatique
    # =====================================================
    exog_vars = [v for v in res.params.index if "TM" in v or "NBJ" in v]
    impact_climatique = (res.pvalues[exog_vars] < 0.05).any()

    commentaire = (
        "Les variables climatiques ne présentent pas d’effet statistiquement "
        "significatif sur les flux touristiques. Le tourisme départemental "
        "semble structurellement peu sensible au climat."
    )
    code = "mauvais"

    # =====================================================
    # 6. Test de cointégration si effet détecté
    # =====================================================
    if impact_climatique:

        ecm = UECM(
            endog=y,
            lags=max(p, 1),
            exog=X,
            order=max(q, 1),
            period=12,
            seasonal=True,
            trend="c",
            causal=True
        )

        ecm_res = ecm.fit()
        bounds = ecm_res.bounds_test(case=5)

        p_lower = bounds.p_values["lower"]
        p_upper = bounds.p_values["upper"]

        if p_upper < 0.05:
            commentaire = (
                "Il existe une relation de long terme entre le climat "
                "et les flux touristiques. Le changement climatique "
                "influence structurellement l’attractivité touristique."
            )
            code = "bon"

        elif p_lower > 0.05:
            commentaire = (
                "Aucune relation de long terme n’est identifiée. "
                "L’impact du climat est limité au court terme."
            )
            code = "mauvais"

        else:
            commentaire = (
                "Les résultats sont ambigus. La relation de long terme "
                "entre climat et tourisme ne peut être établie avec certitude."
            )
            code = "incertain"
    if code == "bon" :
        print("===================================")
        print("Rélation de long terme Analyse")
        print(ecm_res.summary())
        print(bounds)
        print("===================================")
        print("\nPrediction avec le modèle ARDL sur les 36 mois suivants")

        # preparation des données 
        horizon = 36
        X_future = base.loc[base.index[-horizon:], ["TM", "NBJTX30", "NBJNEIG"]]

        #Prediction
        y_pred = res.predict(
        start=Departement.index[-1],
        end=base.index[-1],
        exog_oos=X_future)

        # Intervalle de confiance
        sigma = res.resid.std()
        ic_inf = y_pred - 1.96 * sigma
        ic_sup = y_pred + 1.96 * sigma



        #Affichage de la prediction
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.plot(
            base.index,
            base["OBS_VALUE_CORR"],
            color="black",
            label="Observé"
        )
        
        ax.plot(
            y_pred.index,
            y_pred,
            color="red",
            label="Prévision (36 mois)"
        )
        
        ax.fill_between(
            y_pred.index,
            ic_inf,
            ic_sup,
            color="blue",
            alpha=0.2,
            label="IC 95 %"
        )
        
        ax.axvline(
            Departement.index[-1],
            linestyle="--",
            color="gray"
        )
        
        ax.set_title("Prévision du flux touristique (ARDL – exogènes observées)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Flux touristique (log)")
        ax.legend()
        
        plt.close()



        

    print(commentaire)
    return commentaire, code

#########################################################################################################
# REALISTION DE L'ENCHAINEMENT POUR CHAQUE DEPARTEMENT 
#########################################################################################################
def Test(departement):
    """
    ==========================================================================================
    ANALYSE ÉCONOMÉTRIQUE COMPLÈTE PAR DÉPARTEMENT
    ==========================================================================================

    OBJECTIF SCIENTIFIQUE
    ---------------------
    Cette fonction applique, pour un département donné, l’ensemble du protocole
    économétrique visant à analyser la relation entre le flux touristique et
    les variables climatiques.

    L’approche est volontairement séquentielle et rigoureuse :
        (i)   préparation et validation des données,
        (ii)  analyse de la saisonnalité et de la stationnarité,
        (iii) modélisation dynamique (ARDL / UECM),
        (iv)  conclusion économique.

    Ce schéma correspond aux standards de l’économétrie appliquée
    en séries temporelles (cf. Enders, 2015 ; Pesaran et al., 2001).

    PARAMÈTRE
    ---------
    departement : int
        Numéro du département (ex. 75, 91, etc.)

    SORTIES
    -------
    tuple :
        - code département
        - nom du département
        - information sur la présence des données
        - qualité des données (nombre de NaN)
        - statut et ordre d’intégration du flux touristique
        - statut et ordre d’intégration de TM
        - statut et ordre d’intégration de NBJTX30
        - statut et ordre d’intégration de NBJNEIG
        - conclusion économique
        - code synthétique de résultat ("bon", "mauvais", "incertain")

    REMARQUE
    --------
    Cette fonction constitue le cœur analytique du projet.
    Elle est conçue pour être appelée de manière itérative sur l’ensemble
    des départements français.
    """

    # ======================================================================================
    # 0) RÉFÉRENTIEL DES DÉPARTEMENTS
    # ======================================================================================
    # Justification :
    # La correspondance code ↔ nom est explicitée pour assurer
    # la traçabilité et la lisibilité des résultats agrégés.

    departements = {
        "01": "Ain","02": "Aisne","03": "Allier","04": "Alpes-de-Haute-Provence",
        "05": "Hautes-Alpes","06": "Alpes-Maritimes","07": "Ardèche",
        "08": "Ardennes","09": "Ariège","10": "Aube","11": "Aude",
        "12": "Aveyron","13": "Bouches-du-Rhône","14": "Calvados","15": "Cantal",
        "16": "Charente","17": "Charente-Maritime","18": "Cher","19": "Corrèze",
        "21": "Côte-d'Or","22": "Côtes-d'Armor","23": "Creuse","24": "Dordogne",
        "25": "Doubs","26": "Drôme","27": "Eure","28": "Eure-et-Loir",
        "29": "Finistère","30": "Gard","31": "Haute-Garonne","32": "Gers",
        "33": "Gironde","34": "Hérault","35": "Ille-et-Vilaine","36": "Indre",
        "37": "Indre-et-Loire","38": "Isère","39": "Jura","40": "Landes",
        "41": "Loir-et-Cher","42": "Loire","43": "Haute-Loire",
        "44": "Loire-Atlantique","45": "Loiret","46": "Lot",
        "47": "Lot-et-Garonne","48": "Lozère","49": "Maine-et-Loire",
        "50": "Manche","51": "Marne","52": "Haute-Marne","53": "Mayenne",
        "54": "Meurthe-et-Moselle","55": "Meuse","56": "Morbihan",
        "57": "Moselle","58": "Nièvre","59": "Nord","60": "Oise",
        "61": "Orne","62": "Pas-de-Calais","63": "Puy-de-Dôme",
        "64": "Pyrénées-Atlantiques","65": "Hautes-Pyrénées",
        "66": "Pyrénées-Orientales","67": "Bas-Rhin","68": "Haut-Rhin",
        "69": "Rhône","70": "Haute-Saône","71": "Saône-et-Loire",
        "72": "Sarthe","73": "Savoie","74": "Haute-Savoie",
        "75": "Paris","76": "Seine-Maritime","77": "Seine-et-Marne",
        "78": "Yvelines","79": "Deux-Sèvres","80": "Somme",
        "81": "Tarn","82": "Tarn-et-Garonne","83": "Var","84": "Vaucluse",
        "85": "Vendée","86": "Vienne","87": "Haute-Vienne",
        "88": "Vosges","89": "Yonne","90": "Territoire de Belfort",
        "91": "Essonne","92": "Hauts-de-Seine","93": "Seine-Saint-Denis",
        "94": "Val-de-Marne","95": "Val-d'Oise"
    }

    # Normalisation du code département
    dep_code = f"{departement:02d}"
    if dep_code not in departements:
        return (np.nan,) * 10

    nom_dep = departements[dep_code]

    print("=" * 35)
    print(f"DÉPARTEMENT : {nom_dep}")
    print("=" * 35)

    # ======================================================================================
    # 1) PRÉPARATION ET VALIDATION DES DONNÉES
    # ======================================================================================
    print("\n#1) Préparation des données")

    Departement, donne_pres, base = Donne(departement)

    # Justification :
    # Toute analyse économétrique repose sur des données complètes.
    # En présence de NaN résiduels, les résultats seraient biaisés ou invalides.
    qualite = Departement.isna().sum().sum()
    if qualite > 0:
        return departement, nom_dep, donne_pres, qualite, *(np.nan for _ in range(6))

    # ======================================================================================
    # 2) SAISONNALITÉ ET STATIONNARITÉ
    # ======================================================================================
    print("\n#2) Analyse de la saisonnalité et de la stationnarité")

    ordres_I = {}
    statuts = {}

    for var in ["OBS_VALUE_CORR", "TM", "NBJTX30", "NBJNEIG"]:

        print("\n" + "-" * 30)
        print(f"VARIABLE : {var}")
        print("-" * 30)
        
        # Analyse visuelle 
        Graphique(variable=var, Departement=Departement)
        Correlogramme(var, Departement)
        
        # Justification théorique :
        # Les tests de racine unitaire ne sont valides que
        # si la saisonnalité est correctement traitée en amont.
        ordre_I, statut = Dickey_fuller(var, Departement)

        ordres_I[var] = ordre_I
        statuts[var] = statut

    # ======================================================================================
    # 3) MODÉLISATION DYNAMIQUE
    # ======================================================================================
    
    keys = ["OBS_VALUE_CORR", "TM", "NBJTX30", "NBJNEIG"]
    condition = (ordres_I["OBS_VALUE_CORR"] < 2 )&(ordres_I["TM"] < 2 )&(ordres_I["NBJTX30"] < 2 )&(ordres_I["NBJNEIG"] < 2 )

    print("\n#3) Modélisation économétrique (ARDL / UECM)")
    if condition :
        print("\nLa condition necessaire pour pouvoire faire le modèle ARDL qui est que toutes le variables sont\nintegrées d'ordre inferieur à 1 est satisfait\n")
        conclusion, code = modele(Departement, base)
    else:
        print("\nLa modelisation ARDL n'est pas justifier")
        conclusion, code = "Nous ne pouvons rien conclu avec cette approche de modelisation", "pas_de_modele"
            

        
    # ======================================================================================
    # 5) SORTIE STRUCTURÉE DES RÉSULTATS
    # ======================================================================================
    return (
        departement,
        nom_dep,
        donne_pres,
        qualite,
        (statuts["OBS_VALUE_CORR"], ordres_I["OBS_VALUE_CORR"]),
        (statuts["TM"], ordres_I["TM"]),
        (statuts["NBJTX30"], ordres_I["NBJTX30"]),
        (statuts["NBJNEIG"], ordres_I["NBJNEIG"]),
        conclusion,
        code
    )
#########################################################################################################
# RESULTATS DE L'ETUDE
#########################################################################################################
def Resultat(nombre):
    """
    DESCRIPTION: Pour chaque departement je resume le resultats de l'etude dans une dataFrame
    """
    departement, nom_dep, donne_pres, qualite, OBS_VALUE_CORR, TM, NBJTX30, NBJNEIG, conclusion, code = [],[],[],[],[],[],[],[],[],[]
    for dep in range(1,nombre + 1):
        a, b, c, d, e, f, g, h, i, j= Test(dep)
        departement.append(a) 
        nom_dep.append(b) 
        donne_pres.append(c) 
        qualite.append(d) 
        OBS_VALUE_CORR.append(e) 
        TM.append(f) 
        NBJTX30.append(g) 
        NBJNEIG.append(h)
        conclusion.append(i)
        code.append(j)
        clear_output(wait=True)
    resultat = pd.DataFrame({"departement" : departement,
                             "nom_dep" : nom_dep,
                             "donne_pres" : donne_pres,
                             "qualite" : qualite,
                             "OBS_VALUE_CORR" : OBS_VALUE_CORR,
                             "TM" : TM,
                             "NBJTX30" : NBJTX30,
                             "NBJNEIG" : NBJNEIG,
                             "conclusion" : conclusion,
                             "code" : code})
    return resultat
