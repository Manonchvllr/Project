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
def Donne(dep):
    """
    DESCRIPTION: je prends les données que j'ai enregistrer dans base,
    je filtre ces données pour un departement precis pour lequel je vais realiser le test de correlation
    je limite l'etude à avant 2020 à cause de la crise sanitaire de 2020
    je prends le logaritme du nombre de touristes arrivant dans le departement pour ramener dans des echelles comparables 
    """
    base = pd.read_csv("base.csv")
    Departement = base.loc[(base.DEP == dep),["AAAA","MM", "TM", "OBS_VALUE_CORR", "NBJTX30","NBJNEIG"]]
    donne = Departement.groupby("AAAA")["MM"].apply(lambda x: [int(m) for m in sorted(x.unique())]).to_dict()
    
    Departement["Date"] = Departement.AAAA.astype("str") + "-"  + Departement.MM.astype("str")
    Departement = Departement.groupby("Date")[["TM","NBJTX30","NBJNEIG",  "OBS_VALUE_CORR"]].mean().reset_index()
    Departement = Departement.drop(["AAAA", "MM"], axis = 1, errors = "ignore")
    Departement["Date"] = pd.to_datetime(Departement["Date"])
    Departement = Departement[Departement["Date"] < pd.Timestamp("2020-01-01")]
    Departement = Departement.set_index("Date")
    Departement = Departement.asfreq("MS")
    Departement = Departement.interpolate(method="time")
    
    Departement.OBS_VALUE_CORR = np.log(Departement.OBS_VALUE_CORR)
    for x in donne:
        print(x, donne[x])
    print("le manque de donnes en 2020 et 2021 s'explique par la crise sanitaire de 2020 ainsi nous allons reinstraindre notre analyse sur les periodes avant 2020.\nPour les mois dont les donnes sont manquantes nous avons interpoler")
    return Departement, donne

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
def Desaisonnalisation(var , Departement):
    """
    DESCRIPTION:
    """
    Departement["month"] = Departement.index.month.astype("category")
    modele = sm.OLS.from_formula(
        f"{var} ~ C(month)",
        data=Departement
    ).fit(cov_type="HC1")
    print(modele.summary())
    desaisonnalise = (modele.pvalues[modele.model.exog_names[1:]] < 0.05).any()
    if desaisonnalise:
        print("Après avoir observer la significativité des coeficient saisonniers nous concluons à une saisonalité et devons corriger cette saisonnalité pour le test de stationnarité")
        x = modele.resid
        y = "Serie corriger des variation Saisonnière"
        return x, y
        
    x = Departement[var]
    print("Les coeficients saisonniers ne sont pas signifivatifs donc nous concluons à une une abscence de saisonnalité")
    y = "Serie non transformer"
    return x, y
    

####################################################################################################
# TEST DE DICKEY FULLER: Stationnarite
####################################################################################################
def Dickey_fuller(var, Departement, tendance = None, alpha=0.05, max_diff=5, verbose=True):
    """
    DESCRIPTION: je réalise le test de stationnarité selon la procedure de Dickey-Fuller
    """
    serie, y = Desaisonnalisation(var , Departement)
    nb_diff = 0

    while nb_diff <= max_diff:
        resultat = adfuller(
            serie,
            regression=tendance,
            autolag="AIC"
        )
        
        p_value = resultat[1]

        if verbose:
            print(f"D = {nb_diff} | p-value = {p_value:.4f}")

        if p_value < alpha:
            print("ADF Statistic :", resultat[0])
            print("p-value       :", resultat[1])
            print("Lags utilisés :", resultat[2])
            print("Nb obs        :", resultat[3])
            print(f"status : stationnaire à l'ordre {nb_diff}")
            return nb_diff, y

        # différenciation
        serie = serie.diff().dropna()
        nb_diff = nb_diff + 1

    return nb_diff, y
        
###############################################################################################
# CALCULE DU POIDS DU CLIMAT DANS LA PREDICTION DU FLUX TOURISTIQUE
##############################################################################################
def modele(Departement):
    """
    DESCRIPTION:je vais estimer le modele econometrique afin d'aporter une reponse à la question
    j'utilise le modele ARDL : Y(t) = µ + a1*Y(t-1) + ... + ap*Y(t-p) + b1'X(t) + ... + bq'X(t-q) + e,
    e~BB(0, sd) et on se focalise sur les b:
    si statistiquement different de 0 alors on fait un test de causalite au sens de granger
    sinon on conclut abscence de liaison entre rechaufement climatique et flux touristique
    
    le modèle est valide que si les erreur suivent un bruit blanc que nous testons avec le test de 
    Breuch-Golfrey
    """
    #0) Declaration des variables 
    y = Departement["OBS_VALUE_CORR"]
    X = Departement[["TM","NBJTX30","NBJNEIG"]]   # variables exogènes

    print("#Estimation et choix des modèles le plus Vraisemblabe parmit les modèle valide")
    pmax, qmax = 7, 4
    AIC = []
    BIC = []
    p_q =[]
    for p in range(pmax + 1):
        for q in range(qmax + 1):
            
            try:
                #Specification
                modele_ardl = ARDL(
                    endog=y,
                    lags=p,
                    exog=X,
                    order=q,
                    period = 12,
                    seasonal = True,
                    trend="c",
                    missing="drop"
                )
                # Estimation
                resultats = modele_ardl.fit(cov_type="HC1")
    
                #Validation des residus
                lb = acorr_ljungbox(resultats.resid,lags=[12],return_df=True)
                if lb.lb_pvalue.iloc[0] > 0.05:
                    AIC.append(resultats.aic)
                    BIC.append(resultats.bic)
                    p_q.append((p,q))
                
            except Exception: 
                continue

    #modele le plus parcimonieux
    if not BIC: 
        print("pas de modèle")
        return "Aucun modèle n'est valide Ainsi nous ne pouvons pas baser nos conclusion sur ces modèles", "pas de modèle"
        
    (p,q) = p_q[BIC.index(min(BIC))]
    modele_ardl = ARDL(
                endog=y,
                lags=p,
                exog=X,
                order=q,
                period = 12,
                seasonal = True,
                trend="c",
                missing="drop"
            )
    print(f"le meilleur modèle est ARDL({p,q})")
    # Estimation
    resultats = modele_ardl.fit(cov_type="HC1")
    print("===================================")
    print("Resultats du modèle estimer")
    print(resultats.summary())
    print("===================================")
    print("Dignostique des residus pour la validité")
    resultats.plot_diagnostics(lags = 12, figsize = (8,6))
    commentaire1 = """
        Le flux touristique ne depend pas de changement climatique, les touristes sont indifferents du rechauffement climatique 
        quant à leur decision de la destination touristique
        """
    code = "mauvais"
    exog_names = [n for n in resultats.params.index if "TM" in n or "NBJ" in n]
    causalite = (resultats.pvalues[exog_names] < 0.05).any()
    if causalite :
        if p == 0:
            p = 1
        if q == 0:
            q = 1
        ecm = UECM(
            endog=y,
            lags=p,
            exog=X,
            order=q,
            period = 12,
            seasonal = True,
            causal = True,
            trend="c")
        seasonal_ecm_res = ecm.fit()
        _ = seasonal_ecm_res.ci_resids.plot(title="Cointegrating Error with Seasonality")
        plt.show()
        print(seasonal_ecm_res.ci_summary())
        bounds_test = seasonal_ecm_res.bounds_test(case=5)
        p_lower = bounds_test.p_values["lower"]
        p_upper = bounds_test.p_values["upper"]
        if p_lower > 0.05:
            commentaire1 = """pas le cointegration: Il n'y a pas de relation de long terme entre le flux touristique et le rechauffement climatique mais une relation de court terme"""
        elif p_upper < 0.05:
            commentaire1 = """Le rechauffement climatique impact reellement le flux touristique: Les touristes tiennent compte du climat lors de la prise de leur decision concernant leur destionation"""
            code = "bon"
        else:
            commentaire1 = "Nous ne pouvons pas conclure quant à une relation de longterme"
            code = "incertaint"
    print(commentaire1)
    return commentaire1, code

#########################################################################################################
# REALISTION DE L'ENCHAINEMENT POUR CHAQUE DEPARTEMENT 
#########################################################################################################
def Test(departement):
    """
    DESCRIPTION:
    """
    departements = {
    "01": "Ain","02": "Aisne","03": "Allier","04": "Alpes-de-Haute-Provence","05": "Hautes-Alpes","06": "Alpes-Maritimes","07": "Ardèche",
    "08": "Ardennes","09": "Ariège","10": "Aube","11": "Aude","12": "Aveyron","13": "Bouches-du-Rhône","14": "Calvados","15": "Cantal",
    "16": "Charente","17": "Charente-Maritime","18": "Cher","19": "Corrèze","21": "Côte-d'Or","22": "Côtes-d'Armor","23": "Creuse","24": "Dordogne",
    "25": "Doubs","26": "Drôme","27": "Eure","28": "Eure-et-Loir","29": "Finistère","30": "Gard","31": "Haute-Garonne","32": "Gers","33": "Gironde",
    "34": "Hérault","35": "Ille-et-Vilaine","36": "Indre","37": "Indre-et-Loire","38": "Isère","39": "Jura","40": "Landes","41": "Loir-et-Cher",
    "42": "Loire","43": "Haute-Loire","44": "Loire-Atlantique","45": "Loiret","46": "Lot","47": "Lot-et-Garonne","48": "Lozère","49": "Maine-et-Loire",
    "50": "Manche","51": "Marne","52": "Haute-Marne","53": "Mayenne","54": "Meurthe-et-Moselle","55": "Meuse","56": "Morbihan","57": "Moselle",
    "58": "Nièvre","59": "Nord","60": "Oise","61": "Orne","62": "Pas-de-Calais","63": "Puy-de-Dôme","64": "Pyrénées-Atlantiques","65": "Hautes-Pyrénées",
    "66": "Pyrénées-Orientales","67": "Bas-Rhin","68": "Haut-Rhin","69": "Rhône","70": "Haute-Saône","71": "Saône-et-Loire","72": "Sarthe",
    "73": "Savoie","74": "Haute-Savoie","75": "Paris","76": "Seine-Maritime","77": "Seine-et-Marne","78": "Yvelines","79": "Deux-Sèvres","80": "Somme",
    "81": "Tarn","82": "Tarn-et-Garonne","83": "Var","84": "Vaucluse","85": "Vendée","86": "Vienne","87": "Haute-Vienne","88": "Vosges","89": "Yonne",
    "90": "Territoire de Belfort","91": "Essonne","92": "Hauts-de-Seine","93": "Seine-Saint-Denis","94": "Val-de-Marne","95": "Val-d'Oise"}

    if departement < 10:
        x = f"0{departement}"
        if x not in departements.keys():
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        nom_dep = departements[f"0{departement}"]
    else:
        x = f"{departement}"
        if x not in departements.keys():
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        nom_dep = departements[f"{departement}"]
    print("===================================")
    print(nom_dep)
    print("===================================")
    print("#0) Données")
    Departement, donne_pres = Donne(departement)
    qualite = Departement.isna().sum().sum()
    if qualite > 0:
        return departement, nom_dep, donne_pres, qualite, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    print("===================================")
    print("#1) Saisonnalité et Stationnarite")
    ordres_I = {}
    statuts = {}
    for var in ["OBS_VALUE_CORR","TM","NBJTX30","NBJNEIG"]:
        ordre_I, statut = Dickey_fuller(var, Departement)
        ordres_I[var] = ordre_I
        statuts[var] = statut
        print()
        print("===================================")
        print(f"==========={var}==================")
        print("##a) Test visuel")
        Graphique(variable = var, Departement = Departement)
        Correlogramme(var, Departement)
        print("##b) Test ADF")
    print("===================================")
    print("#2) Modelisation ")
    conclusion, code = modele(Departement)
    return departement, nom_dep, donne_pres, qualite, (statuts["OBS_VALUE_CORR"], ordres_I["OBS_VALUE_CORR"]),(statuts["TM"], ordres_I["TM"]), (statuts["NBJTX30"], ordres_I["NBJTX30"]), (statuts["NBJNEIG"], ordres_I["NBJNEIG"]), conclusion, code


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


#########################################################################################################
# PREDICTION
#########################################################################################################
def Prediction(dep, horizon=15):
    """
    Prévision du flux touristique avec XGBoost
    """
    df, _ = Donne(dep)
    clear_output(wait=True)
    # 1) Mise en fréquence et interpolation
    df = Imputation(df)
    df = df.interpolate(method="time")

    # 2) Création des retards (lags)
    max_lag = 12
    for lag in range(1, max_lag + 1):
        df[f"y_lag_{lag}"] = df["OBS_VALUE_CORR"].shift(lag)

    df = df.dropna()

    # 3) Séparation X / y
    X = df.drop(columns="OBS_VALUE_CORR")
    y = df["OBS_VALUE_CORR"]

    # 4) Validation temporelle
    tscv = TimeSeriesSplit(n_splits=5)

    # 5) Modèle XGBoost
    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42
    )

    # 6) Grille d’hyperparamètres
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.9],
        "colsample_bytree": [0.7, 0.9]
    }

    grid = GridSearchCV(
        model,
        param_grid,
        cv=tscv,
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )

    grid.fit(X, y)
    best_model = grid.best_estimator_

    # ÉVALUATION HORS-ÉCHANTILLON
    
    h_eval = 12  # 12 derniers mois pour l'évaluation
    
    X_train = X.iloc[:-h_eval]
    y_train = y.iloc[:-h_eval]
    
    X_test = X.iloc[-h_eval:]
    y_test = y.iloc[-h_eval:]
    
    best_model.fit(X_train, y_train)
    
    y_pred_test = best_model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae = mean_absolute_error(y_test, y_pred_test)
    mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
    r2 = r2_score(y_test, y_pred_test)
    
    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE (%)": mape,
        "R2": r2
    }
    
    print("=== Évaluation hors-échantillon ===")
    for k, v in metrics.items():
        print(f"{k} : {v:.4f}")


    # 7) Prévision récursive
    future = []
    last_row = X.iloc[-1].copy()

    for _ in range(horizon):
        y_pred = best_model.predict(last_row.values.reshape(1, -1))[0]
        future.append(y_pred)

        # Mise à jour des lags
        for i in range(max_lag, 1, -1):
            last_row[f"y_lag_{i}"] = last_row[f"y_lag_{i-1}"]
        last_row["y_lag_1"] = y_pred

    # 8) Index futur
    future_index = pd.date_range(
        start=df.index[-1] + pd.offsets.MonthBegin(),
        periods=horizon,
        freq="MS"
    )

    future_series = pd.Series(future, index=future_index)

    # 9) Graphique
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["OBS_VALUE_CORR"], label="Observé", color="black")
    plt.plot(future_series.index, future_series, 
             label="Prévision", color="red", linestyle="--")
    plt.axvline(df.index[-1], color="gray", linestyle=":")
    plt.legend()
    plt.title("Prévision du flux touristique")
    plt.show()

    return metrics



#########################################################################################################
# IMPUTATION DES VALEURS MANQUANTES
#########################################################################################################

def Imputation(df, seasonal_period=12, max_order=2):
    """
    Impute les valeurs manquantes après asfreq("MS")
    à l'aide d'un modèle SARIMA pour chaque série.

    PARAMETRES
    ----------
    df : DataFrame
        Index DatetimeIndex
        Colonnes : flux_touris, TM, NBTX30, NBNEIG

    seasonal_period : int
        Période saisonnière (12 pour mensuel)

    max_order : int
        Ordre maximal p,q,P,Q testé

    RETURNS
    -------
    df_imputed : DataFrame
        DataFrame mensuelle complète sans NaN
    """

    # Mise à fréquence mensuelle
    df = df.sort_index()
    df_ms = df.asfreq("MS")

    df_imputed = df_ms.copy()

    for col in df_ms.columns:

        serie = df_ms[col]

        # Si aucune donnée manquante, on passe
        if serie.isna().sum() == 0:
            continue

        y_obs = serie.dropna()

        best_aic = np.inf
        best_model = None

        # Recherche du meilleur SARIMA
        AIC = []
        BIC = []
        p_q =[]
        for p in range(max_order + 1):
            for d in [0, 1]:
                for q in range(max_order + 1):
                    for P in range(max_order + 1):
                        for D in [0, 1]:
                            for Q in range(max_order + 1):
                                try:
                                    model = SARIMAX(
                                        y_obs,
                                        order=(p, d, q),
                                        seasonal_order=(P, D, Q, seasonal_period),
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
                                        # Estimation
                                    resultats = model.fit(cov_type="HC1")
                        
                                    #Validation des residus
                                    lb = acorr_ljungbox(resultats.resid,lags=[12],return_df=True)
                                    if lb.lb_pvalue.iloc[0] > 0.05:
                                        AIC.append(resultats.aic)
                                        BIC.append(resultats.bic)
                                        p_q.append((p,d,q,P,D,Q))

                                except:
                                    continue
        #  selection du meilleur modèle pour la prediction des valeurs manquantes                            
        if not AIC:
            p,q = 0, 0
            best_model = SARIMAX(
                y_obs,
                order=(p, d, q),
                seasonal_order=(P, D, Q, seasonal_period),
                enforce_stationarity=False,
                enforce_invertibility=False)
            # Estimation
            resultats = model.fit(cov_type="HC1")
            
        else:
            (p,d,q,P,D,Q) = p_q[AIC.index(min(AIC))]
            best_model = SARIMAX(
                y_obs,
                order=(p, d, q),
                seasonal_order=(P, D, Q, seasonal_period),
                enforce_stationarity=False,
                enforce_invertibility=False)
            # Estimation
            resultats = model.fit(cov_type="HC1")

        # Prédiction sur toute la période
        pred = best_model.get_prediction(
            start=df_ms.index[0],
            end=df_ms.index[-1]
        ).predicted_mean

        #Imputation uniquement des NaN
        df_imputed[col] = serie.combine_first(pred)

    return df_imputed









    
                                    