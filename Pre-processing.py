from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.preprocessing import PowerTransformer

import numpy as np




MF_raw = pd.read_csv("MF_raw.csv")
SMHI = pd.read_csv("SMHI_MASTER_DF.csv")
Stängningar = pd.read_csv("Stängningar.csv")
konduktivitet = pd.read_csv("konduktivitet_riktig.csv", delimiter=",")
vänern = pd.read_csv("havsnivå_medel.csv")
stängningar_månad = pd.read_csv("Månadsstängningar.csv")
PCA = pd.read_csv("PCA.csv")



stängningar_månad.rename(columns={"Stängda_dagar": "Stängda dagar", "Stängd_tid": "Stängd tid [h]"}, inplace=True)


MF_raw['Datum'] = pd.to_datetime(MF_raw['Datum'])
SMHI['Datum'] = pd.to_datetime(SMHI['Datum'])
Stängningar['Datum'] = pd.to_datetime(Stängningar['Datum'])
konduktivitet['Datum'] = pd.to_datetime(konduktivitet['Datum'])
vänern['Datum'] = pd.to_datetime(vänern['Datum'])
stängningar_månad['Datum'] = pd.to_datetime(stängningar_månad['Datum'])
PCA['Datum'] = pd.to_datetime(PCA['Datum'])





konduktivitet = konduktivitet.groupby("Datum", as_index=False).max()  #  .first() funkar också, borde bli samma resultat



NOT_Master_DataFrame = pd.merge(MF_raw.copy(), Stängningar, on="Datum", how="outer")

NOOT_Master_DataFrame = pd.merge(NOT_Master_DataFrame, vänern, on="Datum", how="outer")


PRE_Master_DataFrame = pd.merge(NOOT_Master_DataFrame, konduktivitet, on="Datum", how="outer")

PREE_Master_DataFrame = pd.merge(PRE_Master_DataFrame, stängningar_månad, on="Datum", how="outer")

PREEE_Master_DataFrame = pd.merge(PREE_Master_DataFrame, PCA, on="Datum", how="outer")




numeric_cols = MF_raw.select_dtypes(include=[np.number]).columns


MF_raw.dropna(subset=MF_raw.columns.difference(
    ["Datum", "Nederbördsmängd [mm]"]), how="all", inplace=True)


df_bakt = MF_raw[["Datum", "Escherichia coli MPN [ant/100ml]",
                  "Koliforma bakterier MPN [ant/100ml]", "Nederbördsmängd [mm]"]]
df_bakt = df_bakt.dropna()

df_rest = MF_raw.drop(["Escherichia coli MPN [ant/100ml]",
                      "Koliforma bakterier MPN [ant/100ml]", "Nederbördsmängd [mm]"], axis=1)

df_rest_drop = df_rest.dropna()  # No imputation


# With interpolation, dropping all rows with only nan
df_rest = df_rest.dropna(
    subset=df_rest.columns.difference(["Datum"]), how="all")


df_rest.set_index('Datum', inplace=True)

df_rest.interpolate(method='time', inplace=True)

df_rest.reset_index(inplace=True)

# Dropping the remaining rows with some nan, 50 rows where dropped 3,5 % of total data because of missing data in 2/3 of parameters
df_rest.dropna(inplace=True)


Correlation_DataFrame = pd.merge(df_rest, df_bakt, on="Datum", how="outer")
Correlation_DataFrame = Correlation_DataFrame.sort_values(
    by='Datum').reset_index(drop=True)

scaler = MinMaxScaler()
Correlation_DataFrame[numeric_cols] = scaler.fit_transform(
    Correlation_DataFrame[numeric_cols])

Master_DataFrame = pd.merge(PREEE_Master_DataFrame, SMHI, on="Datum", how="outer")

Master_DataFrame.drop(["Nederboerdsmaengd"], axis=1, inplace=True)

numeric_cols_2 = Master_DataFrame.select_dtypes(include=[np.number]).columns

pt = PowerTransformer(method='yeo-johnson', standardize=False)  # standardize=True gives mean=0, std=1

df_power = pd.DataFrame(pt.fit_transform(Master_DataFrame[numeric_cols_2]), columns=numeric_cols_2)

Master_STD = pd.DataFrame(scaler.fit_transform(df_power), columns=numeric_cols_2)

Master_STD = pd.concat([Master_DataFrame["Datum"], Master_STD], axis=1)


Master_DataFrame.rename(columns={"Lufttemperatur": "Lufttemperatur [°C]", "Tid (timmar)": "Tid [h]", "Vindhastighet": "Vindhastighet [m/s]"}, inplace=True)

# With all original values
Master_DataFrame.to_csv("Master_DataFrame.csv", index=False)

Correlation_DataFrame.to_csv(
    "Correlation_DataFrame.csv", index=False)  # With interpolation

Master_STD.to_csv("Master_STD.csv", index=False)


