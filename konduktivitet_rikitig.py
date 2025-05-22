import pandas as pd
import numpy as np

df_kond = pd.read_csv("konduktivitet.csv", delimiter=";")

df_kond_1 = pd.read_csv("Lärjeholm konduktivitet 2001-2016.csv", delimiter=";")
df_kond_2 = pd.read_csv("Lärjeholm konduktivitet 2017-2018.csv", delimiter=";")


konduktivitet = pd.concat([df_kond_1, df_kond_2, df_kond], ignore_index=True)





konduktivitet["Konduktivitet [mS/m]"] = konduktivitet["Konduktivitet"].combine_first(
    konduktivitet["Konduktivitet 25°C"])

konduktivitet.drop(konduktivitet.columns[1:6], axis=1, inplace=True)
konduktivitet.iloc[:, 1:] = konduktivitet.iloc[:,
                                                1:].replace({',': '.'}, regex=True)

konduktivitet["Konduktivitet [mS/m]"] = pd.to_numeric(konduktivitet["Konduktivitet [mS/m]"], errors="coerce")

konduktivitet.rename(columns={"Provdatum": "Datum"}, inplace=True)

konduktivitet = konduktivitet.dropna()


konduktivitet.to_csv("konduktivitet_riktig.csv", index=False)
# DROP NAN typ juni 2002 
# Lägg ihop med hela dataframen