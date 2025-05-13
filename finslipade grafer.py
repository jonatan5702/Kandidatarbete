import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re

# Läs in data
SMHI_lufttemperatur_df = pd.read_csv("SMHI_lufttemperatur.csv", sep=";")
SMHI_nederbord_df = pd.read_csv("SMHI_nederbord.csv", sep=";")
SMHI_solinstralning_df = pd.read_csv("SMHI_solinstralning.csv", sep=";")

SMHI_solinstralning_df = SMHI_solinstralning_df.groupby("Datum").agg({
    "Global irradians [W/m2]": "mean",
    "Solskenstid [s]": "sum"
}).reset_index()
SMHI_solinstralning_df["Solskenstid [min]"] = SMHI_solinstralning_df["Solskenstid [s]"] / 60
SMHI_solinstralning_df.drop(columns=["Solskenstid [s]"], inplace=True)



SMHI_vindhast_och_riktning_df = pd.read_csv("SMHI_vindhast_och_riktning.csv", sep=";")

def cirkulärt_medelvärde(vindriktningar):
    radianer = np.radians(vindriktningar)
    x = np.cos(radianer).mean()
    y = np.sin(radianer).mean()
    medelvinkel = np.degrees(np.arctan2(y, x))
    return medelvinkel % 360

SMHI_vindhast_och_riktning_df = SMHI_vindhast_och_riktning_df.groupby("Datum").agg({
    "Vindriktning [°]": lambda x: cirkulärt_medelvärde(x),
    "Vindhastighet [m/s]": "max"
}).reset_index()

säsong_färger = {
    'Vinter': 'blue',
    'Vår': 'green',
    'Sommar': 'mediumvioletred',
    'Höst': 'darkorange'
}

dfs = [SMHI_lufttemperatur_df, SMHI_nederbord_df, SMHI_solinstralning_df, SMHI_vindhast_och_riktning_df]
for df in dfs:
    df["Datum"] = pd.to_datetime(df["Datum"])

    for col in df.columns:
        if col == "Datum":
            continue

        df = df.dropna(subset=[col, "Datum"])
        data = df[col]
        plt.figure(figsize=(13, 6))

        x = (df["Datum"] - df["Datum"].min()).dt.days

        # Glidande medelvärde (justera fönsterstorlek vid behov)
        if col == "Nederbördsmängd [mm]":
            window_size = 60
        else:
            window_size = 28
        
        rolling_mean = data.rolling(window=window_size, center=True).mean()

        plt.grid(True, which='both', axis='both', linestyle='-', linewidth=1, zorder=0)
        plt.grid(True, which='minor', axis='both', linestyle='--', linewidth=0.4, zorder=0)

        # Rita den glidande medelvärdeslinjen
        plt.plot(df["Datum"], rolling_mean, label=f"{window_size}-dagars glidande medel", color="steelblue", linewidth=1, zorder=5)

        # Övergripande regressionslinje
        a, b = np.polyfit(x, data, 1)
        plt.plot(df["Datum"], a*x + b, color="red", label="Samlad trend", linewidth=1, zorder=10)

        # Trendlinjer per säsong
        säsonger = ['Vinter', 'Vår', 'Sommar', 'Höst']
        for säsong_namn in säsonger:
            season_data = df[df["Datum"].dt.month.isin({
                'Vinter': [12, 1, 2],
                'Vår': [3, 4, 5],
                'Sommar': [6, 7, 8],
                'Höst': [9, 10, 11]
            }[säsong_namn])]

            if len(season_data) > 1:
                season_x = (season_data["Datum"] - season_data["Datum"].min()).dt.days
                season_y = season_data[col]
                season_a, season_b = np.polyfit(season_x, season_y, 1)
                plt.plot(season_data["Datum"], season_a*season_x + season_b,
                         label=f"{säsong_namn} trend",
                         color=säsong_färger[säsong_namn],
                         linewidth=1,
                         linestyle='--',
                         zorder=10)


        plt.xlabel("")
        plt.ylabel(col)
        plt.title("")
        plt.xticks(rotation=45)

        år_ticks = pd.date_range(df["Datum"].min(), df["Datum"].max(), freq='YS')  # Början av varje år
        plt.xticks(år_ticks, labels=[str(d.year) for d in år_ticks])


        #plt.minorticks_on()
        plt.legend()

        col = re.sub(r"/", "-", col)
        filename = f"{col}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')

        plt.show()
