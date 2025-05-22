import pandas as pd

df_unsorted = pd.read_csv("Lärjeholm_online.csv", delimiter=";")
"""
Onlinedata som innehåller olikheter och standardenheter och värden. 
Följade rader gör en df med de uppmätta värden och enhet och en med standardvärden och enhet

"""

df_presorted = df_unsorted.drop(df_unsorted.columns[[0, 1, 7, 8]], axis = 1)

df_standard = df_unsorted.drop(df_unsorted.columns[[0, 1, 4, 5, 6]], axis = 1)


"""
Denna funktion räknar antalet olika parametrar eftersom de inte är sorterade efter datum. 
Alltså förekommer flera datum flera gånger och alla parametrar flera gånger.
Vi vill ha en df där datumen förkommer en gång och parametrarna är kolumnnamn.
parameters längre ner ger en lista med alla olika parametrar
"""
def counter(df, column):

    param_counts = df[column].value_counts()
                
    param_lst = param_counts[param_counts > 1].index.tolist()
    
    
    return param_lst


parameters = counter(df_presorted, "Parameter")


"""
wrong_units undersöker om enheterna stämmer med standardenheterna. I de flesta fall gör dem det.
När det gäller alkalinitet var det bara att ändra eftersom ration var 1:1
E coli behöll vi CFU trots att standardenheten var MPN eftersom CFU är ett uppmätningsmetod medan MPN är statistiskt
"""

def wrong_units(df, df_standard):
    # Finding unit which do not match
    
    for param in parameters:
        df_specific = df[df["Parameter"] == param]
        df_standard_specific = df_standard[df_standard["Parameter"] == param]
    
        
        if not df_specific["Enhet"].equals(df_standard_specific["Standardenhet"]):
            print(f"Wrong unit for parameter: {param}")
        else:
            pass

        
# wrong_units(df_presorted, df_standard)
"""
Följande kod sorterar enheterna.
Först byter vi enheten på alkalinitet eftersom vi vet att den har olika med ration är 1:1
sedan för vi om alla värden till float genom att byta "," med "." 
Därefter konverterar vi alla mikrogram till nanogram och uppdaterar värdet genom att multiplicera med 1000
wrong_units används för att se om vi har lyckats. Ser bra ut.
"""


# Sortinng the units
df_sorted = df_presorted.copy()

df_sorted.loc[df_sorted["Enhet"] == "mmol HCO3-/l", "Enhet"] = "mekv/l"



df_sorted["Varde"] = df_sorted["Varde"].replace({',': '.'}, regex=True).astype(float)




mask = (df_sorted["Enhet"] == "µg/l") & (df_sorted["Parameter"] == "Hg")

df_sorted.loc[mask, "Varde"] = df_sorted.loc[mask, "Varde"] * 1000 
df_sorted.loc[mask, "Enhet"] = "ng/l" 

# wrong_units(df_sorted, df_standard)


test = df_sorted[df_sorted["Parameter"] == "Hg"]


"""
Något konstigt med Hg eftersom det är ett hopp mellan 2004 och 2024
Efter detta hopp är det endast mindre 10 ng/l.
2004 var däremot mätvärdet lägre och kanske mer exakt.
Hg bör vi alltså hålla ett öga på i slutskedet
"""

df_sorted["Parameter"] = df_sorted["Parameter"] + " [" + df_sorted["Enhet"] + "]"

df_sorted = df_sorted.drop("Enhet", axis = 1)


df_sorted["Varde"] = df_sorted["Olikhet"].fillna("") + df_sorted["Varde"].astype(str)

df_sorted = df_sorted.drop("Olikhet", axis = 1)


df_sorted["Datum"] = pd.to_datetime(df_sorted["Datum"], errors="coerce").dt.date


df = df_sorted.pivot_table(index="Datum", columns="Parameter", values="Varde", aggfunc="first").reset_index()

check = counter(df_sorted, "Parameter")

df_check = df_sorted[df_sorted["Parameter"] == check[7]]



df_online = df

df_online.iloc[:, 1:] = df_online.iloc[:, 1:].replace({',': '.'}, regex=True)




for col in df_online.columns:
    df_online[col] = df_online[col].astype(str)
       
    condition_less = df_online[col].str.contains('<', na=False)
    df_online.loc[condition_less, col] = df_online.loc[condition_less, col].str.replace('<', '').astype(float) / 2
       
    condition_greater = df_online[col].str.contains('>', na=False)
    df_online.loc[condition_greater, col] = df_online.loc[condition_greater, col].str.replace('>', '').astype(float)
   

df_online = df_online.apply(lambda col: pd.to_numeric(col, errors='coerce') if col.name != "Datum" else col)




df_online.to_csv("Lärjeholm_sorterad_online.csv", index=False)




#%% Check all units
import pandas as pd
import numpy as np

df_unsorted_2001_2016 = pd.read_csv("Lärjeholm 2001-2016.csv", delimiter=";")
df_unsorted_2017_2024 = pd.read_csv("Lärjeholm 2017-2024.csv", delimiter=";")


df_1_omgång_2 = pd.read_csv("Lärjeholm 2001-2016 (omgång 2).csv", delimiter=";")
df_2_omgång_2 = pd.read_csv("Lärjeholm 2017-2024 (omgång 2).csv", delimiter=";")

df_unsorted_2001_2016 = df_unsorted_2001_2016.drop(["Kisel", "Kisel_enhet"], axis=1)
df_2_omgång_2 = df_2_omgång_2.drop(["Unnamed: 1"], axis=1)


df_unsorted_1 = pd.merge(df_unsorted_2001_2016, df_1_omgång_2, on="Provdatum", how="outer")
df_unsorted_2 = pd.merge(df_unsorted_2017_2024, df_2_omgång_2, on="Provdatum", how="outer")





def counts(df, parameter):
    
    param_counts = df[parameter].value_counts()
                
    param_lst = param_counts[param_counts >= 1].index.tolist()
    
    return param_lst

def unit_test(df, starting, step, nr):
    unit = df.iloc[:, starting::step]
    unit_lst = unit.columns.tolist()

    all_units = []
    
    for units in unit_lst:
        test = counts(df, units)
        all_units.extend(test) 
        if len(test) != 1: 
            print(f" '{units}': {test}", nr)

    return all_units




df_sorted_2001_2016 = df_unsorted_1.iloc[:, [0, 1] + list(range(3, df_unsorted_1.shape[1], 2))]

df_sorted_2017_2024 = df_unsorted_2.iloc[:, [0, 1] + list(range(3, df_unsorted_2.shape[1], 2))]


df_sorted_2001_2016.loc[:, "Provdatum"] = pd.to_datetime(df_sorted_2001_2016["Provdatum"], errors="coerce").dt.date
df_sorted_2017_2024.loc[:, "Provdatum"] = pd.to_datetime(df_sorted_2017_2024["Provdatum"], errors="coerce").dt.date






def new_columns(df_sorted, df_unsorted, starting, step, nr):
    all_units = unit_test(df_unsorted, starting, step, nr)
    
    new_columns = [df_sorted.columns[0]] + [str(col) + ' [' + str(unit) + ']' for col, unit in zip(df_sorted.columns[1:], all_units)]

    
    return new_columns


df_sorted_2001_2016.columns = new_columns(df_sorted_2001_2016, df_unsorted_1, 2, 2, 1)

df_sorted_2017_2024.columns = new_columns(df_sorted_2017_2024, df_unsorted_2, 2, 2, 2)



def sorting(df):
    date_column = df.iloc[:, 0]
    
    sorted_columns = df.iloc[:, 1:].columns.sort_values()
    
    sorted_df = pd.concat([date_column, df[sorted_columns]], axis=1)
    return sorted_df

df_1 = sorting(df_sorted_2001_2016)
df_2 = sorting(df_sorted_2017_2024)




all_columns = list(df_1.columns.union(df_2.columns))

same_columns = list(df_1.columns.intersection(df_2.columns))

df_merged = pd.merge(df_1, df_2, on='Provdatum', how='outer')

for col in df_1.columns.intersection(df_2.columns):  
    if col != 'Provdatum':  
        df_merged[col] = df_merged.pop(col + '_x').fillna(df_merged.pop(col + '_y'))

sorted_columns = ['Provdatum'] + sorted(df_merged.columns.difference(['Provdatum']))

df_sorted = df_merged[sorted_columns]



for col in df_sorted.columns:
    if '.1' in col:
        base_col = col.replace('.1', '')
        
        if base_col in df_sorted.columns:
            df_sorted[base_col] = df_sorted[base_col].fillna(df_sorted[col])
            
            df_sorted.drop(columns=[col], inplace=True)



# for param in df_sorted.columns:
#     test_1 = df_sorted[[param]].dropna()
#     if len(test_1) < 50:
#         print(param, " have ", len(test_1), " data points")






        
df_sorted = df_sorted.rename(columns={"Provdatum": "Datum"})
    

df_kov = df_sorted

df_kov.iloc[:, 1:] = df_kov.iloc[:, 1:].replace('------', np.nan)


df_kov.iloc[:, 1:] = df_kov.iloc[:, 1:].replace({',': '.'}, regex=True)

"""
Replacing it with half the number if < and the number if >
"""

df_raw = df_kov.copy()
for col in df_raw.columns:
    df_raw[col] = df_raw[col].astype(str)
       
    condition_less = df_raw[col].str.contains('<', na=False)
    df_raw.loc[condition_less, col] = df_raw.loc[condition_less, col].str.replace('<', '').astype(float) / 2
       
    condition_greater = df_raw[col].str.contains('>', na=False)
    df_raw.loc[condition_greater, col] = df_raw.loc[condition_greater, col].str.replace('>', '').astype(float)
    
"""
Removing > < althoghether
"""
for col in df_kov.columns:
    df_kov[col] = df_kov[col].astype(str)

    # Replace values containing '<' with NaN
    condition_less = df_kov[col].str.contains('<', na=False)
    df_kov.loc[condition_less, col] = np.nan

    # Replace values containing '>' with NaN
    condition_greater = df_kov[col].str.contains('>', na=False)
    df_kov.loc[condition_greater, col] = np.nan
    


df_kov = df_kov.apply(lambda col: pd.to_numeric(col, errors='coerce') if col.name != "Datum" else col)
df_raw = df_raw.apply(lambda col: pd.to_numeric(col, errors='coerce') if col.name != "Datum" else col)


numeric_cols = df_kov.select_dtypes(include=[np.number]).columns

for param in numeric_cols:

    test = df_kov[[param]].dropna()
    
    if len(test) < 50:
        print(param, " have ", len(test), " data points")



df_kov.to_csv("Lärjeholm_sorterad.csv", index=False)
df_raw.to_csv("Lärjeholm_sorterad_rå.csv", index=False)




#%% Plotta för att se vilka vi kan ta bort 

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

df_kov = pd.read_csv("Lärjeholm_sorterad.csv", delimiter=",")
df_kov['Datum'] = pd.to_datetime(df_kov['Datum'], errors='coerce')

parameter_groups = {
    "Antimon": [
        "Antimon ICP-MS [µg/l]",
        "Antimon [µg/l]"
    ],
    "Bly": [
        "Bly ICP-MS [µg/l]",
        "Bly ICP-MS filtr [µg/l]",
        "Bly ICP-MS uppsl [µg/l]"
    ],
    "Zink": [
        "Zink ICP-MS [µg/l]",
        "Zink ICP-MS filtr [µg/l]",
        "Zink ICP-MS uppsl [µg/l]"
    ],
    "Koppar": [
        "Koppar ICP-MS [µg/l]",
        "Koppar ICP-MS filtr [µg/l]",
        "Koppar ICP-MS uppsl [µg/l]"
    ],
    "Krom": [
        "Krom ICP-MS [µg/l]",
        "Krom ICP-MS filtr [µg/l]",
        "Krom ICP-MS uppsl [µg/l]"
    ],
    "Arsenik": [
        "Arsenik ICP-MS [µg/l]",
        "Arsenik ICP-MS filtr [µg/l]"
    ],
    "Barium": [
        "Barium ICP-MS [µg/l]"
    ],
    "Bor": [
        "Bor ICP-MS [µg/l]",
        "Bor [µg/l]"
    ],
    "Kadmium": [
        "Kadmium ICP-MS [µg/l]",
        "Kadmium ICP-MS filtr [µg/l]",
        "Kadmium ICP-MS uppsl [µg/l]"
    ],
    "Järn": [
        "Järn ICP-MS [mg/l]",
        "Järn ICP-MS uppsl [mg/l]",
        "Järn flamma [mg/l]"
    ],
    "Magnesium": [
        "Magnesium ICP-MS [mg/l]"
    ],
    "Mangan": [
        "Mangan ICP-MS [mg/l]",
        "Mangan ICP-MS uppsl [mg/l]"
    ],
    "Nickel": [
        "Nickel ICP-MS [µg/l]",
        "Nickel ICP-MS filtr [µg/l]",
        "Nickel ICP-MS uppsl [µg/l]"
    ],
    "Silver": [
        "Silver ICP-MS [µg/l]"
    ],
    "Selen": [
        "Selen ICP-MS [µg/l]"
    ],
    "Uran": [
        "Uran ICP-MS [µg/l]"
    ],
    "Vanadin": [
        "Vanadin ICP-MS [µg/l]"
    ],
    "Vismut": [
        "Vismut ICP-MS [µg/l]"
    ],
    "Kalcium": [
        "Kalcium ICP-MS [mg/l]"
    ],
    "Kalium": [
        "Kalium ICP-MS [mg/l]"
    ],
    "Kobolt": [
        "Kobolt ICP-MS [µg/l]"
    ],
    "Kisel": [
        "Kisel [mg/l]"
    ],
    "Koliforma bakterier": [
        "Koliforma bakterier MPN [ant/100ml]",
        "Koliforma bakterier [CFU/100ml]"
    ],
    "Escherichia coli": [
        "Escherichia coli MPN [ant/100ml]",
        "Escherichia coli [CFU/100ml]"
    ],
    "Kvicksilver": [
        "Kvicksilver AAS [µg/l]",
        "Kvicksilver AAS totalt [µg/l]",
        "Kvicksilver FLUO [ng/l]",
        "Kvicksilver FLUO filtr [ng/l]",
        "Kvicksilver ICP-MS [µg/l]",
        "Kvicksilver ICP-MS uppsl [µg/l]"
    ],
    "TOC": [
        "TOC [mg/l]"
    ],
    "DOC": [
        "DOC [mg/l]"
    ],
    "Extinktion 254nm": [
        "Extinktion 254nm [ae/cm]",
        "Extinktion 254nm ofiltr [ae/cm]"
    ],
    "Färgtal 410 nm": [
        "Färgtal 410 nm [mg/l Pt]",
        "Färgtal [mg/l Pt]"
    ],
    "Transmission 254 nm": [
        "Transmission 254 nm ofiltr [%]"
    ]
}

for group, parameters in parameter_groups.items():
    plt.figure(figsize=(12, 6))
    
    for param in parameters:
        if param in df_kov.columns:
            plt.scatter(df_kov["Datum"], df_kov[param], label=param, marker='o')
    
    plt.title(f"{group} över Tid")
    plt.xlabel("Månad")
    plt.ylabel("Värde")
    plt.legend(loc='upper left')

    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    plt.gca().set_xlim([df_kov['Datum'].min(), df_kov['Datum'].max()])

    plt.gca().xaxis.set_major_locator(mdates.YearLocator(1)) 
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())

    plt.xticks(rotation=45)

    plt.grid(True)
    plt.tight_layout()
    plt.show()


#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime



df_kov = pd.read_csv("Lärjeholm_sorterad.csv", delimiter=",")
df_raw = pd.read_csv("Lärjeholm_sorterad_rå.csv", delimiter=",")



df_online = pd.read_csv("Lärjeholm_sorterad_online.csv", delimiter=",")

def sorting(df):
    date_column = df.iloc[:, 0]
    
    sorted_columns = df.iloc[:, 1:].columns.sort_values()
    
    sorted_df = pd.concat([date_column, df[sorted_columns]], axis=1)
    return sorted_df




dropped_param_kov = ["Antimon ICP-MS [µg/l]", "Antimon [µg/l]", "Bly ICP-MS filtr [µg/l]",
"Bly ICP-MS uppsl [µg/l]",
"Zink ICP-MS filtr [µg/l]",
"Zink ICP-MS uppsl [µg/l]",
"Koppar ICP-MS filtr [µg/l]",
"Koppar ICP-MS uppsl [µg/l]",
"Krom ICP-MS filtr [µg/l]",
"Krom ICP-MS uppsl [µg/l]",
"Arsenik ICP-MS filtr [µg/l]",
"Bor [µg/l]",
"Kadmium ICP-MS filtr [µg/l]",
"Kadmium ICP-MS uppsl [µg/l]",
"Järn ICP-MS uppsl [mg/l]",
"Järn flamma [mg/l]",
"Mangan ICP-MS uppsl [mg/l]",
"Nickel ICP-MS filtr [µg/l]",
"Nickel ICP-MS uppsl [µg/l]",
"Selen ICP-MS [µg/l]",
"Silver ICP-MS [µg/l]",
"Strontium ICP-MS [µg/l]",
"Vismut ICP-MS [µg/l]",
"Kvicksilver AAS [µg/l]",
"Kvicksilver AAS totalt [µg/l]",
"Kvicksilver FLUO [ng/l]",
"Kvicksilver FLUO filtr [ng/l]",
"Kvicksilver ICP-MS [µg/l]", 
"Kvicksilver ICP-MS uppsl [µg/l]",
"DOC [mg/l]", 
"Transmission 254 nm ofiltr [%]", 
"Extinktion 254nm ofiltr [ae/cm]" ]

dropped_param_online = ["Ca [mg/l]", "Cr [µg/l]", "Cu [µg/l]", "E Coli/LTLSB [(35°C,MPN) / 100ml]", 
                        "Fe [mg/l]", "As [µg/l]", "Cd [µg/l]", "Co [µg/l]", "Hg [ng/l]", "K [mg/l]",
                        "Mg [mg/l]", "Mn [µg/l]", "NO2+NO3-N [µg/l N]", "NO2-N [µg/l N]", "Ni [µg/l]",
                        "Na [mg/l]",
                        "Org-N [µg/l N]", "PO4-P [µg/l P]", "Pb [µg/l]", "SO4 [mg/l S]", "TOC [mg/l C]",
                        "Tot Ant Kol Bakt [ant/100ml]", "V [µg/l]", "Zn [µg/l]", "Övrig fosfor [µg/l]"]

df_kov = df_kov.drop(dropped_param_kov, axis=1)

df_raw = df_raw.drop(dropped_param_kov, axis=1)



df_online = df_online.drop(dropped_param_online, axis = 1)



def merging(df_1, df_2):

    mf = pd.merge(df_1, df_2, on='Datum', how='outer')
    mf['Datum'] = pd.to_datetime(mf['Datum'], errors='coerce').dt.date
    
    mf = mf.sort_values(by='Datum').reset_index(drop=True)
    
    
    mf = sorting(mf)
    
    """
    Här har jag kombinerat dessa parametrar för att de är samma sak men olika mätmetoder
    Det är värt att från om vi ska prioritera MPN över CFU eftersom det är MPN vi kommer kalla kolumnen för 
    Eller om vi behåller så många CFU för det är riktiga mätvärden
    """
    
    
    mf["Kisel [mg/l]"] = mf["Kisel [mg/l]"].combine_first(mf["Si [mg/l]"])
    mf.drop(columns=["Si [mg/l]"], inplace=True)
    
    
    
    mf["Escherichia coli MPN [ant/100ml]"] = mf["Escherichia coli [CFU/100ml]"].combine_first(mf["Escherichia coli MPN [ant/100ml]"])
    mf.drop(columns=["Escherichia coli [CFU/100ml]"], inplace=True)
    
    
    
    mf["Koliforma bakterier MPN [ant/100ml]"] = mf["Koliforma bakterier [CFU/100ml]"].combine_first(mf["Koliforma bakterier MPN [ant/100ml]"])
    mf.drop(columns=["Koliforma bakterier [CFU/100ml]"], inplace=True)
    
    
    mf["Färgtal [mg/l Pt]"] = mf["Färgtal [mg/l Pt]"].combine_first(mf["Färgtal 410 nm [mg/l Pt]"])
    mf.drop(columns=["Färgtal 410 nm [mg/l Pt]"], inplace=True)
    
    
    mf["Färgtal [mg/l Pt]"] = mf["Färgtal [mg/l Pt]"].combine_first(mf["Färgtal [mg Pt/l]"])
    mf.drop(columns=["Färgtal [mg Pt/l]"], inplace=True)
    
    
    mf["Extinktion 254nm [ae/cm]"] = mf["Extinktion 254nm [ae/cm]"].combine_first(mf["Extinktion 254 nm [ae/cm]"])
    mf.drop(columns=["Extinktion 254 nm [ae/cm]"], inplace=True)
    
    
    mf = mf.groupby("Datum", as_index=False).max()  #  .first() funkar också, borde bli samma resultat
    
    return mf

mf = merging(df_kov, df_online)
mf_raw = merging(df_raw, df_online)






df_SMHI = pd.read_csv("Regndata_SMHI_Nordstan.csv", delimiter=";", skiprows=13, usecols=[0, 1, 2, 3, 4])


df_SMHI['Representativt dygn'] = pd.to_datetime(df_SMHI['Representativt dygn'], errors='coerce').dt.date
mf['Datum'] = pd.to_datetime(mf['Datum'], errors='coerce').dt.date



df_SMHI = df_SMHI[df_SMHI["Representativt dygn"] >= datetime(2001, 1, 1).date()].reset_index(drop=True)
df_SMHI = df_SMHI.drop(df_SMHI.columns[[0, 1]], axis = 1)
df_SMHI = df_SMHI.rename(columns={"Representativt dygn": "Datum"})
df_SMHI = df_SMHI.rename(columns={"Nederbördsmängd": "Nederbördsmängd [mm]"})


# for idx, kval in df["Kvalitet"].items():
#     if kval != "G":
#         print(idx, "Test") 

df_SMHI = df_SMHI.drop(df_SMHI.columns[[2]], axis = 1)

"""
Endast ett fåtal misstänkta värden, det antas att dessa inte komma ha en påverkan på den totala trenden. 
Oavsett kommer isolation forest förhoppningsvis ta hand om det.
"""

MF = pd.merge(mf, df_SMHI, on="Datum", how="outer")
MF = MF.sort_values(by='Datum').reset_index(drop=True)


MF_raw = pd.merge(mf_raw, df_SMHI, on="Datum", how="outer")
MF_raw = MF_raw.sort_values(by='Datum').reset_index(drop=True)



# for col in MF_raw.columns:
#     print(col, len(MF_raw[col].dropna()), len(MF[col].dropna()))
    

    
for col in MF_raw.columns:
    if len(MF_raw[col].dropna()) < 1000:
        MF_raw.drop(col, axis=1, inplace=True)
        
for col in MF.columns:
    if len(MF[col].dropna()) < 1000:
        MF.drop(col, axis=1, inplace=True)
        

MF_raw.dropna(subset=MF_raw.columns.difference(["Datum"]), how="all", inplace=True)
MF.dropna(subset=MF.columns.difference(["Datum"]), how="all", inplace=True)

MF_raw.drop("Kadmium ICP-MS [µg/l]", axis=1, inplace=True)

for col in MF_raw.columns:
    print(col, len(MF_raw[col].dropna()) - len(MF[col].dropna()))
    
test2 = MF_raw.isnull().mean() * 100
test1 = MF.isnull().mean() * 100

"""
We have two dataframes for all which have >< (MF_raw) kept as * 1/2 or 1.
MF is where <> is removed, for every step in the future we will use MF_raw, 
since there are so few <> and we want to get as much data as possible without introducing bias 
"""


MF.to_csv("MF.csv", index=False) 
MF_raw.to_csv("MF_raw.csv", index=False)

