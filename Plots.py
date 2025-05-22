#%% Defining functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from numpy.polynomial.polynomial import polyfit
from datetime import datetime
from matplotlib.patches import Patch



Master_DataFrame = pd.read_csv("Master_DataFrame.csv") # Original data with < > replaced with factor 1/2 or 1 


redline = pd.read_excel("Intag_stangning.xlsx")



Master_DataFrame['Datum'] = pd.to_datetime(Master_DataFrame['Datum'], errors='coerce').dt.date

Master_DataFrame["E.coli / Koliforma bakterier"] = Master_DataFrame["Escherichia coli MPN [ant/100ml]"] / Master_DataFrame["Koliforma bakterier MPN [ant/100ml]"]

# Run this to define all functions the plots are below
plt.rcParams.update({
    'font.size': 12,               # base font size for text
    'axes.labelsize': 14,          # axis label font size
    'xtick.labelsize': 11,         # x-axis tick font size
    'ytick.labelsize': 11,         # y-axis tick font size
    'legend.fontsize': 12,         # legend font size
    'figure.titlesize': 16,        # figure title font size (if used)
    'axes.titlesize': 14,          # axes title font size
    # 'lines.linewidth': 2,          # default line width
    # 'lines.markersize': 6,         # default marker size
    # 'grid.linewidth': 0.5,         # thinner grid lines
    # 'axes.grid': True,             # enable grid by default
})

        
        
def histo(df, tuning):
    for param in df.select_dtypes(include=[np.number]).columns:
        
        plt.hist(df[param], bins=tuning, edgecolor='white', alpha=0.7)
        
        plt.xlabel(f"{param}")
        plt.ylabel('Frekvens')
        
        plt.show()
        

def overlapping(df, highlight):
    columns = list(enumerate(df.select_dtypes(include=[np.number]).columns))

    print("The parameters are:\n")
    for i in range(0, len(columns), 2):
        col1 = f"{columns[i][0]}: {columns[i][1]}"
        col2 = f"{columns[i+1][0]}: {columns[i+1][1]}" if i + 1 < len(columns) else ""
        print(f"{col1:<30} {col2}")

    param1 = df.select_dtypes(include=[np.number]).columns[int(input("Choose the first parameter (index): "))]
    param2 = df.select_dtypes(include=[np.number]).columns[int(input("Choose the second parameter (index): "))]

    if param1 not in df.columns or param2 not in df.columns:
        print("Error: One or both of the parameters are not in the df")
        return

    print("Chosen parameters: ", param1, param2)

    df['Datum'] = pd.to_datetime(df['Datum'])
    
    if param1 == "Escherichia coli MPN [ant/100ml]" or param2 == "Escherichia coli MPN [ant/100ml]":
        ecoli = int(input("Do you want threshold value? (1/0): "))
    else:
        ecoli = 0

    start_date = input("Choose start date (YYYY-MM-DD): ")
    end_date = input("Choose end date (YYYY-MM-DD): ")

    window_1 = int(input(f"Floating mean window for {param1} (1 is mean for one day): "))
    window_2 = int(input(f"Floating mean window for {param2} (1 is mean for one day): "))

    test = int(input("Closed time (1/0): "))

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    df = df[(df['Datum'] >= start) & (df['Datum'] <= end)]

    special_params = ["Stängd tid [h]", "Stängda dagar"]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    valid_data_1 = df[['Datum', param1]].dropna()
    valid_data_2 = df[['Datum', param2]].dropna()

    if param1 not in special_params:
        valid_data_1['Rolling_Mean'] = valid_data_1[param1].rolling(window=window_1, min_periods=1).mean()
    else:
        valid_data_1['Rolling_Mean'] = valid_data_1[param1]

    line1, = ax1.plot(valid_data_1["Datum"], valid_data_1['Rolling_Mean'], label=f"{param1}", color="darkorange")
    ax1.set_ylabel(param1)
    ax1.tick_params(axis='y')
    # ax1.set_ylim(top=199)




    ax2 = ax1.twinx()

    if param2 not in special_params:
        valid_data_2['Rolling_Mean'] = valid_data_2[param2].rolling(window=window_2, min_periods=1).mean()
    else:
        valid_data_2['Rolling_Mean'] = valid_data_2[param2]

    line2, = ax2.plot(valid_data_2["Datum"], valid_data_2["Rolling_Mean"], label=f"{param2}")
    ax2.set_ylabel(param2)
    ax2.tick_params(axis='y')

    if test == 1:
        highlight['Intag stängt'] = pd.to_datetime(highlight['Intag stängt'])
        highlight['Intaget öppet'] = pd.to_datetime(highlight['Intaget öppet'])
        for _, row in highlight.iterrows():
            closed_patch = Patch(facecolor='skyblue', edgecolor='skyblue', alpha=0.3, label='Stängt råvattenintag')

            ax1.axvspan(row['Intag stängt'], row['Intaget öppet'], color='skyblue', alpha=0.3)
            
            
    custom_red_boxes = []
    add_custom = int(input("Do you want to add custom red-highlighted periods? (1/0): "))
    
    
    if add_custom == 1:
        num_periods = int(input("How many periods do you want to add? "))
        
        for i in range(num_periods):
            red_start = input(f"Enter start date for red box #{i+1} (YYYY-MM-DD): ")
            red_end = input(f"Enter end date for red box #{i+1} (YYYY-MM-DD): ")
            
            try:
                red_start_dt = pd.to_datetime(red_start)
                red_end_dt = pd.to_datetime(red_end)
                ax1.axvspan(red_start_dt, red_end_dt, color='red', alpha=0.2)
                custom_red_boxes.append((red_start_dt, red_end_dt))
                
            except Exception as e:
                print(f"Invalid date input: {e}")

        red_patch = Patch(facecolor='red', edgecolor='red', alpha=0.2, label='Observera')

    ecoli_param_name = "Escherichia coli MPN [ant/100ml]"
    
    legend_handles = [line1, line2]

    if ecoli == 1:
        ecoli_ax = ax1 if param1 == ecoli_param_name else ax2
        ecoli_line = ecoli_ax.axhline(
            y=100,
            color='red',
            linestyle=':',
            linewidth=1.5,
            label="Riktvärde E.coli"
        )
        legend_handles.append(ecoli_line)
    
    if test == 1:
        legend_handles.append(closed_patch)
    if add_custom == 1 and custom_red_boxes:
        legend_handles.append(red_patch)
    
    ax1.legend(handles=legend_handles, loc='upper left')


    plt.xlim(valid_data_1["Datum"].min(), valid_data_1["Datum"].max())

    days_diff = abs((end - start).days)
    if days_diff < 190:
        ax1.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    elif days_diff < 365:
        ax1.xaxis.set_minor_locator(mdates.DayLocator(interval=2))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    elif days_diff < 4 * 365:
        ax1.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    elif days_diff < 10 * 365:
        ax1.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    else:
        ax1.xaxis.set_major_locator(mdates.YearLocator(1))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    ax1.grid(True, zorder=0)
    fig.autofmt_xdate()
    plt.tight_layout()

    # plt.xticks(rotation=45)

    plt.show()

    

def standardoverlapping(df, highlight):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    columns = list(enumerate(numeric_columns))

    print("The parameters are:\n")
    for i in range(0, len(columns), 2):
        col1 = f"{columns[i][0]}: {columns[i][1]}"
        col2 = f"{columns[i+1][0]}: {columns[i+1][1]}" if i + 1 < len(columns) else ""
        print(f"{col1:<30} {col2}")

    selected_indices = input("Enter indices of parameters to plot (comma-separated): ")
    selected_indices = [int(i) for i in selected_indices.split(",")]
    selected_params = [numeric_columns[i] for i in selected_indices]

    print("Chosen parameters:", selected_params)
    
    selected_window = input("Enter rolling window of parameters to plot (comma-separated): ")
    selected_window = [int(i) for i in selected_window.split(",")]

    df['Datum'] = pd.to_datetime(df['Datum'])
    
    
    if "Escherichia coli MPN [ant/100ml]" in selected_params:
        ecoli = int(input("Thresholdvalue for ecoli? (1/0): "))
    else:
        ecoli = 0
    
    
    
    start_date = input("Choose start date (YYYY-MM-DD): ")
    end_date = input("Choose end date (YYYY-MM-DD): ")

    reg_line = int(input("Do you want regression line (1/0): "))

    test = int(input("Normalized data? (1 or 0): "))
    test_2 = int(input("Intake closed? (1 or 0): "))

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    df = df[(df['Datum'] >= start) & (df['Datum'] <= end)]

    special_params = ["Stängd tid [h]", "Stängda dagar"]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    if ecoli == 1:
        ecoli_param_name = "Escherichia coli MPN [ant/100ml]"
        ecoli_in_plot = ecoli_param_name in selected_params
    
    
    for param in selected_params:
        valid_data = df[['Datum', param]].dropna()

        if param not in special_params:
            for window in selected_window:
                valid_data['Rolling_Mean'] = valid_data[param].rolling(window=window, min_periods=1).mean()
        else:
            valid_data['Rolling_Mean'] = valid_data[param]

        ax1.plot(valid_data["Datum"], valid_data['Rolling_Mean'], label=f"{param}")

        if reg_line == 1:
            x_dates = valid_data['Datum']
            x_num = mdates.date2num(x_dates)

            trend = np.polyfit(x_num, valid_data[param], 1)
            fit = np.poly1d(trend)

            # Plot the trendline
            x_fit = np.linspace(x_num.min(), x_num.max(), 100)
            ax1.plot(mdates.num2date(x_fit), fit(x_fit), color="red", linestyle="--", linewidth=2, label=f"Trendlinje")

            k, m = fit
            ax1.text(0.95, 0.9 - selected_params.index(param)*0.05, f'{param}: k={k*365:.1e}',
                     horizontalalignment='right', verticalalignment='top', transform=ax1.transAxes,
                     fontsize=10, color='black', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    
    if ecoli == 1:
        ax1.axhline(
            y=100,
            color='red',
            linestyle=':',
            linewidth=1.5,
            label="Riktvärde E. coli"
        )

    
    
    if test_2 == 1:
        highlight['Intag stängt'] = pd.to_datetime(highlight['Intag stängt'])
        highlight['Intaget öppet'] = pd.to_datetime(highlight['Intaget öppet'])
        for _, row in highlight.iterrows():
            ax1.axvspan(row['Intag stängt'], row['Intaget öppet'], color='skyblue', alpha=0.3)

    ax1.set_ylabel("Normaliserat värde" if test == 1 else f"{param}")
    ax1.tick_params(axis='y')
    plt.xlim(df["Datum"].min(), df["Datum"].max())

    days_diff = abs((end - start).days)
    if days_diff < 190:
        ax1.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    elif days_diff < 365:
        ax1.xaxis.set_minor_locator(mdates.DayLocator(interval=2))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    elif days_diff < 4 * 365:
        ax1.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    elif days_diff < 10 * 365:
        ax1.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    else:
        ax1.xaxis.set_major_locator(mdates.YearLocator(1))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    ax1.grid(True, zorder=0)
    plt.xticks(rotation=45)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()



def overlapping_three_params(df, highlight):
    columns = list(enumerate(df.select_dtypes(include=[np.number]).columns))

    print("The parameters are:\n")
    for i in range(0, len(columns), 2):
        col1 = f"{columns[i][0]}: {columns[i][1]}"
        col2 = f"{columns[i+1][0]}: {columns[i+1][1]}" if i + 1 < len(columns) else ""
        print(f"{col1:<30} {col2}")

    param1 = df.select_dtypes(include=[np.number]).columns[int(input("Choose the first parameter (index): "))]
    param2 = df.select_dtypes(include=[np.number]).columns[int(input("Choose the second parameter (index): "))]
    param3 = df.select_dtypes(include=[np.number]).columns[int(input("Choose the third parameter (index): "))]

    print("Chosen parameters: ", param1, param2, param3)

    df['Datum'] = pd.to_datetime(df['Datum'])

    ecoli_param_name = "Escherichia coli MPN [ant/100ml]"
    ecoli = ecoli_param_name in [param1, param2, param3] and int(input("Do you want threshold value? (1/0): "))

    start_date = input("Choose start date (YYYY-MM-DD): ")
    end_date = input("Choose end date (YYYY-MM-DD): ")

    window_1 = int(input(f"Floating mean window for {param1} (1 is mean for one day): "))
    window_2 = int(input(f"Floating mean window for {param2} (1 is mean for one day): "))
    window_3 = int(input(f"Floating mean window for {param3} (1 is mean for one day): "))

    test = int(input("Closed time (1/0): "))

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    df = df[(df['Datum'] >= start) & (df['Datum'] <= end)]

    special_params = ["Stängd tid [h]", "Stängda dagar"]

    fig, ax1 = plt.subplots(figsize=(14, 6))

    def prepare_data(param, window):
        valid = df[['Datum', param]].dropna()
        if param not in special_params:
            valid['Rolling_Mean'] = valid[param].rolling(window=window, min_periods=1).mean()
        else:
            valid['Rolling_Mean'] = valid[param]
        return valid

    d1 = prepare_data(param1, window_1)
    d2 = prepare_data(param2, window_2)
    d3 = prepare_data(param3, window_3)

    line1, = ax1.plot(d1['Datum'], d1['Rolling_Mean'], label=param1, color="darkorange")
    ax1.set_ylabel(param1)
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    line2, = ax2.plot(d2['Datum'], d2['Rolling_Mean'], label=param2, color="green")
    ax2.set_ylabel(param2)
    ax2.tick_params(axis='y')

    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.15))
    line3, = ax3.plot(d3['Datum'], d3['Rolling_Mean'], label=param3)
    ax3.set_ylabel(param3)
    ax3.tick_params(axis='y')

    if test == 1:
        highlight['Intag stängt'] = pd.to_datetime(highlight['Intag stängt'])
        highlight['Intaget öppet'] = pd.to_datetime(highlight['Intaget öppet'])
        for _, row in highlight.iterrows():
            ax1.axvspan(row['Intag stängt'], row['Intaget öppet'], color='skyblue', alpha=0.3)

    handles = [line1, line2, line3]
    if ecoli == 1:
        if ecoli_param_name == param1:
            ecoli_ax = ax1
        elif ecoli_param_name == param2:
            ecoli_ax = ax2
        else:
            ecoli_ax = ax3
        ecoli_line = ecoli_ax.axhline(y=100, color='red', linestyle=':', linewidth=1.5, label="Riktvärde E.coli")
        handles.append(ecoli_line)

    days_diff = (end - start).days
    if days_diff < 190:
        ax1.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    elif days_diff < 365:
        ax1.xaxis.set_minor_locator(mdates.DayLocator(interval=2))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    elif days_diff < 4 * 365:
        ax1.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    elif days_diff < 10 * 365:
        ax1.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    else:
        ax1.xaxis.set_major_locator(mdates.YearLocator(1))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    ax1.grid(True, zorder=0)
    ax1.legend(handles=handles, loc='upper left')
    fig.autofmt_xdate()
    ax1.set_xlim([start, end])

    plt.tight_layout()
    # plt.xticks(rotation=45)
    plt.show()
    
    
def scatter_plot(df):
    columns = list(enumerate(df.select_dtypes(include=[np.number]).columns))

    print("The parameters are:\n")
    for i in range(0, len(columns), 2):
        col1 = f"{columns[i][0]}: {columns[i][1]}"
        col2 = f"{columns[i+1][0]}: {columns[i+1][1]}" if i + 1 < len(columns) else ""
        print(f"{col1:<30} {col2}")

    param1 = df.select_dtypes(include=[np.number]).columns[int(input("Choose the first parameter (index): "))]
    param2 = df.select_dtypes(include=[np.number]).columns[int(input("Choose the second parameter (index): "))]

    if param1 not in df.columns or param2 not in df.columns:
        print("Error: One or both of the parameters are not in the df")
        return

    print("Chosen parameters: ", param1, param2)

    window_1 = int(input(f"Floating mean window for {param1} (1 is mean for one day): "))
    window_2 = int(input(f"Floating mean window for {param2} (1 is mean for one day): "))
    


    valid_data = df[[param1, param2]].copy()
    
    valid_data[param1] = valid_data[param1].rolling(window=window_1, min_periods=1).mean()
    valid_data[param2] = valid_data[param2].rolling(window=window_2, min_periods=1).mean()
    
    print(len(valid_data[param1]))
    print(len(valid_data[param2]))

    valid_data = valid_data.dropna()
    
    print(len(valid_data))

    


    corr = valid_data[param1].corr(valid_data[param2], method="spearman")


    b, m = polyfit(valid_data[param1], valid_data[param2], 1)

    plt.figure(figsize=(13, 6))
    plt.plot(valid_data[param1], b + m * valid_data[param1], '-', color="red", linewidth=1.5)

    plt.scatter(valid_data[param1], valid_data[param2], s=2.5, alpha=0.7)
    plt.xlabel(param1)
    plt.ylabel(param2)
    plt.text(0.95, 0.95, f'Korrelationskoefficient = {round(corr, 2)}',
      horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes, 
      fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    plt.grid(True)
    plt.show()


#%% Plotting Histogram
histo(Master_DataFrame, 50)

# 1. df
# 2. number of bins



#%% Plotting 2 parameters on each other. 


overlapping(Master_DataFrame, redline)




#%% Overlapping but with one axis

standardoverlapping(Master_DataFrame, redline)


#%% Overlapping three params


overlapping_three_params(Master_DataFrame, redline)

#%% Scatterplot for specific params with a rolling mean

scatter_plot(Master_DataFrame)


