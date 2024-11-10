from setup import Session
from pyspark.sql.functions import mean
import plotly.express as px
import pycountry
import streamlit as st
from pyspark.sql.functions import lit,col
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import functions as F


def read_file(uploaded_file):
    spark=Session()
    df=spark.read_file(uploaded_file)
    return df

def fillna_mean(df, include=set()): 
    means = df.agg(*(
        mean(x).alias(x) for x in df.columns if x in include
    ))
    return df.fillna(means.first().asDict())

def prepare_temperature_data(df, selected_areas, start_year, end_year):
    # Filter the dataset by selected areas
    df_filtered = df.filter(df['Area'].isin(selected_areas))
    
    # Select relevant years based on the range
    selected_years = [f'Y{year}' for year in range(start_year, end_year + 1)]
    
    # Melt the data to long format, where each row corresponds to a month, year, and temperature change
    df_long = df_filtered.select(
        'Area', 'Months', *selected_years
    ).withColumn(
        "Month_Index", F.expr("CASE WHEN Months = 'January' THEN 1 WHEN Months = 'February' THEN 2 " +
                              "WHEN Months = 'March' THEN 3 WHEN Months = 'April' THEN 4 " +
                              "WHEN Months = 'May' THEN 5 WHEN Months = 'June' THEN 6 " +
                              "WHEN Months = 'July' THEN 7 WHEN Months = 'August' THEN 8 " +
                              "WHEN Months = 'September' THEN 9 WHEN Months = 'October' THEN 10 " +
                              "WHEN Months = 'November' THEN 11 WHEN Months = 'December' THEN 12 END")
    )
    
   # Reshape the dataset so each year becomes a row instead of a column
    stack_expr = f"stack({len(selected_years)}, " + ", ".join(
        [f"'{year}', Y{year}" for year in range(start_year, end_year + 1)]
    ) + ") as (Year, Temperature_Change)"
    
    df_melted = df_long.selectExpr("Area", "Months", "Month_Index", stack_expr)
    
    # Sort the data by area, month, and year for better visualization
    df_sorted = df_melted.orderBy('Area', 'Month_Index', 'Year')
    
    return df_sorted


# Function to plot the data
def plot_temperature_change(df_long):
    # Convert PySpark DataFrame to Pandas for plotting
    pd_df = df_long.toPandas()

    # Plotting the data using matplotlib
    plt.figure(figsize=(10, 6))
    for area in pd_df['Area'].unique():
        area_data = pd_df[pd_df['Area'] == area]
        for year in area_data['Year'].unique():
            year_data = area_data[area_data['Year'] == year]
            plt.plot(year_data['Months'], year_data['Temperature_Change'], marker='o', label=f"{area} - {year}")
    
    plt.title('Temperature Change Over Months')
    plt.xlabel('Months')
    plt.ylabel('Temperature Change')
    plt.xticks(rotation=90)
    plt.legend(title='Area and Year', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    st.pyplot(plt)


def get_top10_countries(df_selected, country_codes, selected_year):
        """
        Retrieves the top 10 countries with the highest temperature changes for the selected year.
        Returns a list of dictionaries with Area and TemperatureChange.
        """
        selected_year_col = f'Y{selected_year}'

        # Select Area and TemperatureChange for the selected year
        df_year = df_selected.select('Area', selected_year_col).withColumnRenamed(selected_year_col, 'TemperatureChange')

        # Order by TemperatureChange descending and take top 10
        top10_df = df_year.orderBy(col('TemperatureChange').desc()).limit(10)

        # Collect data
        top10_data = []
        for row in top10_df.collect():
            area = row['Area']
            temp_change = row['TemperatureChange']
            top10_data.append({
                'Area': area,
                'TemperatureChange': temp_change
            })

        return top10_data



def map_area_to_country_code(df_selected):
    """
    Maps 'Area' names to ISO Alpha-3 country codes using PyCountry.
    Returns a dictionary mapping Area to CountryCode.
    """
    # Collect distinct areas
    areas = df_selected.select('Area').distinct().rdd.flatMap(lambda x: x).collect()



    area_to_code = {}
    for area in areas:
        try:
            #corrected_name = name_corrections.get(area, area)
            country = pycountry.countries.lookup(area)
            area_to_code[area] = country.alpha_3
        except:
            # Handle areas without a corresponding country code
            area_to_code[area] = None

    return area_to_code


def prepare_plot_data(df_selected, country_codes, year_columns):
    """
    Prepares data for Plotly visualization.
    Returns a list of dictionaries with Area, CountryCode, Year, TemperatureChange.
    """
    plot_data = []
    for row in df_selected.collect():
        area = row['Area']
        country_code = country_codes.get(area)
        if not country_code:
            continue  # Skip areas without a valid country code

        for year_col in year_columns:
            year = int(year_col[1:])  # Extract year as integer
            temp_change = row[year_col]
            plot_data.append({
                'Area': area,
                'CountryCode': country_code,
                'Year': year,
                'TemperatureChange': temp_change
            })

    return plot_data


def get_top_10_areas(df, year_columns):
    """
    Calculate the average temperature change for each area and return the top 10.
    """
    # Calculate average temperature change across the selected years
    avg_temp_df = df.select('Area', *year_columns) \
                    .withColumn('Average_Temperature_Change', 
                                sum([df[year_col] for year_col in year_columns]) / len(year_columns)) \
                    .groupBy('Area') \
                    .agg({'Average_Temperature_Change': 'avg'}) \
                    .orderBy('avg(Average_Temperature_Change)', ascending=False)

    # Convert to Pandas DataFrame to easily sort and filter the top 10 areas
    top_areas = avg_temp_df.limit(10).toPandas()

    return top_areas

def generate_explanation(start_year, end_year,top_10_areas=None,selected_areas=None):
    """
    Generates an explanation for the selected area and year range.
    """
    if selected_areas!=None:
        explanation = f"""
        You have selected **{', '.join(selected_areas)}** for the years **{start_year} to {end_year}**.
        The graph shows the temperature changes for each month in the selected years.

        The lines in the graph represent how the temperature changed from January to December over the selected years.
        You can observe fluctuations in temperature change over time, which may indicate climate variability.
        """
        return explanation
    else:
       
       top_area = top_10_areas.Area[0]
       top_temp_change = top_10_areas['avg(Average_Temperature_Change)'][0]
       explanation= f"""
       
       
       This graph shows the top 10 areas with the highest average temperature change between {start_year} and {end_year}.\n\nEach bar represents a region, and the height of the bar corresponds to the average temperature change over the selected time period.
       The color scale indicates the intensity of the temperature change, where darker red shows areas with the highest increase.\n\n**{top_area}** has the highest average temperature change of **{top_temp_change:.2f}°C** during this period, indicating significant climate change impacts in the region.\n\nOther areas in the top 10 include {', '.join(top_10_areas['Area'].iloc[1:].values)}.\n\nThese regions have shown substantial temperature increases, which could be due to a variety of factors such as global warming, deforestation, and urbanization. \n\nThese changes can have profound impacts on ecosystems and human societies.
       
       """
       return explanation
    
def calculate_seasonal_temperatures(df, start_year, end_year):
    """
    Calculates the average temperature for each season (Spring, Summer, Fall, Winter)
    for the selected range of years.
    """
    selected_years = [f'Y{year}' for year in range(start_year, end_year + 1)]
    
    df_spring = df.select('Area', 'Months', *selected_years).filter(F.col('Months').isin('March', 'April', 'May'))
    df_summer = df.select('Area', 'Months', *selected_years).filter(F.col('Months').isin('June', 'July', 'August'))
    df_fall = df.select('Area', 'Months', *selected_years).filter(F.col('Months').isin('September', 'October', 'November'))
    df_winter = df.select('Area', 'Months', *selected_years).filter(F.col('Months').isin('December', 'January', 'February'))
    
    df_spring_avg = df_spring.select([F.avg(F.col(year)).alias(f"{year}") for year in selected_years])
    df_summer_avg = df_summer.select([F.avg(F.col(year)).alias(f"{year}") for year in selected_years])
    df_fall_avg = df_fall.select([F.avg(F.col(year)).alias(f"{year}") for year in selected_years])
    df_winter_avg = df_winter.select([F.avg(F.col(year)).alias(f"{year}") for year in selected_years])
    
    spring_pd = df_spring_avg.toPandas().transpose().reset_index()
    summer_pd = df_summer_avg.toPandas().transpose().reset_index()
    fall_pd = df_fall_avg.toPandas().transpose().reset_index()
    winter_pd = df_winter_avg.toPandas().transpose().reset_index()

    spring_pd.columns = ['Year', 'Spring_Temperature']
    summer_pd.columns = ['Year', 'Summer_Temperature']
    fall_pd.columns = ['Year', 'Fall_Temperature']
    winter_pd.columns = ['Year', 'Winter_Temperature']

    return spring_pd, summer_pd, fall_pd, winter_pd