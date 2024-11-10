import streamlit as st
from projects.climate import read_file,fillna_mean,get_top_10_areas,map_area_to_country_code,prepare_plot_data,prepare_temperature_data,plot_temperature_change,generate_explanation,calculate_seasonal_temperatures
import numpy as np
import pyspark.sql.functions as F
from pyspark.sql.functions import col
import plotly.express as px
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
from collections import Counter

def welcome():
    st.header('Welcome to my project page')
def climate():    
    st.header("Global Temperature Change Visualization")
    df = read_file('dataset/Environment_Temperature_change_E_All_Data_NOFLAG.csv')
    df = fillna_mean(df, df.columns[8:])
    df_drop=df.drop('Element Code','Area Code','Element','Area Code (M49)','Months Code','Unit')
    # Define the list of months
    months_list = ['January', 'February', 'March', 'April', 'May', 'June', 
                'July', 'August', 'September', 'October', 'November', 'December']

    # Filter the DataFrame
    df_filtered = df_drop.filter(
        (col('Element') == "Temperature change") & (col('Months').isin(months_list))
    )
    with st.container():
        st.dataframe(df_filtered, use_container_width=True)
        
        tab1, tab2, tab3,tab4 = st.tabs(["Temperature Change Over Months", "Global Temperature Change Over Time", "Top 10 Areas with Highest Temperature Change","Average Temperature Change by Season"])

        with tab1:
            # Sidebar for Area and Year Range selection
            available_areas = [row['Area'] for row in df_filtered.select('Area').distinct().collect()]
            selected_areas = st.multiselect('Select Areas', available_areas, default=[available_areas[0]])
            start_year, end_year = st.slider('Select Year Range', 1961, 2023, (1961, 1965))
        

            with st.spinner("Loading...."):
                time.sleep(1)
            
            # Prepare and plot data
                if selected_areas and start_year <= end_year:
                    df_long = prepare_temperature_data(df_filtered, selected_areas, start_year, end_year)
                    plot_temperature_change(df_long)
                else:
                    st.write("Please select valid areas and a year range.")
                
                with st.expander("See explanation"):
                    explanation_text = generate_explanation(start_year, end_year,None,selected_areas)
                    st.markdown(explanation_text)


        with tab2:
            year_columns = [f'Y{year}' for year in range(1961, 2024)]
            df_selected = df.select('Area', *year_columns)  
                # Map areas to country codes
            country_codes = map_area_to_country_code(df_selected)

            # Prepare the data for the plot
            plot_data = prepare_plot_data(df_selected, country_codes, year_columns)

            # Convert the plot data to a Pandas DataFrame for Plotly
            plot_df = pd.DataFrame(plot_data)

            # Plotly choropleth for global temperature change visualization with animation
            fig = px.choropleth(
                plot_df,
                locations='CountryCode',
                color='TemperatureChange',
                hover_name='Area',
                animation_frame='Year',
                color_continuous_scale='RdYlBu',
                title='Global Temperature Change Over Time',
                labels={'TemperatureChange': 'Temperature Change (°C)'},
                projection='natural earth'
            )

            # Set play/pause button to true for auto-playing the animation
            fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 1000  # 1 second per frame
            fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 500  # Smooth transitions

            # Display the plot
            st.plotly_chart(fig)



        with tab3:

            # Streamlit UI components
            st.markdown('**Top 10 Areas with Highest Temperature Change**')

            # Sidebar for year range selection
            start_year_, end_year_ = st.slider('Select Year Range', 1961, 2023, (1961, 1970))

            # Filter the data to only the columns needed (area and selected years)
            year_columns = [f'Y{year}' for year in range(start_year_, end_year_ + 1)]
            df_selected = df.select('Area', *year_columns)

            # Get the top 10 areas with the highest temperature change
            top_10_areas = get_top_10_areas(df_selected, year_columns)

            with st.spinner("Loading...."):
                time.sleep(1)
                # Plot the top 10 areas in a bar chart
                fig = px.bar(
                    top_10_areas,
                    x='Area',
                    y='avg(Average_Temperature_Change)',
                    title=f'Top 10 Areas with Highest Temperature Change ({start_year_} - {end_year_})',
                    labels={'avg(Average_Temperature_Change)': 'Average Temperature Change (°C)', 'Area': 'Area'},
                    color='avg(Average_Temperature_Change)',
                    color_continuous_scale='RdYlBu'
                )

                # Display the plot
                st.plotly_chart(fig)
                with st.expander("See explanation"):
                    explanation_text = generate_explanation(start_year, end_year,top_10_areas,None)
                    st.markdown(explanation_text)
        with tab4:
            st.markdown('**Seasonal Temperature Change Visualization**')

            # Calculate seasonal average temperatures for the selected range of years
            spring_data, summer_data, fall_data, winter_data = calculate_seasonal_temperatures(df, 1961, 2023)
            

            # Plotting the seasonal temperature changes
            plt.figure(figsize=(10, 6))
            plt.plot(spring_data['Year'], spring_data['Spring_Temperature'], label="Spring's average temperature", color='green')
            plt.plot(summer_data['Year'], summer_data['Summer_Temperature'], label="Summer's average temperature", color='orange')
            plt.plot(fall_data['Year'], fall_data['Fall_Temperature'], label="Fall's average temperature", color='red')
            plt.plot(winter_data['Year'], winter_data['Winter_Temperature'], label="Winter's average temperature", color='blue')

            plt.title('Average Temperature Change Over Seasons')
            plt.xlabel('Year')
            plt.xticks(rotation=90)
            plt.ylabel('Average Temperature (°C)')
            plt.legend()
            plt.grid(True)

            # Display the plot in Streamlit
            st.pyplot(plt)
            with st.expander("See explanation"):
                    explanation_text = generate_explanation(start_year, end_year,top_10_areas,None)
                    st.markdown(
                        f"""
This graph represents the **average temperature change** for each season (Spring, Summer, Fall, and Winter) 
from **{1961} to {2023}**. Each line corresponds to one of the four seasons, showing the trend 
in temperature changes across the selected time period.

#### Observations:
- **Winter** (blue line) appears to show a more rapid increase in average temperature over the years, particularly from the mid-1900s onwards.
- **Summer** (orange line) and **Spring** (green line) also show a steady increase in temperatures, but not as sharp as Winter.
- **Fall** (red line) displays a moderate temperature increase but is relatively stable compared to the other seasons.

#### Conclusion:
- The overall trend shows that temperatures are rising in all seasons, with **Winter** showing the largest temperature increases, 
which could suggest stronger effects of climate change during the colder months.
- These trends provide important insights into seasonal climate changes and can help in understanding the broader impacts of global warming.
"""


                    )

def music_mental_health():
    st.header('Mental Health & Music Relationship')
    df=read_file('dataset/mental_health_music/mxmh_survey_results.csv')
    df=df.toPandas()
    st.dataframe(df)

    # Remove columns whose null values are less than 10
    df.dropna(subset=['Age','Primary streaming service','While working','Instrumentalist','Composer','Foreign languages','Music effects'])

    df['BPM']=df['BPM'].fillna(df['BPM'].median())

    option = st.selectbox(
            "EDA and Analysis",
            ("Describe", "Correlation", "Pairplot","3D plot","Comparation","Statistic","Distribution","Survey","Conclusion"),
            index=None,
            placeholder="Select contact method...",
    )
    df1 = ['Age','Hours per day','BPM','Anxiety','Depression','Insomnia','OCD']
    if option=='Describe':
        x=df.describe()[1:].T.style.background_gradient(cmap='mako', axis=1)
        st.dataframe(x)

    elif option=='Correlation':
        
        def correlated_map(dataframe, plot=False):
            

            corr = dataframe[df1].corr()
            if plot:
                sns.set(rc={'figure.figsize': (20, 12)})
                sns.heatmap(corr, cmap="mako", annot=True, linewidths=.6 , cbar = False)
                plt.xticks(rotation=60, size=10)
                plt.yticks(size=10)
                plt.title('Analysis of Correlations', size=14)
                st.pyplot(plt)
        correlated_map(df, plot=True)
        with st.expander('Explaination'):
            st.write('Anxiety, Depression, OCD, and Insomina are positively correlated. Other numerical variables are not correlated, indicating no linear relationship.')

        sel = ['Age', 'Hours per day', 'While working', 'Exploratory', 'BPM','Anxiety', 'Depression', 'Insomnia', 'OCD', 'Music effects']
        freq = {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Very frequently': 3}
        df_genre = df[[col for col in df.columns if col.startswith('Frequency')]]
        df_genre.replace(freq, inplace=True)
        df_sel = pd.concat([df[sel], df_genre], axis=1)
        df_sel['While working'] = df_sel['While working'].map({'Yes': 1, 'No': 0})
        df_sel['Exploratory'] = df_sel['Exploratory'].map({'Yes': 1, 'No': 0})
        df_sel['Music effects'] = df_sel['Music effects'].map({'Improve': 1, 'No effect': 0, 'Worsen': -1})
        plt.figure(figsize=(12, 10))
        sns.heatmap(df_sel.corr(), annot=False, cmap='mako')
        plt.title('Correlation Between Music Listening and Mental Health')
        st.pyplot(plt)
        with st.expander('Explaination'):
            st.write("This heatmap doesn't give us clear insights into which features significantly improve mental health according to the listeners. However, it does show some cool connections between music genres: people who listen to hip-hop often also enjoy rap and R&B, and metal fans tend to like rock as well. Plus, it shows that anxiety and depression are linked, which makes sense since they’re both mental health problems.")

       

    elif option=='Pairplot':

        sns.pairplot(data=df[df1], diag_kind='kde',hue="BPM",palette='PRGn',corner=True)
        st.pyplot(plt)

        st.markdown('**Age**')
        ff = ['Hours per day','BPM','Anxiety','Depression','Insomnia','OCD']
        fig=plt.figure(figsize=(25,18))
        for i,col in enumerate(ff):
            
            ax=fig.add_subplot(3,2,i+1)
            ax.title.set_text(f'Age vs {col}')
            sns.scatterplot(x='Age',y=col,hue='Age',data=df,palette="mako")
        st.pyplot(plt)

        st.markdown('**Hours per day**')
        ff = ['Age','BPM','Anxiety','Depression','Insomnia','OCD']
        fig=plt.figure(figsize=(25,18))
        for i,col in enumerate(ff):
            
            ax=fig.add_subplot(4,2,i+1)
            ax.title.set_text(f'Hours per day vs {col}')
            sns.scatterplot(x='Hours per day',y=col,hue='Age',data=df,palette="mako")
        st.pyplot(plt)


        st.markdown('**BPM**')
        ff = ['Age','Hours per day','Anxiety','Depression','Insomnia','OCD']
        fig=plt.figure(figsize=(25,18))
        for i,col in enumerate(ff):
            
            ax=fig.add_subplot(3,2,i+1)
            ax.title.set_text(f'BPM vs {col}')
            sns.scatterplot(x='BPM',y=col,hue='Age',data=df,palette="mako")
        st.pyplot(plt)



        st.markdown('**Anxiety**')
        ff = ['Age','Hours per day','BPM','Depression','Insomnia','OCD']
        fig=plt.figure(figsize=(25,18))
        for i,col in enumerate(ff):
            
            ax=fig.add_subplot(3,2,i+1)
            ax.title.set_text(f'Anxiety vs {col}')
            sns.scatterplot(x='Anxiety',y=col,hue='Age',data=df,palette="mako")
        
        st.pyplot(plt)



        st.markdown('**Depression**')
        ff = ['Age','Hours per day','BPM','Anxiety','Insomnia','OCD']
        fig=plt.figure(figsize=(25,18))
        for i,col in enumerate(ff):
            
            ax=fig.add_subplot(3,2,i+1)
            ax.title.set_text(f'Depression vs {col}')
            sns.scatterplot(x='Depression',y=col,hue='Age',data=df,palette="mako")
        st.pyplot(plt)

        st.markdown('**Insomnia**')
        ff = ['Age','Hours per day','BPM','Anxiety','Depression','OCD']
        fig=plt.figure(figsize=(25,18))
        for i,col in enumerate(ff):
            
            ax=fig.add_subplot(3,2,i+1)
            ax.title.set_text(f'Insomnia vs {col}')
            sns.scatterplot(x='Insomnia',y=col,hue='Age',data=df,palette="mako")
        
        st.pyplot(plt)



        st.markdown('**OCD**')
        ff = ['Age','Hours per day','BPM','Anxiety','Depression','Insomnia']
        fig=plt.figure(figsize=(25,18))
        for i,col in enumerate(ff):
            ax=fig.add_subplot(3,2,i+1)
            ax.title.set_text(f'OCD vs {col}')
            sns.scatterplot(x='OCD',y=col,hue='Age',data=df,palette="mako")
        
        st.pyplot(plt)

    elif option=="3D plot":
       # Assuming 'df' is your dataframe containing 'Insomnia', 'Anxiety', 'Depression', and 'Age' columns
        fig = px.scatter_3d(df, x='Insomnia', y='Anxiety', z='Depression', 
                            color='Age', color_continuous_scale="PRGn", template='plotly_white')

        # Update marker size
        fig.update_traces(marker=dict(size=5))

        # Display the 3D scatter plot using Streamlit
        st.plotly_chart(fig)


    elif option=='Comparation':

        fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        sns.kdeplot(x='Age', y='Hours per day', data=df, ax=axs[0, 0], color='#065A60')
        axs[0, 0].set_title('Age vs Hours per day')
        axs[0, 0].grid()
        sns.kdeplot(x='Depression', y='Anxiety', data=df, ax=axs[0, 1], color='#144552')
        axs[0, 1].set_title('Depression vs Anxiety')
        axs[0, 1].grid()
        sns.kdeplot(x='Anxiety', y='Insomnia', data=df, ax=axs[1, 0], color='#212F45')
        axs[1, 0].set_title('Anxiety vs Insomnia')
        axs[1, 0].grid()
        sns.kdeplot(x='Depression', y='Hours per day', data=df, ax=axs[1, 1], color='#312244')
        axs[1, 1].set_title('Depression vs Hours per day')
        axs[1, 1].grid()
        plt.tight_layout()
        st.pyplot(plt)
        with st.expander('Explaination'):
            st.write('Densities are visible. These plots show that young people spend more time on music.')

    elif option=='Statistic':
        columns1 = ['Age','Hours per day','BPM','Anxiety','Depression','Insomnia','OCD']
        palette ="mako"
        for column in columns1:
            plt.figure(figsize=(15,2))
            sns.violinplot(x=df[column], palette=palette)
            plt.title(column)
            st.pyplot(plt)

        palette ="mako"
        for column in columns1:
            plt.figure(figsize=(15,2))
            sns.boxplot(x=df[column], palette=palette)
            plt.title(column)
            stats = df[column].describe()
            stats_text = ", ".join([f"{key}: {value:.2f}" for key, value in stats.items()])
            st.markdown(f"\n\n{column} Statistics:\n\n{stats_text}")
            st.pyplot(plt)
        

        feature_list = ['Age','Hours per day','BPM','Anxiety','Depression','Insomnia','OCD']
        def StDev_method (df,n,features):
            outliers = []
            for column in features:
                data_mean = df[column].mean()
                data_std = df[column].std()
                cut_off = data_std * 3     
                outlier_list_column = df[(df[column] < data_mean - cut_off) | (df[column] > data_mean + cut_off)].index
                outliers.extend(outlier_list_column)
            outliers = Counter(outliers)        
            OUT= list( k for k, v in outliers.items() if v > n )
            df1 = df[df[column] > data_mean + cut_off]
            df2 = df[df[column] < data_mean - cut_off]
            st.write('Total number of outliers is:', df1.shape[0]+ df2.shape[0])
            
            return OUT  
        Outliers_StDev = StDev_method(df,1,feature_list)
        df_out2 = df.drop(Outliers_StDev, axis = 0).reset_index(drop=True)
        with st.expander('Explaination'):
            st.write("""
                The dispersion of Each Feature at each specified feature can be seen in above plots. 
            """)
            st.write("""As you can tell, there are many outliers in the dataset, which is typical for this kind of data. Outliers are considered innocent until proven guilty, so unless there is a valid reason, they shouldn't be eliminated. The pairplot and scatterplots analysis doesn't show any noisy data and in the above code shows that total number of outliers is 0, so we will not be discarding any data.
            """)
   
    elif option=='Distribution':
        dff = df.drop(['Timestamp', 'Permissions'], axis=1)
        # Renaming frequency columns to only the name of the musical genre
        dff = (dff.rename(columns={
            'Frequency [Classical]': 'Classical',          
            'Frequency [Country]': 'Country',               
            'Frequency [EDM]': 'EDM',                  
            'Frequency [Folk]': 'Folk',                  
            'Frequency [Gospel]': 'Gospel',                
            'Frequency [Hip hop]': 'Hip hop',               
            'Frequency [Jazz]': 'Jazz',                  
            'Frequency [K pop]': 'K pop',                 
            'Frequency [Latin]': 'Latin',                 
            'Frequency [Lofi]': 'Lofi',                  
            'Frequency [Metal]': 'Metal',                 
            'Frequency [Pop]': 'Pop',                   
            'Frequency [R&B]': 'R&B',                  
            'Frequency [Rap]': 'Rap',               
            'Frequency [Rock]': 'Rock',                 
            'Frequency [Video game music]': 'Games music'}))
        
        st.html("""
            <p>The distribution of age records skews towards younger ages, particularly clustering around 18 years old, although our dataset covers a wide range up to 70 years old. Therefore, it's pertinent to analyze the characteristics of different age groups:</p>
                <ul>
                    <li>Teenagers: Up to 19 years old</li>
                    <li>Adults: Between 20 and 59 years old</li>
                    <li>Seniors: 60 years old and above</li>
                </ul>
            <p>To simplify the plotting process, we will add a new column in our dataset indicating the age bracket of each individual.</p>
            """)
        
        dff['Age group'] = pd.cut(dff['Age'], bins=[9, 19, 59, 70], labels=['Teenager', 'Adults', 'Seniors'])

        stream = (dff.groupby('Age group')['Primary streaming service'].agg(['value_counts']).reset_index().rename(columns={'value_counts':'quantity'}))


        Teenagers = pd.DataFrame(stream.query('`Age group` == "Teenager"'))
        Adults = pd.DataFrame(stream.query('`Age group` == "Adults"'))
        Seniors = pd.DataFrame(stream.query('`Age group` == "Seniors"'))


        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
        sns.barplot(
            x='quantity', 
            y='Primary streaming service',
            data=Teenagers, 
            edgecolor='black', 
            palette='mako',
            ax=ax1)
        for container in ax1.containers:
            ax1.bar_label(container, padding=3)
        ax1.set_title('Responses from teenagers aged 10 to 19 regarding their preferred streaming service.', fontsize=15)
        ax1.set_xlabel('Number of users', fontsize=12)
        ax1.set_ylabel('')
        quantity = list(Teenagers['quantity'].values)
        services = list(Teenagers['Primary streaming service'].values)
        colors = ['#5e60ce', '#5390d9', '#4ea8de', '#48bfe3', '#56cfe1','#64dfdf']
        explode = (0.02, 0.02, 0.02, 0.02, 0.02, 0.02)
        wedges, texts, autotexts = ax2.pie(
            quantity,
            labels=services,
            autopct='%1.1f%%',
            colors=colors,
            explode=explode)
        ax2.set_title('Distribution of Streaming Services reported by Teenagers (10-19 years old)', fontsize=15)
        for text in texts + autotexts:
            text.set_color('black')
        plt.tight_layout()
        st.pyplot(plt)
        
        with st.expander('Explaination'):
            st.write('Spotify is the most popular streaming service among teenagers.')
        

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
        sns.barplot(
            x='quantity', 
            y='Primary streaming service',
            data=Adults, 
            edgecolor='black', 
            palette='mako',
            ax=ax1)
        for container in ax1.containers:
            ax1.bar_label(container, padding=3)
        ax1.set_title('Responses from Adults aged 20 to 59 regarding their preferred streaming service', fontsize=15)
        ax1.set_xlabel('Number of users', fontsize=12)
        ax1.set_ylabel('')
        quantity = list(Adults['quantity'].values)
        services = list(Adults['Primary streaming service'].values)
        colors = ['#5e60ce', '#5390d9', '#4ea8de', '#48bfe3', '#56cfe1','#64dfdf']
        explode = (0.02, 0.02, 0.02, 0.02, 0.02, 0.02)
        wedges, texts, autotexts = ax2.pie(
            quantity,
            labels=services,
            autopct='%1.1f%%',
            colors=colors,
            explode=explode)
        ax2.set_title('Distribution of Streaming Services reported by Adults (20-59 years old)', fontsize=15)
        for text in texts + autotexts:
            text.set_color('black')
        plt.tight_layout()
        st.pyplot(plt)
        with st.expander('Explaination'):
            st.write('Spotify is the most popular streaming service among adults.')


        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
        sns.barplot(
            x='quantity', 
            y='Primary streaming service',
            data=Seniors, 
            edgecolor='black', 
            palette='mako',
            ax=ax1)
        for container in ax1.containers:
            ax1.bar_label(container, padding=3)
        ax1.set_title('Responses from Seniors aged +60 regarding their preferred streaming service.', fontsize=15)
        ax1.set_xlabel('Number of users', fontsize=12)
        ax1.set_ylabel('')
        quantity = list(Seniors['quantity'].values)
        services = list(Seniors['Primary streaming service'].values)
        colors = ['#5e60ce', '#5390d9', '#4ea8de', '#48bfe3', '#56cfe1','#64dfdf']
        explode = (0.02, 0.02, 0.02, 0.02, 0.02, 0.02)
        wedges, texts, autotexts = ax2.pie(
            quantity,
            labels=services,
            autopct='%1.1f%%',
            colors=colors,
            explode=explode)
        ax2.set_title('Distribution of Streaming Services reported by Seniors (+60 years old)', fontsize=15)
        for text in texts + autotexts:
            text.set_color('black')
        plt.tight_layout()
        st.pyplot(plt)
        with st.expander('Explaination'):
            st.write('YouTube Music is the most popular streaming service among teenagers.')




    

        ww = dff["Primary streaming service"]
        plt.figure(figsize=(12, 4))
        plt.xticks(rotation=50)
        sns.countplot(x=ww,palette='mako')
        st.pyplot(plt)
        with st.expander('Explaination'):
            st.write(' Overall, Spotify and YouTube Music are the most popular streaming services.')
        

        plt.figure(figsize=(22, 10))
        sns.histplot(dff, y="Anxiety",palette='mako',hue="Age group", multiple="stack")
        st.pyplot(plt)
            
        
        plt.figure(figsize=(22, 10))
        sns.histplot(dff, y="Depression",bins=10,palette='mako',hue="Age group", multiple="stack")
        st.pyplot(plt)



        plt.figure(figsize=(22, 10))
        sns.histplot(dff, y="Insomnia",palette='mako',hue="Age group", multiple="stack")
        st.pyplot(plt)


        plt.figure(figsize=(22, 10))
        sns.histplot(dff, y="OCD",palette='mako',hue="Age group", multiple="stack")
        st.pyplot(plt)

        plt.figure(figsize=(22, 10))
        sns.histplot(dff, y="Fav genre",palette='mako',hue="Age group", multiple="stack")
        st.pyplot(plt)

    elif option=='Survey':
        dff = df.drop(['Timestamp', 'Permissions'], axis=1)
        # Renaming frequency columns to only the name of the musical genre
        dff = (dff.rename(columns={
            'Frequency [Classical]': 'Classical',          
            'Frequency [Country]': 'Country',               
            'Frequency [EDM]': 'EDM',                  
            'Frequency [Folk]': 'Folk',                  
            'Frequency [Gospel]': 'Gospel',                
            'Frequency [Hip hop]': 'Hip hop',               
            'Frequency [Jazz]': 'Jazz',                  
            'Frequency [K pop]': 'K pop',                 
            'Frequency [Latin]': 'Latin',                 
            'Frequency [Lofi]': 'Lofi',                  
            'Frequency [Metal]': 'Metal',                 
            'Frequency [Pop]': 'Pop',                   
            'Frequency [R&B]': 'R&B',                  
            'Frequency [Rap]': 'Rap',               
            'Frequency [Rock]': 'Rock',                 
            'Frequency [Video game music]': 'Games music'}))
        fig, ax = plt.subplots(figsize=(12, 10), nrows=2, ncols=2)
        colors = ['#5e60ce', '#5390d9'] 
        df['While working'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax[0, 0], colors=colors, title='While working')
        df['Instrumentalist'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax[0, 1], colors=colors, title='Instrumentalist')
        df['Composer'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax[1, 0], colors=colors, title='Composer')
        df['Exploratory'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax[1, 1], colors=colors, title='Exploratory')
        plt.suptitle("Music Listener's Survey Responses")
        plt.tight_layout()
        st.pyplot(plt)

        st.write("Does music improve/worsen respondent's mental health conditions?")

        colors = ['#5e60ce', '#5390d9','#4ea8de','#48bfe3'] 
        df['Music effects'].value_counts().plot.pie(autopct='%1.1f%%',colors=colors)
        plt.title('Music Effects on Mental Health')



        mentalsss = ["Anxiety", "Depression", "Insomnia", "OCD"]
        mental = df[mentalsss]
        mental.round(0).astype(int)
        disorder_count = []
        for disorder in mentalsss:
            x=0
            while x !=11:
                count =  (mental[disorder].values == x).sum()
                disorder_count.append(count)
                x +=1
        labels = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        x = np.arange(len(labels))
        width = 0.15
        fig, ax = plt.subplots(figsize=(35, 15))
        o = ax.bar(x-2*width, disorder_count[0:11], width, label="Anxiety", color = '#212F45')
        oo = ax.bar(x-width, disorder_count[11:22], width, label="Depression", color = '#272640')
        oo = ax.bar(x, disorder_count[22:33], width, label="Insomnia", color = '#312244')
        oo = ax.bar(x+width, disorder_count[33:], width, label="OCD", color = '#144552')
        ax.set_ylim([0, 170])
        ax.set_ylabel('Number of Rankings')
        ax.set_xlabel('Ranking')
        ax.set_title('Mental health ranking distribution')
        ax.set_xticks(x, labels)
        ax.legend()
        st.pyplot(plt)



        figure, axes = plt.subplots(4, 4, figsize=(18, 10))
        genres = ['Classical', 'Country', 'EDM', 'Folk', 'Gospel', 'Hip hop', 'Jazz', 'K pop', 'Latin', 'Lofi', 'Metal', 'Pop', 'R&B', 'Rap', 'Rock', 'Games music']
        palette = 'mako'
        for i, genre in enumerate(genres):
            row = i // 4
            col = i % 4
            ax = sns.countplot(ax=axes[row, col], x=dff[genre], palette=palette)
            total = len(dff[genre])
            for p in ax.patches:
                percentage = '{:.1f}%'.format(100 * p.get_height() / total)
                x = p.get_x() + p.get_width() / 2 - 0.05
                y = p.get_height()
                ax.annotate(percentage, (x, y), ha='center', va='bottom')
        plt.tight_layout()
        st.pyplot(plt)

        with st.expander('Expalaination'):
            st.write("Approximately 35% of people rarely listen to classical music, while 22% never listen to it at all, and 15% listen very frequently. Country music is even less popular, with 50% of people never listening to it, 36% listening rarely, and only 7% listening very frequently. EDM (Electronic Dance Music) is never listened to by 42% of people, but 11% listen to it very frequently. Folk music sees similar patterns, with 39% of people never listening to it and 10% listening very frequently. Gospel music is particularly unpopular, as nearly 75% of people do not listen to it. Hip hop, on the other hand, is quite popular. Although 23% of people do not listen to it, the remaining 77% do, making it a widely enjoyed genre. Jazz music has fallen out of favor, with 35% of people never listening to it and 33% listening rarely. K-pop also sees low listenership, with 56% of people never listening to it. Latin music shares a similar fate, with 60% of people never listening to it. Lofi music has a more balanced audience: 36% listen to it and 32% listen more than rarely. Metal music is quite popular, with 70% of people listening to it at least sometimes or very frequently. Pop music stands out as the most popular genre, with 93% of people listening to it and 36% of them listening very frequently. R&B, however, is less popular: 30% of people do not listen to it, 28% listen rarely, and 24% listen sometimes. Rap music has a varied audience: 26% never listen to it, 30% listen rarely, and the rest listen more often. Rock music is very popular, with 44% of people listening very frequently and 28% listening sometimes. Lastly, video game music is never listened to by 32% of people.")


        figure,axes=plt.subplots(2,2,figsize=(30,15))
        sns.lineplot(ax=axes[0,0],x=dff['Fav genre'],y=dff['Insomnia'],ci=None,color='#4D194D')
        sns.lineplot(ax=axes[0,1],x=dff['Fav genre'],y=dff['OCD'],ci=None,color='#272640')
        sns.lineplot(ax=axes[1,0],x=dff['Fav genre'],y=dff['Depression'],ci=None,color='#1B3A4B')
        sns.lineplot(ax=axes[1,1],x=dff['Fav genre'],y=dff['Anxiety'],ci=None,color='#006466')
        plt.tight_layout()
        for ax in axes.flat:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
        st.pyplot(plt)

        st.html("""
         

    <p>Participants exhibiting high levels of insomnia show a preference for gospel music over country music. Those with high OCD tend to favor Lofi and Rap genres the most. Similarly, individuals experiencing high levels of depression show a preference for Lofi and Hip hop music. Lastly, participants with high levels of anxiety tend to listen most to Folk, K-pop, Jazz, and Rock music genres.</p>
    <ul>
    <li><b>Insomnia</b>: Individuals with insomnia may find solace in calming and soothing music like gospel, which could potentially help them relax or distract from their sleep difficulties.</li>

    <li><b>OCD</b>: Lofi and Rap music genres, characterized by repetitive beats and rhythms, might appeal to individuals with OCD tendencies who seek comfort in structured patterns or rhythms.</li>

    <li><b>Depression</b>: Lofi and Hip hop music, often expressing introspective themes and emotions, could resonate with those experiencing depression, offering a way to connect emotionally with the music.</li>

   <li><b> Anxiety</b>: Folk, K-pop, Jazz, and Rock music genres vary widely in style but may offer different forms of emotional release or distraction for individuals dealing with anxiety, depending on personal preference for rhythmic complexity, lyrical content, or instrumental elements.</li>

    </ul>    
                    
                    
    """)
        
        colors = ['#006466', '#065A60', '#0B525B', '#144552','#1B3A4B', '#212F45', '#272640', '#312244','#3E1F47', '#4D194D']
        fig = px.pie(dff, names='Fav genre', title='Genre Preferences',color_discrete_sequence=colors)
        st.plotly_chart(fig)




        colors = ['#312244', '#212F45', '#144552', '#006466']
        fig = px.sunburst(dff, path=["Fav genre"], values="Hours per day", color="Music effects",color_discrete_sequence=colors)
        total = dff["Hours per day"].sum()
        fig.update_traces(textinfo="label+percent entry")
        fig.update_layout(
            title="Top Fav Genre VS Hours per day",
            title_font={"size": 20},
            margin=dict(t=50, b=50, l=0, r=0))
        fig.update_layout(width=1000, height=800)
        st.plotly_chart(fig)


        sns.catplot(
            data=dff.sort_values("Fav genre"),
            x="Fav genre", y="BPM", kind="boxen",height=6, aspect=2,width = 0.5,showfliers=False, palette='mako')
        plt.xticks(rotation = 90)
        plt.title('Genre vs BPM')
        plt.ylim(50, 210)
        st.pyplot(plt)
    
    elif option=="Conclusion":
        st.markdown("*Conclusion*")
        st.markdown("Gospel Music: Most commonly listened to by individuals experiencing insomnia and in older age groups, where it has been reported to help improve the condition.")
        
        st.markdown("Lofi Music: Predominantly favored by individuals dealing with OCD, Anxiety, and Depression, typically in their mid-20s. It has shown to improve these conditions among participants.")

        st.markdown("Video Game Music: Shows a detrimental effect on all mental health conditions and should therefore be avoided. Listeners are usually in their early 20s, suggesting potential impacts on social functioning.")

        st.markdown("R&B, Jazz, K-pop, Country, EDM, Hip hop, Folk, Metal, and Latin Music: Generally either have a positive impact in improving mental health conditions or have no discernible negative effects reported.")
        
        st.markdown("Rock Music: Should be avoided by individuals dealing with Insomnia and Depression due to its potential to worsen these conditions more than improve them.")

        st.markdown("Classical Music: Should be avoided by individuals dealing with OCD, Anxiety, and Depression as it has a higher likelihood of worsening these conditions compared to improving them. However, it shows potential for improving Insomnia.")

        st.markdown("Listening Habits: Listening to Lofi and Gospel music for 4-6 hours has been associated with improvements in the mentioned conditions. Conversely, even 2 hours of exposure to Video Game music and pop has shown worsening effects in some participants.")
        
        st.markdown("""
                Anxiety Levels:

                All listeners have some degree of anxiety, but those who favor Rock, Jazz, K-pop, Hip hop, Pop, and Folk music are particularly affected, with anxiety levels exceeding 6. This suggests that these genres might be associated with higher stress or anxiety, possibly due to their energetic or emotionally charged nature.

                """)

        st.markdown("""
            Insomnia Levels:

            Generally, listeners experience low levels of insomnia (below 4). However, those who listen to Metal, Lofi, and Gospel music experience higher levels of insomnia. This could be due to the intense or calming effects of these genres, which might impact sleep patterns differently.




            """)
        
        st.markdown("""
                    OCD Levels:

                    OCD tendencies are more pronounced in listeners of Rap and Lofi music, with maximum levels above 3. This indicates a potential correlation between these genres and obsessive-compulsive behaviors or traits.
                    """)
        

        st.markdown("""
            Depression Levels:

            High levels of depression (above 5) are noted among listeners of Lofi, Hip hop, and Rock music. These genres might resonate more with individuals experiencing depressive symptoms or might influence mood negatively.




            """)
        
        st.markdown("""

            Mental Health Improvements and Deteriorations:

            While music generally provides therapeutic benefits, listeners of Rock, Video Game Music, Pop, Rap, and Classical music often experience worsening conditions. This suggests that these genres might exacerbate mental health issues for some individuals.

            """)
        
        st.html("""
            OCD and Music: Listeners of Rock, Video Game Music, Pop, Rap, and Classical music show varying levels of OCD, with the highest levels found among Classical music listeners. This might be due to the structured and repetitive nature of Classical music, which could influence obsessive behaviors.
            <br/>
            Anxiety and Music: Anxiety is present in listeners of Rock, Video Game Music, Pop, and Classical music, with Video Game Music listeners exhibiting the highest levels. The immersive and often intense nature of Video Game Music might contribute to increased anxiety.
             <br/>
            Insomnia and Music: Insomnia is notably present among listeners of Rock, Video Game Music, Pop, and Classical music. The stimulating nature of these genres, particularly Video Game Music, might disrupt sleep patterns.
            <br/>
            Depression and Music: High levels of depression are found among listeners of Rock, Video Game Music, Pop, and Classical music. These genres might either attract listeners who are already depressed or contribute to depressive moods.
            <br/>
            These insights underscore the importance of considering musical preferences and their potential impact on mental health conditions, highlighting certain genres that may be beneficial or detrimental depending on individual circumstances.
            It's evident that various types of music can profoundly influence an individual's mental state. Therefore, considering one's current mood, overall disposition, and the duration of music exposure, recommendation systems in music applications can effectively suggest genres that potentially alleviate or improve their mental condition.
                <br/>
            Music has a unique ability to evoke emotions and alter psychological states. For instance, genres like Gospel and Lofi have shown positive impacts on listeners dealing with specific conditions such as insomnia, OCD, anxiety, and depression. These findings suggest that tailored music recommendations can play a crucial role in enhancing mental well-being.""")








page_names_to_funcs = {
    "Welcome":welcome,
    "Climate Change Analysis": climate,
    "Mental Health & Music Relationship": music_mental_health,
}





demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()