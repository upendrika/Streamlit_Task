import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map


def clean_experience(x):
    if x ==  'More than 50 years':
        return 50
    if x == 'Less than 1 year':
        return 0.5
    return float(x)


def clean_education(x):
    if 'Bachelor’s degree' in x:
        return 'Bachelor’s degree'
    if 'Master’s degree' in x:
        return 'Master’s degree'
    if 'Professional degree' in x or 'Other doctoral' in x:
        return 'Post grad'
    return 'Less than a Bachelors'


@st.cache_data
def load_data():
    df = pd.read_csv("survey_results_public.csv")
    df = df[["Country", "EdLevel", "YearsCodePro", "Employment", "ConvertedComp"]]
    df = df[df["ConvertedComp"].notnull()]
    df = df.dropna()
    df = df[df["Employment"] == "Employed full-time"]
    df = df.drop("Employment", axis=1)

    country_map = shorten_categories(df.Country.value_counts(), 400)
    df["Country"] = df["Country"].map(country_map)
    df = df[df["ConvertedComp"] <= 250000]
    df = df[df["ConvertedComp"] >= 10000]
    df = df[df["Country"] != "Other"]

    df["YearsCodePro"] = df["YearsCodePro"].apply(clean_experience)
    df["EdLevel"] = df["EdLevel"].apply(clean_education)
    df = df.rename({"ConvertedComp": "Salary"}, axis=1)
    return df

df = load_data()

def show_explore_page():
    st.title("Explore Software Engineer Salaries")
    st.image('sof_salary.jpg', width=700)
   
    
    # Display the first 10 rows of the dataset
    st.write("### Dataset :")
    st.dataframe(df.head(10))
    data = df["Country"].value_counts()

    fig1, ax1 = plt.subplots()
    ax1.pie(data, labels=data.index, autopct="%1.1f%%", shadow=True, startangle=20, wedgeprops=dict(width=0.3))
    ax1.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Add a white circle at the center to create a donut shape
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    st.write("""#### Number of Data from different countries""")

    st.pyplot(fig1)

    
    st.write(
        """
    #### Mean Salary Based On Country
    """
    )

    data = df.groupby(["Country"])["Salary"].mean().sort_values(ascending=True)
    st.bar_chart(data)

    st.write(
        """
    #### Mean Salary Based On Experience
    """
    )

    data = df.groupby(["YearsCodePro"])["Salary"].mean().sort_values(ascending=True)
    st.line_chart(data)


    # Correlation heatmap
    st.write("#### Correlation Heatmap")
    # Select only numeric columns for correlation matrix
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr = numeric_df.corr()
    plt.figure(figsize=(10, 6))
    sns.set_style('whitegrid')
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    st.pyplot(plt)

    # Scatter plot of Years of Experience vs Salary
    st.write("#### Scatter Plot of Years of Experience vs Salary")
    plt.figure(figsize=(10, 10))
    plt.title('Scatter Plot')
    sns.set_style("darkgrid")
    sns.scatterplot(x='YearsCodePro', y='Salary', data=df)
    st.pyplot(plt)

    # Boxplot of Salary by Education Level
    st.write("#### Boxplot of Salary by Education Level")
    plt.figure(figsize=(12, 8))
    sns.set_style("darkgrid")
    sns.boxplot(x='EdLevel', y='Salary', data=df)
    plt.xticks(rotation=45)
    st.pyplot(plt)


