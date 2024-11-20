from abc import ABC, abstractmethod
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from us_visa.exception import CustomException
from us_visa.logger import logging
import sys

# Abstract Base Class for Independent Vs Dependent Variable Analysis
# -----------------------------------------------
# This class defines a template for Independent Vs Dependent Variable.
# Subclasses must implement the methods to identify and visualize missing values.
class IndependentVsDependentVariableAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame, independent_feature:str = None, target_feature: str = None):
        """
        Performs a complete missing values analysis by identifying and visualizing missing values.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.
        independent_feature: Feature to plot frequency.
        target_feature: Feature to be catgory of analysis

        Returns:
        None: This method performs the analysis and visualizes missing values.
        """
        self.visualize_frequency_category(df, independent_feature, target_feature)
        self.visualize_proportion(df, independent_feature, target_feature)
    
    @abstractmethod
    def visualize_frequency_category(self, df: pd.DataFrame, independent_feature:str, target_feature: str):
        """
        Visualize the frequency of independent variable agains target variable values in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.

        Returns:
        None: This method should print the count of independent values vs targte column.
        """
        pass

    @abstractmethod
    def visualize_proportion(self, df: pd.DataFrame, independent_feature:str, target_feature: str):
        """
        Visualize the proportion of independent variable agains target variable values in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.

        Returns:
        None: This method should print the proportion of independent values vs targte column.
        """
        pass

# Concrete Class for Independent Vs Dependent Variable Analysis
# -------------------------------------------------
# This class implements methods to visualize frequency of categorical feature that is targte variable and visualize the proportion of target variable against independent variable in dataframe.
class CategoricalVsCategoricalAnalysis(IndependentVsDependentVariableAnalysisTemplate):
    def visualize_frequency_category(self, df: pd.DataFrame, independent_feature:str, target_feature: str):
        
        plt.figure(figsize=(6, 4), dpi=200)
        p = sns.countplot(x=independent_feature, data=df, hue= target_feature, palette="muted")
        
        # Add labels to each bar
        for patch in p.patches:
            # Only annotate bars with non-zero height
            if patch.get_height() > 0: 
                height = patch.get_height()
                p.text(patch.get_x() + patch.get_width() / 2., height, f'{int(height)}', ha='center', va='bottom')

        plt.title(f"Count Plot of {independent_feature} Vs {target_feature}")
        plt.xlabel(independent_feature)
        plt.ylabel("Count")
        # plt.xticks(rotation=45)
        plt.legend(title="Visa Status", fancybox=True)
        plt.tight_layout()
        plt.show()

    def visualize_proportion(self, df: pd.DataFrame, independent_feature:str, target_feature: str):
        
        print(df.groupby(independent_feature)[target_feature].value_counts(normalize=True).to_frame()*100)
        group_df = (df.groupby(independent_feature)[target_feature].value_counts(normalize=True)
                    .mul(100)
                    .reset_index()
                    .rename(columns={'proportion': 'Percentage'})  # Customize column names
                    )

        # Plot the bar chart
        plt.figure(figsize=(10, 6))
        bar_plot = sns.barplot(
            data=group_df,
            x=independent_feature,
            y='Percentage',
            hue=target_feature,
            palette='Set2'
        )

        # Add labels on each bar
        for p in bar_plot.patches:
            if p.get_height() > 0:  # Only annotate bars with non-zero height
                percentage = f'{p.get_height():.1f}%'
                x = p.get_x() + p.get_width() / 2  # Center the label horizontally
                y = p.get_height()  # Place the label at the top of the bar
                bar_plot.annotate(percentage, (x, y), ha='center', va='bottom', fontsize=10, color='black')

        # Add titles and labels
        plt.title(f"Certified vs Denied Cases by {independent_feature} (%)", fontsize=16)
        plt.xlabel("Continent", fontsize=14)
        plt.ylabel("Percentage (%)", fontsize=14)
        plt.legend(title="Case Status")
        plt.xticks(rotation=45)

        # Show the plot
        plt.tight_layout()
        plt.show()
