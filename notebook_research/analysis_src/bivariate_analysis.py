from abc import ABC, abstractmethod
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from us_visa.exception import CustomException
from us_visa.logger import logging
import sys

# Abstract Base Class for Bivariate Analysis Strategy
# ----------------------------------------------------
# This class defines a common interface for bivariate analysis strategies.
# Subclasses must implement the analyze method.
class BivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature_x: str, feature_y: str, plot_type: str, hue: str = None):
        """
        Perform bivariate analysis on a specific feature pair of the dataframe using a specified plot type.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature_x (str): The name of the first feature (x-axis).
        feature_y (str): The name of the second feature (y-axis).
        plot_type (str): The name of the plot.
        hue (str, optional): The feature to use for color coding (if applicable).

        Returns:
        None: This method visualizes the relationship between two features.
        """
        pass

# Concrete Strategy for Numerical vs Numerical Analysis
# ------------------------------------------------------
# This strategy analyzes the relationship between two numerical features using plot given by user
class NumericalVsNumericalBivariateAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature_x: str, feature_y: str, plot_type: str, hue: str = None):
        """
        Plots the specified type of plot for two numerical features.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature_x (str): The name of the first numerical feature/column.
        feature_y (str): The name of the second numerical feature/column.
        plot_type (str): The type of plot to use for analysis.
        hue (str, optional): The feature to use for color coding (if applicable).

        Returns:
        None: Displays a specific plot.
        """

        plt.figure(figsize=(6, 4), dpi=200)

        plot_functions = {
            'scatter': self.scatter,
            'line': self.line,
            'hexbin': self.hexbin,
            'correlation_heatmap': self.correlation_heatmap
        }
        plot_func = plot_functions.get(plot_type)
        if plot_func:
            plot_func(df, feature_x, feature_y, hue)
        else:
            raise CustomException(f"Unsupported plot type '{plot_type}' for numerical vs. numerical bivariate analysis.", sys)
        

    def scatter(self, df, feature_x, feature_y, hue=None):
        sns.scatterplot(data=df, x=feature_x, y=feature_y, hue=hue)
        plt.title(f"Scatter Plot of {feature_x} vs {feature_y}")
        plt.tight_layout()
        plt.show()

    def line(self, df, feature_x, feature_y, hue=None):
        sns.lineplot(data=df, x=feature_x, y=feature_y, hue=hue)
        plt.title(f"Line Plot of {feature_x} vs {feature_y}")
        plt.tight_layout()
        plt.show()

    def hexbin(self, df, feature_x, feature_y, hue=None):
        plt.hexbin(df[feature_x], df[feature_y], gridsize=50, cmap='Blues')
        plt.colorbar(label='Count')
        plt.title(f"Hexbin Plot of {feature_x} vs {feature_y}")
        plt.xlabel(feature_x)
        plt.ylabel(feature_y)
        plt.tight_layout()
        plt.show()

    def correlation_heatmap(self, df, feature_x, feature_y, hue=None):
        corr = df[[feature_x, feature_y]].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        plt.title(f"Correlation Heatmap between {feature_x} and {feature_y}")
        plt.tight_layout()
        plt.show()

# Concrete Strategy for Numerical vs Categorical Analysis
# --------------------------------------------------------
# This strategy analyzes the relationship between a numerical feature and a categorical feature.
class NumericalVsCategoricalBivariateAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature_x: str, feature_y: str, plot_type: str, hue: str = None):
        """
        Plots the specified type of plot for a numerical vs categorical feature.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature_x (str): The numerical feature for analysis.
        feature_y (str): The categorical feature for analysis.
        plot_type (str): The type of plot to use for analysis.
        hue (str, optional): The feature to use for color coding (if applicable).

        Returns:
        None: Displays a specific plot.
        """

        plt.figure(figsize=(6, 4), dpi=200)

        plot_functions = {
            'box': self.box,
            'violin': self.violin,
            'swarm': self.swarm,
            'strip': self.strip,
            'boxplot': self.boxplot
        }
        
        plot_func = plot_functions.get(plot_type)
        if plot_func:
            plot_func(df, feature_x, feature_y, hue)
        else:
            raise CustomException(f"Unsupported plot type '{plot_type}' for numerical vs. categorical bivariate analysis.", sys)
        
    def box(self, df, feature_x, feature_y, hue=None):
        sns.boxplot(x=feature_y, y=feature_x, data=df, hue=hue)
        plt.title(f"Box Plot of {feature_x} grouped by {feature_y}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def boxplot(self, df, feature_x, feature_y, hue=None):
        clr1 = ['#1E90FF', '#DC143C']
        sns.boxplot(data=df, x=df[feature_x], y=df[feature_y], palette=clr1)
        plt.title(f"Box Plot of {feature_x} grouped by {feature_y}")
        plt.xlabel(feature_x)
        plt.tight_layout()
        plt.show()

    def violin(self, df, feature_x, feature_y, hue=None):
        sns.violinplot(x=feature_y, y=feature_x, data=df, hue=hue)
        plt.title(f"Violin Plot of {feature_x} grouped by {feature_y}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def swarm(self, df, feature_x, feature_y, hue=None):
        sns.swarmplot(x=feature_y, y=feature_x, data=df, hue=hue)
        plt.title(f"Swarm Plot of {feature_x} grouped by {feature_y}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def strip(self, df, feature_x, feature_y, hue=None):
        sns.stripplot(x=feature_y, y=feature_x, data=df, hue=hue)
        plt.title(f"Strip Plot of {feature_x} grouped by {feature_y}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Concrete Strategy for Categorical vs Categorical Analysis
# --------------------------------------------------------
# This strategy analyzes the relationship between a categorical feature and a categorical feature.
class CategoricalVsCategoricalBivariateAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature_x: str, feature_y: str, plot_type: str, hue: str = None):
        """
        Plots the specified type of plot for two categorical features.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature_x (str): The name of the first categorical feature.
        feature_y (str): The name of the second categorical feature.
        plot_type (str): The type of plot to use for analysis.
        hue (str, optional): The feature to use for color coding (if applicable).

        Returns:
        None: Displays a specific plot.
        """

        plt.figure(figsize=(6, 4), dpi=200)
        
        plot_functions = {
            'stacked_bar': self.stacked_bar,
            'grouped_bar': self.grouped_bar
        }

        plot_func = plot_functions.get(plot_type)
        if plot_func:
            plot_func(df, feature_x, feature_y, hue)
        else:
            raise CustomException(f"Unsupported plot type '{plot_type}' for categorical vs. categorical analysis.", sys)

    def stacked_bar(self, df, feature_x, feature_y, hue=None):
        cross_tab = pd.crosstab(df[feature_x], df[feature_y], normalize='index')
        cross_tab.plot(kind='bar', stacked=True, colormap='viridis')
        plt.title(f"Stacked Bar Plot of {feature_x} vs {feature_y}")
        plt.xlabel(feature_x)
        plt.ylabel('Proportion')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def grouped_bar(self, df, feature_x, feature_y, hue=None):
        cross_tab = pd.crosstab(df[feature_x], df[feature_y])
         # Create the bar plot
        ax = cross_tab.plot(kind='bar', colormap='viridis')
        
        # Add labels on top of each bar
        for p in ax.patches:  # Iterate over each bar
            height = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2, height, f'{int(height)}', 
                    ha='center', va='bottom')  # Position the label
        
        # Set the plot title and labels
        plt.title(f"Grouped Bar Plot of {feature_x} vs {feature_y}")
        plt.xlabel(feature_x)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


# Context Class that uses a BivariateAnalysisStrategy
class BivariateAnalyzer:
    def __init__(self):
        self._strategy = None

    def execute_analysis(self, df: pd.DataFrame, feature_x: str, feature_y: str, plot_type: str, hue: str = None):
        if pd.api.types.is_numeric_dtype(df[feature_x]) and pd.api.types.is_numeric_dtype(df[feature_y]):
            self._strategy = NumericalVsNumericalBivariateAnalysis()
        elif pd.api.types.is_numeric_dtype(df[feature_x]) and df[feature_y].dtype == 'object':
            self._strategy = NumericalVsCategoricalBivariateAnalysis()
        elif (df[feature_x].dtype == 'object' and df[feature_y].dtype == 'object'):
            self._strategy = CategoricalVsCategoricalBivariateAnalysis()
        else:
            raise CustomException("Unsupported combination of feature types for bivariate analysis.", sys)

        self._strategy.analyze(df, feature_x, feature_y, plot_type, hue)


# Example usage
# if __name__ == "__main__":
#     df = pd.read_csv("/Users/aadarsh/Desktop/Data Scientist/Projects/US-Visa-Approval-Prediction/us_visa/data/extracted_data/EasyVisa.csv")

#     # Initialize BivariateAnalyzer
#     bivariate_analyzer = BivariateAnalyzer()

    # Example: Numerical vs Numerical
    # bivariate_analyzer.execute_analysis(df, feature_x='prevailing_wage', feature_y='yr_of_estab', plot_type='scatter', hue='case_status')

    # Example: Numerical vs Categorical
    # bivariate_analyzer.execute_analysis(df, feature_x='prevailing_wage', feature_y='full_time_position', plot_type='violin', hue='case_status')

    # Example: Categorical vs Categorical
    # bivariate_analyzer.execute_analysis(df, feature_x='full_time_position', feature_y='case_status', plot_type='grouped_bar')




