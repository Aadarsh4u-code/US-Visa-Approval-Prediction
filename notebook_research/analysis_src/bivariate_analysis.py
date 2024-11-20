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
            'histogram': self.histogram,
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
    
    def histogram(self, df, feature_x, feature_y, hue=None):
        """
        Plot histograms for numerical vs numerical features.

        Parameters:
        df (pd.DataFrame): Dataframe containing the data.
        feature_x (str): First numerical feature.
        feature_y (str): Second numerical feature.
        hue (str, optional): Categorical feature to group the data.
        """
        if hue:
            sns.histplot(data=df, x=feature_x, hue=hue, kde=True, bins=50, multiple='stack')
        else:
            sns.histplot(data=df, x=feature_x, kde=True, bins=50, color='blue')
        plt.title(f"Histogram of {feature_x}" + (f" with hue {hue}" if hue else ""))
        plt.xlabel(feature_x)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

    def scatter(self, df, feature_x, feature_y, hue=None):
        sns.scatterplot(data=df, x=feature_x, y=feature_y, hue=hue)
        plt.title(f"Scatter Plot of {feature_x} Vs {feature_y}")
        plt.tight_layout()
        plt.show()

    def line(self, df, feature_x, feature_y, hue=None):
        sns.lineplot(data=df, x=feature_x, y=feature_y, hue=hue)
        plt.title(f"Line Plot of {feature_x} Vs {feature_y}")
        plt.tight_layout()
        plt.show()

    def hexbin(self, df, feature_x, feature_y, hue=None):
        plt.hexbin(df[feature_x], df[feature_y], gridsize=50, cmap='Blues')
        plt.colorbar(label='Count')
        plt.title(f"Hexbin Plot of {feature_x} Vs {feature_y}")
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
            'boxplot': self.boxplot,
            'violin': self.violin,
            'barplot': self.barplot,
            'lineplot': self.lineplot
        }
        
        plot_func = plot_functions.get(plot_type)
        if plot_func:
            plot_func(df, feature_x, feature_y, hue)
        else:
            raise CustomException(f"Unsupported plot type '{plot_type}' for numerical vs. categorical bivariate analysis.", sys)
        
    def boxplot(self, df, feature_x, feature_y, hue=None):
        sns.boxplot(x=feature_x, y=feature_y, data=df, hue=hue)
        plt.title(f"Box Plot of {feature_x} Vs {feature_y}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def violin(self, df, feature_x, feature_y, hue=None):
        sns.violinplot(x=feature_y, y=feature_x, data=df, hue=hue)
        plt.title(f"Violin Plot of {feature_x} Vs {feature_y}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def barplot(self, df, feature_x, feature_y, hue=None):
        plt.figure(figsize=(6, 6), dpi=200)
        p = sns.barplot(data=df, x=feature_y, y=feature_x, hue=hue)
        # Add labels to each bar
        for patch in p.patches:
            height = patch.get_height()
            p.text(patch.get_x() + patch.get_width() / 2., height, f'{int(height)}', ha='center', va='bottom')
        plt.title(f"Bar Plot of {feature_x} Vs {feature_y}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def lineplot(self, df, feature_x, feature_y, hue=None):
        sns.lineplot(data=df, x=feature_y, y=feature_x, hue=hue, marker="o")
        plt.title(f"Line Plot of {feature_x} Vs {feature_y}")
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

        plt.figure(figsize=(8, 6), dpi=200)
        
        plot_functions = {
            'stacked_bar': self.stacked_bar,
            'grouped_bar': self.grouped_bar
        }

        plot_func = plot_functions.get(plot_type)
        if plot_func:
            plot_func(df, feature_x, feature_y, hue)
        else:
            raise CustomException(f"Unsupported plot type '{plot_type}' for categorical vs. categorical analysis.", sys)

    # Compares two categorical features.
    def stacked_bar(self, df, feature_x, feature_y, hue=None):
        cross_tab = pd.crosstab(df[feature_x], df[feature_y], normalize='index')
        cross_tab.plot(kind='bar', stacked=True, colormap='viridis')
        plt.title(f"Stacked Bar Plot of {feature_x} Vs {feature_y}")
        plt.xlabel(feature_x)
        plt.ylabel('Proportion')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # Shows grouped frequency distributions.
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
        plt.title(f"Grouped Bar Plot of {feature_x} Vs {feature_y}")
        plt.xlabel(feature_x)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Concrete Strategy for Multiple Feature  Bivariant Analysis Strategy
# ----------------------------------------------
# This class allows to display multiple feature into a single frame i.e. subplots.
class MultiPlotBivariateAnalysis(BivariateAnalysisStrategy):
    def __init__(self, features: list, n_cols:int = 2, plot_type: str = None):
        """
        Initializes with a list of features and the number of columns for subplots.

        Parameters:
        features (list): List of feature names to analyze.
        n_cols (int): Number of columns for the subplot layout.
        plot_type (str): Type of plot to use for each feature.
        """
        self.features = features
        self.n_cols = n_cols
        self.plot_type = plot_type

    # def analyze(self, df: pd.DataFrame, feature: str = None, plot_type: str = None):
    def analyze(self, df: pd.DataFrame, feature_x: str, feature_y: str, plot_type: str, hue: str = None):
        """
        Plots multiple features in a grid layout.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        """
        n_rows = (len(self.features) + self.n_cols - 1) // self.n_cols  # Calculate required rows
        fig, axes = plt.subplots(n_rows, self.n_cols, figsize=(self.n_cols * 6, n_rows * 5), dpi=200)
        axes = axes.flatten()

        # Dictionary to handle different plot types
        plot_functions = {
            'histogram': self.histogram,
            'boxplot': self.boxplot,
            'kde': self.kde,
            # 'density': self.density,
            # 'violin': self.violin,
            # 'bar': self.bar,
            # 'pie': self.pie,
            # 'count': self.count
        }

        for i, feature in enumerate(self.features):
            ax = axes[i]
            # The := operator is a concise way to assign and check a variable at the same time.
            if plot_func := plot_functions.get(self.plot_type):
                plot_func(df, feature, ax, hue, target_feature = feature_y)  # Pass the axis to each plotting function
                ax.set_title(f"{self.plot_type.title()} of {feature} vs {feature_y}")
                ax.set_xlabel(feature)
                ax.set_ylabel("Density" if pd.api.types.is_numeric_dtype(df[feature]) else "Frequency")
 
            else:
                logging.warning(f"Unsupported plot type '{self.plot_type}' for feature '{feature}'.")

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.tight_layout()
        plt.show()

    def histogram(self, df, feature, ax, hue, target_feature):
        if hue:
            sns.histplot(data=df, x=feature, ax=ax, hue=hue, kde=True, bins=50, multiple='stack')
        else:
            sns.histplot(data=df, x=feature, ax=ax, kde=True, bins=50, color='blue')

    def kde(self, df, feature, ax, hue, target_feature):
        if hue:
            sns.kdeplot(data=df, x=feature, ax=ax, hue=hue, shade=True)
        else:
            sns.kdeplot(data=df, x=feature, ax=ax, shade=True, color='blue')


    def boxplot(self, df, feature, ax, hue, target_feature):
        sns.boxplot(y=feature, x=target_feature, data=df, ax=ax, hue=hue)
        

# Context Class that uses a BivariateAnalysisStrategy
class BivariateAnalyzer:
    def __init__(self, strategy: BivariateAnalysisStrategy = None):
        self._strategy = strategy

    def set_strategy(self, strategy: BivariateAnalysisStrategy):
        """
        Set or change the strategy dynamically.
        
        Parameters:
        strategy (BivariateAnalysisStrategy): The strategy to set.
        """
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature_x: str, feature_y: str, plot_type: str, hue: str = None):
        if self._strategy is None:
            raise CustomException("Strategy is not set. Use `set_strategy` to set a strategy before executing inspection.", sys)
        return self._strategy.analyze(df, feature_x, feature_y, plot_type, hue)
    

# Example usage
# if __name__ == "__main__":
#     df = pd.read_csv("/Users/aadarsh/Desktop/Data Scientist/Projects/US-Visa-Approval-Prediction/us_visa/data/extracted_data/EasyVisa.csv")

#     bivariate_analyzer = BivariateAnalyzer()
    # Example: Numerical vs Numerical
    # bivariate_analyzer.set_strategy(NumericalVsNumericalBivariateAnalysis())
    # bivariate_analyzer.execute_analysis(df, feature_x='yr_of_estab', feature_y='prevailing_wage', plot_type='histogram', hue='case_status')

    # Example: Numerical vs Categorical
    # bivariate_analyzer.set_strategy(NumericalVsCategoricalBivariateAnalysis())
    # bivariate_analyzer.execute_analysis(df, feature_x='prevailing_wage', feature_y='continent', plot_type='boxplot', hue='case_status')

    # Example: Categorical vs Categorical
    # bivariate_analyzer.set_strategy(CategoricalVsCategoricalBivariateAnalysis())
    # bivariate_analyzer.execute_analysis(df, feature_x='region_of_employment', feature_y='unit_of_wage', plot_type='stacked_bar', hue='case_status')
    # bivariate_analyzer.execute_analysis(df, feature_x='region_of_employment', feature_y='unit_of_wage', plot_type='grouped_bar', hue='case_status')

    # Example: Multiplot Numerical vs Numerical or Numerical vs Categorical
    # bivariate_analyzer.set_strategy(MultiPlotBivariateAnalysis(features=['no_of_employees', 'yr_of_estab', 'prevailing_wage'], plot_type='boxplot'))
    # bivariate_analyzer.execute_analysis(df, feature_x='', feature_y='case_status', plot_type='boxplot', hue='case_status')