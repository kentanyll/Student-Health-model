import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

def show_count_plots(df:pd.DataFrame, col:str, draw_pie_chart:bool = True, save_path:str = None):
    counts = df[col].value_counts()

    fig, ax = plt.subplots(1,2,figsize=(14,6))

    ax[0].barh(counts.index, counts.values, color='steelblue')
    ax[0].set_xlabel('Count')
    ax[0].set_title('Bar Chart Class Distribution')

    if draw_pie_chart:
        ax[1].pie(counts.values, labels=counts.index, autopct='%1.1f%%')
        ax[1].set_title('Pie Chart Class Distribution')
    
    if save_path:
        plt.savefig(save_path, dpi=400)
    
    plt.tight_layout()
    plt.show()

def show_corr(df:pd.DataFrame, save_path:str=None, h=15, w=12):
    """Plot a correlation heatmap for all numeric columns in a DataFrame."""
    plt.figure(figsize=(h,w))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)

    plt.title('Correlation Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=400)

    plt.show()

def show_outliers(df:pd.DataFrame, save_path:str=None):
    """Plot boxplots for all columns in a DataFrame to spot outliers visually."""

    plt.figure(figsize=(14,6))
    sns.boxplot(df)
    plt.xticks(rotation=45)
    plt.grid()
    plt.title('Box Plot')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=400)
    
    plt.show()

def remove_outliers_iqr(
    df: pd.DataFrame,
    columns: list = None,
    factor: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers using the IQR method.

    Args:
        df: input DataFrame
        columns: numerical columns to process (defaults to the u/g/r/i/z magnitude bands)
        factor: IQR multiplier controlling how aggressive the filtering is

    Returns:
        DataFrame with outlier rows removed
    """

    df = df.copy()

    if columns is None:
        columns = ['Sleep_Hours', 'Study_Hours', 'Social_Media_Hours']

    mask = pd.Series(True, index=df.index)

    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - factor * iqr
        upper = q3 + factor * iqr

        mask &= df[col].between(lower, upper)

    return df.loc[mask].reset_index(drop=True)

def show_confusion_matrix(y_true, y_pred, target_names=None, save_path:str=None):
    """Plot a confusion matrix heatmap given true and predicted labels"""

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=None if target_names is None else target_names, 
                yticklabels=None if target_names is None else target_names)
    
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=400)
    plt.show()

def plot_roc_curve(y_true, y_probas, target_names):
    """"""

    n_classes = len(target_names)

    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    if n_classes == 2 and (y_bin.ndim == 1 or y_bin.shape[1] == 1):
        y_bin = np.hstack([1 - y_bin, y_bin])

    plt.figure(figsize=(10,6))

    colors=['blue', 'orange', 'green']

    for i, (name, color) in enumerate(zip(target_names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:,i], y_probas[:,i])
        roc_auc = auc(fpr,tpr)
        plt.plot(fpr,tpr,color=color,
                 label=f'{name} (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.show()