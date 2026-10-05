import numpy as np
import pandas as pd

def add_int_features(df:pd.DataFrame) -> pd.DataFrame:
    """"""

    df = df.copy()

    df['Pressure_minus_Support'] = df['Exam_Pressure'] - df['Family_Support']
    df['Free_Time'] = 24 - df['Sleep_Hours'] - df['Study_Hours']

    return df

def add_float_features(df:pd.DataFrame) -> pd.DataFrame:
    """"""

    df = df.copy()

    df['Pressure_x_Study'] = df['Exam_Pressure'] * df['Study_Hours']
    df['Pressure_x_Sleep'] = df['Exam_Pressure'] * df['Sleep_Hours']
    df['Attendance_x_Study_Hours'] = df['Attendance'] * df['Study_Hours']
    df['Study_per_Sleep'] = df['Study_Hours'] / df['Sleep_Hours'].replace(0, np.nan)
    df['Social_to_Study_ratio'] = df['Social_Media_Hours'] / df['Study_Hours'].replace(0, np.nan)
    df['Free_Time_to_Social_ratio'] = df['Free_Time'] / df['Social_Media_Hours'].replace(0, np.nan)
    

    return df

def add_bool_features(df:pd.DataFrame) -> pd.DataFrame:
    """"""

    df = df.copy()

    df['Sleep_Deficit'] = (df['Sleep_Hours'] < 6).astype(int)
    df['Low_Attendance'] = (df['Attendance'] < 50).astype(int)
    df['Low_Study'] = (df['Study_Hours'] < df['Study_Hours'].mean()).astype(int)
    df['High_Social'] = (df['Social_Media_Hours'] > 4).astype(int)

    return df

def add_month_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Encode month as sin/cos"""

    df = df.copy()

    month_radians = (2 * np.pi * df["Month"]) / 12

    df["Month_sin"] = np.sin(month_radians)
    df["Month_cos"] = np.cos(month_radians)

    return df

def build_features(df:pd.DataFrame) -> pd.DataFrame:
    """Run the full report-aligned feature-engineering pipeline in order"""

    df = df.copy()

    df = add_int_features(df)
    df = add_float_features(df)
    df = add_bool_features(df)
    df = add_month_transform(df)

    return df
