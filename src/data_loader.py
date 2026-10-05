import pandas as pd

def load_data(df_path, include_class=True) -> pd.DataFrame:
    """
    Load a .csv and apply dataset-spicific columnt cleanup.
    
    Args:
        df_path: path to the .csv file
        include_class: if True, label encode the 'class' column
    
    Returns:
        Cleaned Dataframe
    """

    df = pd.read_csv(df_path)

    cols = ['Student_Type']
    dummy_data = pd.get_dummies(df[cols], dtype=int)
    df.drop(cols, inplace=True, axis=1)
    df = pd.concat([df, dummy_data], axis=1)

    if include_class:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df['Stress_Level'] = le.fit_transform(df['Stress_Level'])
        print(f'Label inverse is {le.classes_}')
    
    return df