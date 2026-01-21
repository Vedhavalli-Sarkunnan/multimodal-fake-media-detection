import pandas as pd

def preprocess_text(df, lowercase=True):
    text_cols = [col for col in df.columns if col in ["title", "body"]]
    
    df = df.dropna(subset=["label"]).drop_duplicates().reset_index(drop=True)
    df[text_cols] = df[text_cols].fillna("")
    df[text_cols] = df[text_cols].apply(
        lambda col: col.str.lower() if lowercase else col
    ).apply(lambda col: col.str.strip().str.replace(r'\s+', ' ', regex=True))
    print(df['label'].value_counts())

    return df
