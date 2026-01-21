import pandas as pd

def load_ISOT_dataset(ISOT_fake_file_path, ISOT_true_file_path):
    ISOT_fake_df = pd.read_csv(ISOT_fake_file_path, engine='c', on_bad_lines='skip')
    ISOT_fake_df['label'] = 0
    
    ISOT_true_df = pd.read_csv(ISOT_true_file_path, engine='c', on_bad_lines='skip')
    ISOT_true_df['label'] = 1
    
    ISOT_df = pd.concat([ISOT_fake_df, ISOT_true_df])
    print(f"ISOT dataset: {len(ISOT_df)} rows")
    
    ISOT_df.drop('subject', axis=1, inplace=True)
    ISOT_df.drop('date', axis=1, inplace=True)
    ISOT_df = ISOT_df.sample(frac=1, random_state=16) #shuffling 100% of the rows
    return ISOT_df

def load_FakeNewsNet_dataset(gossipcop_fake_file_path, gossipcop_real_file_path, politifact_fake_file_path, politifact_real_file_path):
    gossipcop_fake_df = pd.read_csv(gossipcop_fake_file_path, engine='c', on_bad_lines='skip')
    gossipcop_fake_df['label'] = 0
    politifact_fake_df = pd.read_csv(politifact_fake_file_path, engine='c', on_bad_lines='skip')
    politifact_fake_df['label'] = 0
    gossipcop_true_df = pd.read_csv(gossipcop_real_file_path, engine='c', on_bad_lines='skip')
    gossipcop_true_df['label'] = 1
    politifact_true_df = pd.read_csv(politifact_real_file_path, engine='c', on_bad_lines='skip')
    politifact_true_df['label'] = 1
    
    FakeNewsNet_df = pd.concat([gossipcop_fake_df, politifact_fake_df, gossipcop_true_df, politifact_true_df])
    print(f"FakeNewsNet dataset: {len(FakeNewsNet_df)} rows")
    FakeNewsNet_df = FakeNewsNet_df.sample(frac=1, random_state=16)
    FakeNewsNet_df = FakeNewsNet_df[['title', 'label']]
    return FakeNewsNet_df

def load_Figshare_dataset(figshare_file_path):
    figshare_df = pd.read_csv(figshare_file_path, engine='c', on_bad_lines='skip')
    figshare_df = figshare_df[['text', 'target']]
    figshare_df.rename(columns={"text": "body", "target": "label"}, inplace=True)
    figshare_df["label"] = figshare_df["label"].str.lower().map({"true": 1, "fake": 0})
    print(f"FigShare Dataset: {len(figshare_df)} rows")
    return figshare_df
