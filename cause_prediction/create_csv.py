import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("backup_data/wildfire_full.csv")

columns = [
    "FOD_ID",
    "FIRE_YEAR",
    "DISCOVERY_DOY",
    "LATITUDE",
    "LONGITUDE",
    "STATE",
    "NWCG_CAUSE_CLASSIFICATION"
]

df = df[df["NWCG_CAUSE_CLASSIFICATION"].isin(["Human", "Natural"])]
df = df[columns].dropna()
df = df.drop_duplicates(subset="FOD_ID")

train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["NWCG_CAUSE_CLASSIFICATION"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df["NWCG_CAUSE_CLASSIFICATION"]
)

train_df.to_csv("cause_prediction/wildfires_cause_train.csv", index=False)
val_df.to_csv("cause_prediction/wildfires_cause_val.csv", index=False)
test_df.to_csv("cause_prediction/wildfires_cause_test.csv", index=False)

print("Training:", train_df.shape)
print("Validation:", val_df.shape)
print("Testing:", test_df.shape)