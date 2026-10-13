"""Exercise 3 - Preparing the Spaceship Titanic dataset for a tanh-based network."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[4]
FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = ROOT / "data" / "SpaceshipTitanic" / "train.csv"

SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
NUM_COLS = ["Age"] + SPEND_COLS + ["TotalSpend"]
CAT_COLS = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
DROP_COLS = ["PassengerId", "Name", "Cabin"]

rng = np.random.default_rng(42)

# ---------------------------------------------------------------- A: describe
df = pd.read_csv(DATA_PATH)

target_balance = df["Transported"].value_counts(normalize=True)
print("A. Target balance (Transported):")
print(target_balance.to_string())

missing = df.isna().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_table = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
missing_table = missing_table[missing_table["missing_count"] > 0].sort_values(
    "missing_count", ascending=False
)
print("\nA. Missing values per column:")
print(missing_table.to_string())

spend_stats = df[SPEND_COLS].agg(["mean", "median", "max"]).T
print("\nA. Spending column statistics:")
print(spend_stats.to_string())

numerical_features = ["Age"] + SPEND_COLS
categorical_features = ["HomePlanet", "CryoSleep", "Cabin", "Destination", "VIP"]
print(f"\nA. Numerical features: {numerical_features}")
print(f"A. Categorical features: {categorical_features}")

# ---------------------------------------------------------------- B: split
train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df["Transported"], random_state=42
)
print(f"\nB. Train shape: {train_df.shape}, Test shape: {test_df.shape}")
print(f"B. Train target balance:\n{train_df['Transported'].value_counts(normalize=True).to_string()}")
print(f"B. Test target balance:\n{test_df['Transported'].value_counts(normalize=True).to_string()}")


def engineer(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["TotalSpend"] = frame[SPEND_COLS].sum(axis=1, skipna=True)
    frame = frame.drop(columns=DROP_COLS)
    return frame


train_fe = engineer(train_df)
test_fe = engineer(test_df)

y_train = train_fe.pop("Transported").astype(int)
y_test = test_fe.pop("Transported").astype(int)

# ---------------------------------------------------------------- C: preprocess
# log1p on spending columns before scaling (right-skewed, many zeros)
log_cols = SPEND_COLS + ["TotalSpend"]
foodcourt_before = train_fe["FoodCourt"].dropna().copy()

for frame in (train_fe, test_fe):
    for col in log_cols:
        frame[col] = np.log1p(frame[col].clip(lower=0))

foodcourt_after = train_fe["FoodCourt"].dropna().copy()

num_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)
cat_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)
preprocessor = ColumnTransformer(
    [
        ("num", num_pipeline, NUM_COLS),
        ("cat", cat_pipeline, CAT_COLS),
    ]
)

X_train = preprocessor.fit_transform(train_fe)
X_test = preprocessor.transform(test_fe)

print(f"\nC. Final training feature matrix shape: {X_train.shape}")
print(f"C. Final test feature matrix shape: {X_test.shape}")

num_slice_train = X_train[:, : len(NUM_COLS)]
num_slice_test = X_test[:, : len(NUM_COLS)]
print(f"C. Scaled numeric range (train): min={num_slice_train.min():.4f}, max={num_slice_train.max():.4f}")
print(f"C. Scaled numeric range (test):  min={num_slice_test.min():.4f}, max={num_slice_test.max():.4f}")

# ---------------------------------------------------------------- D: checks + Figure 6
nan_train = np.isnan(X_train).sum() if not hasattr(X_train, "toarray") else np.isnan(X_train.toarray()).sum()
print(f"\nD. Remaining NaNs in X_train: {nan_train}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(foodcourt_before, bins=40, color="skyblue")
axes[0].set_title("FoodCourt before log1p transform")
axes[0].set_xlabel("FoodCourt ($)")
axes[0].set_ylabel("Count")

axes[1].hist(foodcourt_after, bins=40, color="salmon")
axes[1].set_title("FoodCourt after log1p transform")
axes[1].set_xlabel("log1p(FoodCourt)")
axes[1].set_ylabel("Count")

fig.suptitle("Figure 6 - FoodCourt distribution before/after log transform")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig6_foodcourt_before_after.png", dpi=150)
plt.close(fig)

if __name__ == "__main__":
    pass
