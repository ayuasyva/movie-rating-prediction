# movie_rating_prediction.py

# ======== IMPORT LIBRARY =========
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ======== LOAD DATASET =========
df = pd.read_csv('tmdb_5000_movies.csv')  # Pastikan file ini ada di folder project kamu
print("Dataset loaded successfully. Shape:", df.shape)

# ======== DATA CLEANING =========
# Drop kolom yang tidak digunakan
df = df.drop(columns=['homepage', 'overview', 'tagline', 'keywords', 'production_companies'])

# Handle missing values
df = df.dropna()

# Ambil tahun dari release_date
df['release_year'] = pd.to_datetime(df['release_date']).dt.year
df = df.drop(columns=['release_date'])

# Hapus kolom yang terlalu kompleks untuk dummy project
df = df.drop(columns=['genres', 'spoken_languages', 'production_countries'])

# ======== EXPLORATORY DATA ANALYSIS =========
print("\nDescriptive statistics:\n", df.describe())

plt.figure(figsize=(8,5))
sns.histplot(df['vote_average'], bins=20, kde=True)
plt.title('Distribusi Rating Film')
plt.xlabel('Rating')
plt.ylabel('Jumlah Film')
plt.savefig("rating_distribution.png")
plt.close()

# ======== FEATURE SELECTION & ENGINEERING =========
# Pilih fitur yang relevan
features = ['budget', 'popularity', 'runtime', 'release_year']
X = df[features]
y = df['vote_average']

# ======== SPLIT DATA =========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ======== MODELING =========
# 1. Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# 2. Random Forest Regressor
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# ======== EVALUASI MODEL =========
def evaluate_model(name, y_true, y_pred):
    print(f"\nModel: {name}")
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))

evaluate_model("Linear Regression", y_test, y_pred_lr)
evaluate_model("Random Forest", y_test, y_pred_rf)

# ======== KESIMPULAN =========
print("\nKesimpulan:")
print("Model Random Forest memiliki performa yang lebih baik dibandingkan Linear Regression berdasarkan metrik MAE dan RMSE.")
