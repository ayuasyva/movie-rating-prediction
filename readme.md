# 🎬 Movie Rating Prediction with Machine Learning

Proyek ini bertujuan untuk memprediksi **rating film** berdasarkan berbagai fitur seperti durasi, popularitas, tahun rilis, dan anggaran produksi. Proyek ini merupakan bagian dari portofolio saya sebagai **freelance data scientist pemula**.

---

## 📦 Dataset
Dataset yang digunakan adalah **TMDB 5000 Movie Dataset** yang tersedia di [Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).

File utama:
- `tmdb_5000_movies.csv`

---

## 🔧 Tools & Library
Project ini dibangun menggunakan:
- Python
- Pandas, Numpy
- Matplotlib, Seaborn
- Scikit-learn

---

## 📊 Tahapan Project
1. **Data Cleaning**
   - Menghapus kolom yang tidak relevan
   - Menangani missing values
   - Ekstraksi tahun dari tanggal rilis

2. **Exploratory Data Analysis (EDA)**
   - Distribusi rating film
   - Korelasi antar fitur

3. **Feature Engineering**
   - Fitur utama: `budget`, `popularity`, `runtime`, `release_year`

4. **Modeling**
   - Linear Regression
   - Random Forest Regressor

5. **Evaluasi Model**
   - Metrik: MAE (Mean Absolute Error) dan RMSE (Root Mean Squared Error)

---

## 📈 Hasil Evaluasi
| Model              | MAE     | RMSE    |
|-------------------|---------|---------|
| Linear Regression | 0.679   | 0.978   |
| Random Forest     | **0.632** | **0.874** |

> 🔍 **Kesimpulan:** Model Random Forest memiliki performa lebih baik dibanding Linear Regression dalam memprediksi rating film.

---

## 💡 Insight
- Popularitas dan runtime berkontribusi besar dalam prediksi rating.
- Film dengan anggaran besar belum tentu mendapatkan rating tinggi.
- Tahun rilis film memiliki pola tren terhadap rata-rata rating.

---

## 🧑‍💻 Tentang Saya
Nama: **Ayu Asyva Irfita**  
Posisi:  Data Scientist  
Portofolio: [Instagram - @asyvayuirfita](https://www.instagram.com/asyvayuirfita)  
Email: ayuasyvairfita16@gmail.com

---

## 📁 Struktur Project
