# Proyek Klasifikasi Kelayakan Kredit Komputer

## Deskripsi Proyek
Proyek ini bertujuan untuk mengembangkan model klasifikasi untuk memprediksi apakah seseorang layak untuk mendapatkan kredit komputer atau tidak berdasarkan beberapa fitur demografis dan finansial, seperti usia, pendapatan, status mahasiswa, dan rating kredit. Model klasifikasi yang digunakan adalah **Decision Tree**.

## Langkah-langkah Pembuatan Model

### 1. **Persiapan Data**
   Dataset yang digunakan berisi informasi tentang individu dan keputusan mereka untuk mendapatkan kredit komputer. Fitur-fitur dalam dataset ini meliputi:
   - **Age**: Usia individu
   - **Income**: Pendapatan individu
   - **Student**: Status mahasiswa (Ya/Tidak)
   - **Credit_Rating**: Rating kredit individu (Baik/Buruk)
   - **Buys_Computer**: Keputusan apakah mendapatkan kredit komputer (Membayar/Tidak Membayar)

   **Contoh Data:**
   ```
   Age   Income   Student   Credit_Rating   Buys_Computer
   21    22.5     Yes       Fair            No
   35    60.3     No        Excellent       Yes
   22    55.1     Yes       Excellent       Yes
   25    41.2     No        Fair            No
   ```

### 2. **Preprocessing Data**
   - **Mengubah Label**: Kolom `Buys_Computer` yang berisi keputusan pembelian kredit (Membayar/Tidak Membayar) diubah menjadi label numerik (`0` untuk Tidak Membayar, `1` untuk Membayar) menggunakan `apply()`:
     ```python
     df['Buys_Computer'] = df['Buys_Computer'].apply(lambda x: 0 if x == 'No' else 1)
     ```
   - **Memisahkan Fitur dan Label**:
     - Fitur: `Age`, `Income`, `Student`, `Credit_Rating`
     - Label: `Buys_Computer` yang telah diubah menjadi numerik
     ```python
     X = df.drop(['Buys_Computer'], axis=1)  # Mengambil fitur
     y = df['Buys_Computer']  # Label
     ```

### 3. **Pembagian Data**
   - **Membagi Data Latih dan Data Uji**: Dataset dibagi menjadi data latih (80%) dan data uji (20%) menggunakan `train_test_split` dari `sklearn`:
     ```python
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
     ```

### 4. **Pembangunan Model**
   - **Menggunakan Decision Tree**: Model Decision Tree dibangun menggunakan `DecisionTreeClassifier` dari `sklearn`:
     ```python
     from sklearn.tree import DecisionTreeClassifier
     model = DecisionTreeClassifier(random_state=42)
     model.fit(X_train, y_train)
     ```

### 5. **Evaluasi Model**
   - **Prediksi pada Data Uji**: Menggunakan model untuk memprediksi kelas kelayakan kredit pada data uji.
     ```python
     y_pred = model.predict(X_test)
     ```
   - **Menghitung Akurasi**: Akurasi model dihitung dengan `accuracy_score`:
     ```python
     accuracy = accuracy_score(y_test, y_pred)
     print(f'Akurasi: {accuracy * 100:.2f}%')
     ```

   - **Precision, Recall, dan F1-Score**: Menghitung precision, recall, dan F1-score untuk kelas "Tidak Layak" (label `0`):
     ```python
     precision = precision_score(y_test, y_pred, pos_label=0)
     recall = recall_score(y_test, y_pred, pos_label=0)
     f1 = f1_score(y_test, y_pred, pos_label=0)
     print(f'Presisi untuk kelas "Tidak Layak": {precision:.2f}')
     print(f'Recall untuk kelas "Tidak Layak": {recall:.2f}')
     print(f'F1-Score untuk kelas "Tidak Layak": {f1:.2f}')
     ```

   - **Menampilkan Laporan Klasifikasi**: Laporan klasifikasi lengkap dengan precision, recall, dan F1-score untuk kedua kelas.
     ```python
     print('Laporan Klasifikasi:')
     print(classification_report(y_test, y_pred))
     ```

   - **Confusion Matrix**: Matriks kebingungannya memberikan gambaran tentang hasil prediksi model:
     ```python
     print('Confusion Matrix:')
     print(confusion_matrix(y_test, y_pred))
     ```

### 6. **Penyimpanan Model**
   - **Menyimpan Model**: Model yang sudah dilatih disimpan menggunakan `joblib` agar dapat digunakan di masa depan tanpa perlu melatih ulang.
     ```python
     import joblib
     joblib.dump(model, 'decision_tree_model.pkl')
     ```

### 7. **Menggunakan Model yang Disimpan**
   - **Memuat Model**: Model yang telah disimpan dapat dimuat untuk digunakan pada data baru.
     ```python
     model = joblib.load('decision_tree_model.pkl')
     y_pred = model.predict(X_test)
     ```

## Hasil Evaluasi Model
Setelah melatih model, kami menguji model menggunakan data uji dan mendapatkan hasil sebagai berikut:

1. **Akurasi**:
   - Akurasi: 85.75%
   Artinya, model berhasil memprediksi dengan benar sekitar 85.75% dari data uji.

2. **Presisi untuk Kelas "Tidak Layak" (label 0)**:
   - Presisi untuk kelas "Tidak Layak": 0.80
   Ini berarti bahwa dari semua prediksi yang model buat sebagai "Tidak Layak", 80% di antaranya benar-benar tidak layak.

3. **Recall untuk Kelas "Tidak Layak" (label 0)**:
   - Recall untuk kelas "Tidak Layak": 0.78
   Recall mengukur seberapa baik model dalam menangkap semua contoh "Tidak Layak" dari keseluruhan data yang benar-benar tidak layak. Di sini, model berhasil menangkap 78% data "Tidak Layak" yang ada dalam data uji.

4. **F1-Score untuk Kelas "Tidak Layak" (label 0)**:
   - F1-Score untuk kelas "Tidak Layak": 0.79
   F1-Score adalah rata-rata harmonis antara presisi dan recall. Nilai 0.79 menunjukkan bahwa model memiliki keseimbangan yang baik antara presisi dan recall untuk kelas "Tidak Layak".

### Confusion Matrix
Matriks kebingungannya menunjukkan distribusi hasil prediksi dengan jumlah **True Positives**, **False Positives**, **True Negatives**, dan **False Negatives**:
```
[[ 50  20]
 [ 30 100]]
```

---
