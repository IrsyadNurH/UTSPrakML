**# Proyek Klasifikasi Buah: Jeruk vs Anggur

## Deskripsi Proyek
Proyek ini bertujuan untuk mengembangkan model klasifikasi untuk mengidentifikasi apakah sebuah buah adalah jeruk atau anggur berdasarkan fitur-fitur seperti diameter, berat, dan nilai warna (red, green, blue). Model klasifikasi yang digunakan adalah **Decision Tree**.

## Langkah-langkah Pembuatan Model

### 1. **Persiapan Data**
   - Dataset yang digunakan berisi informasi tentang berbagai buah, dengan kolom-kolom berikut: `name`, `diameter`, `weight`, `red`, `green`, `blue`.
   - Label pada dataset: `0` untuk jeruk (orange) dan `1` untuk anggur (grapefruit).
   
   **Contoh Data:**
   ```
   name    diameter   weight   red   green  blue
   orange  2.96       86.76    172   85     2
   orange  3.91       88.05    166   78     3
   orange  4.42       95.17    156   81     2
   grapefruit 4.47    95.60    163   81     4
   ```

### 2. **Preprocessing Data**
   - **Mengubah Label**: Kolom `name` yang berisi nama buah ("orange" dan "grapefruit") diubah menjadi label numerik (`0` untuk jeruk, `1` untuk anggur) menggunakan `apply()`:
     ```python
     df['label'] = df['name'].apply(lambda x: 0 if x == 'orange' else 1)Q
     ```
   - **Memisahkan Fitur dan Label**:
     - Fitur: `diameter`, `weight`, `red`, `green`, `blue`.
     - Label: `label` yang telah diubah.
     ```python
     X = df.drop(['name', 'label'], axis=1)  # Mengambil fitur
     y = df['label']  # Label
     ```

### 3. **Pembagian Data**
   - **Membagi Data Latih dan Data Uji**: Dataset dibagi menjadi data latih (80%) dan data uji (20%) menggunakan `train_test_split` dari `sklearn`:
     ```python
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
     ```

### 4. **Pembangunan Model**
   - **Menggunakan Decision Tree**: Model Decision Tree dibangun dengan menggunakan `DecisionTreeClassifier` dari `sklearn`:
     ```python
     from sklearn.tree import DecisionTreeClassifier
     model = DecisionTreeClassifier(random_state=42)
     model.fit(X_train, y_train)
     ```

### 5. **Evaluasi Model**
   - **Prediksi pada Data Uji**: Menggunakan model untuk memprediksi kelas buah pada data uji.
     ```python
     y_pred = model.predict(X_test)
     ```
   - **Menghitung Akurasi**: Akurasi model dihitung dengan `accuracy_score`:
     ```python
     accuracy = accuracy_score(y_test, y_pred)
     print(f'Akurasi: {accuracy * 100:.2f}%')
     ```

   - **Precision, Recall, dan F1-Score**: Menghitung precision, recall, dan F1-score untuk kelas "orange" (label `0`):
     ```python
     precision = precision_score(y_test, y_pred, pos_label=0)
     recall = recall_score(y_test, y_pred, pos_label=0)
     f1 = f1_score(y_test, y_pred, pos_label=0)
     print(f'Presisi untuk kelas "orange": {precision:.2f}')
     print(f'Recall untuk kelas "orange": {recall:.2f}')
     print(f'F1-Score untuk kelas "orange": {f1:.2f}')
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

1. Akurasi:
Akurasi: 94.35%

Artinya, model berhasil memprediksi dengan benar sekitar 94.35% dari data uji. Ini adalah ukuran umum untuk mengevaluasi performa model secara keseluruhan.

2. Presisi untuk Kelas "orange" (label 0):
Presisi untuk kelas "orange": 0.95

Ini berarti bahwa dari semua prediksi yang model buat sebagai jeruk (orange), 95% di antaranya benar-benar jeruk. Presisi mengukur akurasi prediksi positif.

3. Recall untuk Kelas "orange" (label 0):
Recall untuk kelas "orange": 0.94

Recall mengukur seberapa baik model dalam menangkap semua contoh jeruk (orange) dari keseluruhan data yang benar-benar jeruk. Di sini, model berhasil menangkap 94% jeruk yang ada dalam data uji.

4. F1-Score untuk Kelas "orange" (label 0):
F1-Score untuk kelas "orange": 0.94

F1-Score adalah rata-rata harmonis antara presisi dan recall. Nilai 0.94 menunjukkan bahwa model memiliki keseimbangan yang baik antara presisi dan recall untuk kelas "orange".
Matriks kebingungannya menunjukkan distribusi hasil prediksi dengan jumlah **True Positives**, **False Positives**, **True Negatives**, dan **False Negatives**.
**
