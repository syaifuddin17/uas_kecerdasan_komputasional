# cnn_hoax_classifier.py

# ===============================
# CNN untuk Klasifikasi Berita Hoaks atau Asli
# ===============================

# Instalasi TensorFlow (jika di Colab)
try:
    import tensorflow as tf
except ImportError:
    !pip install tensorflow
    import tensorflow as tf

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# --------------------------------
# 1. Dataset Manual (20+ Data)
# --------------------------------
data = [
    ("Vaksin COVID-19 menyebabkan autisme", "hoax"),
    ("Pemerintah mengumumkan cuti bersama Lebaran", "real"),
    ("Minum air kelapa bisa menyembuhkan kanker", "hoax"),
    ("BMKG mengumumkan prakiraan cuaca ekstrem", "real"),
    ("Hoaks vaksinasi COVID bisa mengubah DNA manusia", "hoax"),
    ("Presiden meresmikan jalan tol baru", "real"),
    ("Menghirup uap garam bisa membunuh virus corona", "hoax"),
    ("Kementerian Kesehatan keluarkan aturan baru", "real"),
    ("Makan cabai merah bisa sembuhkan flu berat", "hoax"),
    ("Bank Indonesia keluarkan uang edisi khusus", "real"),
    ("Cuci tangan bisa membunuh bakteri", "real"),
    ("Matahari bisa menyembuhkan COVID-19", "hoax"),
    ("Virus COVID-19 adalah ciptaan manusia", "hoax"),
    ("Kementerian Pendidikan meluncurkan kurikulum baru", "real"),
    ("Meminum jus lemon bisa membunuh virus", "hoax"),
    ("Presiden menyampaikan pidato kenegaraan", "real"),
    ("Minyak kayu putih terbukti sembuhkan COVID", "hoax"),
    ("Polri tangkap pelaku terorisme di Jakarta", "real"),
    ("Bawang putih bisa jadi vaksin alami", "hoax"),
    ("KPU umumkan hasil pemilu 2024", "real")
]

# --------------------------------
# 2. Preprocessing
# --------------------------------
texts = [t[0] for t in data]
labels = [t[1] for t in data]

# Tokenisasi
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, padding='post', maxlen=50)

# Encode label
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, encoded_labels, test_size=0.2, random_state=42)

# --------------------------------
# 3. Model CNN
# --------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=5000, output_dim=32, input_length=50),
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# --------------------------------
# 4. Training
# --------------------------------
print("\n=== Training Model ===\n")
history = model.fit(np.array(X_train), np.array(y_train), epochs=10, validation_data=(X_test, y_test))

# --------------------------------
# 5. Evaluasi
# --------------------------------
print("\n=== Evaluasi Model ===\n")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Akurasi: {accuracy*100:.2f}%")

# --------------------------------
# 6. Prediksi Contoh
# --------------------------------
print("\n=== Contoh Prediksi ===")
test_sentence = ["Pemerintah memperpanjang PPKM sampai akhir bulan"]
seq = tokenizer.texts_to_sequences(test_sentence)
padded = pad_sequences(seq, maxlen=50, padding='post')
pred = model.predict(padded)
label = "hoax" if pred[0][0] < 0.5 else "real"
print(f"Teks: {test_sentence[0]}")
print(f"Prediksi: {label} (Probabilitas: {pred[0][0]:.2f})")
