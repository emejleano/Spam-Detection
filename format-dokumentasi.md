# Submission 1: Nama Proyek Anda
Nama:Emejleano Rusmin Nggepo

Username dicoding:emejleano

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [spam and ham dataset](https://www.kaggle.com/datasets/venkateshch22384/spam-and-ham-email-dataset) |
| Masalah | Email spam merupakan masalah yang mengganggu pengguna internet secara luas. Email spam tidak hanya mengganggu, tetapi juga dapat membawa risiko keamanan dan memakan waktu pengguna. Dengan meningkatnya jumlah email yang masuk, deteksi dan klasifikasi otomatis antara spam dan ham menjadi penting untuk meningkatkan produktivitas dan keamanan pengguna. |
| Solusi machine learning | Solusi yang diusulkan adalah memanfaatkan teknik machine learning untuk membangun sistem yang dapat membedakan antara email spam dan ham secara otomatis. Dengan menggunakan model machine learning, sistem dapat memfilter email yang masuk ke dalam kotak masuk pengguna berdasarkan kemungkinan menjadi spam atau ham. |
| Metode pengolahan | Data email akan diolah dengan teknik pengolahan data seperti pembuatan statistik, pembuatan skema, validasi contoh, dan transformasi data. Pembagian data dilakukan dengan rasio 80:20 untuk data training dan evaluasi. Proses transformasi akan melibatkan pengubahan nama fitur yang telah ditransformasi dan penerapan one-hot encoding untuk fitur kelas. |
| Arsitektur model | Model akan menggunakan arsitektur dengan tiga layer Dense, masing-masing dengan 256, 64, dan 16 neuron dengan fungsi aktivasi ReLU. Layer output akan menggunakan Dense dengan satu neuron dan fungsi aktivasi sigmoid untuk mengklasifikasikan antara spam dan ham. Model akan dikompilasi menggunakan optimizers Adam dengan learning rate 0.001, loss binary crossentropy, dan metrik BinaryAccuracy. |
| Metrik evaluasi | Metrik evaluasi yang digunakan meliputi AUC, Precision, Recall, ExampleCount, dan BinaryAccuracy. Metrik-metrik ini akan memberikan gambaran tentang seberapa baik model dapat membedakan antara spam dan ham. |
| Performa model | Performa model akan dievaluasi berdasarkan akurasi dan lossnya. Model yang baik akan memiliki akurasi yang tinggi dan loss yang rendah. Dari hasil evaluasi, model ini mendapatkan akurasi 0.97 pada proses training dan validasi, serta loss sebesar 0.0817, menunjukkan kinerja yang baik dalam klasifikasi spam dan ham. |
| Opsi deployment | Deksripsi tentang opsi deployment |
| Web app | Tautan web app yang digunakan untuk mengakses model serving. Contoh: [nama-model]|
| Monitoring | Monitoring sistem akan dilakukan menggunakan Prometheus dan Grafana. Sistem akan memantau setiap permintaan (request) yang masuk dan menampilkan statusnya, termasuk jika request berhasil diklasifikasikan, jika request tidak ditemukan, atau jika terdapat argumen yang tidak valid. Ini akan membantu dalam memantau kinerja sistem secara real-time. |
