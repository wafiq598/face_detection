import cv2 #Ini adalah Library Open Cv

def main():
    # Muat Haar Cascade untuk deteksi wajah
    # Pastikan file 'haarcascade_frontalface_default.xml' ada di direktori yang sama
    # atau berikan path lengkap ke file tersebut.
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Inisialisasi kamera (0 adalah ID untuk kamera default)
    # Jika Anda memiliki beberapa kamera, coba ganti angka 0 dengan 1, 2, dst.
    cap = cv2.VideoCapture(0)

    # Periksa apakah kamera berhasil dibuka
    if not cap.isOpened():
        print("Error: Tidak dapat membuka kamera.")
        return

    print("Tekan 'q' untuk keluar dari jendela deteksi wajah.")

    while True:
        # Baca frame demi frame dari kamera
        ret, frame = cap.read()

        # Jika frame tidak berhasil dibaca, keluar dari loop
        if not ret:
            print("Error: Gagal membaca frame.")
            break

        # Ubah frame menjadi grayscale untuk deteksi yang lebih cepat
        # Haar Cascades bekerja lebih baik pada gambar grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Deteksi wajah dalam frame grayscale
        # - scaleFactor: Seberapa banyak ukuran gambar dikurangi pada setiap skala gambar.
        #                Ini mengompensasi fakta bahwa wajah bisa lebih dekat atau lebih jauh.
        # - minNeighbors: Berapa banyak tetangga yang harus dimiliki setiap kandidat persegi panjang
        #                 untuk mempertahankan kandidatnya. Nilai yang lebih tinggi menghasilkan
        #                 lebih sedikit deteksi palsu tetapi dapat melewatkan beberapa wajah.
        # - minSize: Ukuran objek minimum yang dianggap sebagai wajah.
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Gambar persegi panjang di sekitar setiap wajah yang terdeteksi
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) # Warna biru, tebal 2

        # Tampilkan frame dengan deteksi wajah
        cv2.imshow('Deteksi Wajah Real-time', frame)

        # Keluar dari loop jika tombol 'q' ditekan
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Bebaskan sumber daya kamera dan tutup semua jendela OpenCV
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()