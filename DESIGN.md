# Design Direction: Proyeksi Box Office TMDB 5000

## 1. Identitas & Karakter Produk
- **Produk**: Aplikasi web interaktif berbasis Machine Learning untuk memproyeksikan apakah sebuah produksi film akan balik modal (*Box Office Hit*, `revenue > budget`) atau merugi (*Box Office Flop*) berdasarkan arsip **TMDB 5000 Movie Dataset**.
- **Mood & Referensi Visual**: Ruang pemutaran teater sinema (*Obsidian* & *Charcoal*) dengan tiket fisik berwarna krem (*Warm Film Ivory*) ber-stub *Crimson*, serta tipografi tajuk *Playfair Display* `88px` dan *DM Sans* untuk tabel data.
- **Design Read**:
  > Reading this as: Interactive ML Box Office analytical tool for film students and cinema analysts, in an editorial screening-room visual language, dial ENERGY 2 / RHYTHM 2 / MOTION 1.
- **Dial Liveliness**: `Dial: ENERGY 2 / RHYTHM 2 / MOTION 1`

## 2. Sistem Palet Enam Warna (R-29)
1. **Obsidian (`#0D0F12`)**: Latar utama halaman.
2. **Charcoal (`#161920`)**: Latar bagian tabel selebar halaman dan kanvas grafik.
3. **Ivory (`#F4F1EA`)**: Teks utama di atas latar gelap serta warna dasar kartu tiket prediksi.
4. **Silver (`#A39E93`)**: Teks pendukung, keterangan sumbu, dan label sekunder.
5. **Amber (`#E5A93C`)**: Tombol aksi utama dan sorotan nilai metrik terbaik pada tabel evaluasi.
6. **Crimson (`#C83E3B`)**: Penanda penting khusus untuk stub tiket prediksi, label `"PEMBANDING"`, dan garis bawah tautan.

## 3. Tipografi (R-06)
- **Judul Utama (`Playfair Display`)**: `'Playfair Display', Georgia, serif` berukuran `88px` pada tajuk utama untuk menghadirkan bobot poster dan tajuk utama jurnal sinema.
- **Tabel & Data (`DM Sans`)**: `'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif` dengan `font-variant-numeric: tabular-nums` untuk keterbacaan tabel selebar halaman, angka anggaran, dan metrik evaluasi.

## 4. Komposisi & Gerakan (`Energy 2 / Rhythm 2 / Motion 1`)
- **Tanpa animasi saat halaman dimuat (`MOTION 1`)**: Gerakan dibatasi hanya pada transisi *hover* pendek (`0.15s ease`) untuk tombol, tautan bernavigasi garis bawah Crimson, dan baris tabel.
- **Pergantian komposisi per bagian (`RHYTHM 2`)**:
  1. **Bagian Utama**: Dua kolom asimetris (kiri: parameter produksi film dan tombol Amber; kanan: kartu tiket krem `#F4F1EA` dengan stub Crimson `#C83E3B` dan label `"PEMBANDING"`).
  2. **Bagian Tabel Selebar Halaman**: Tabel perbandingan model dan tabel eksplorasi arsip TMDB 5000 ditempatkan selebar halaman (*full-width*) di atas latar Charcoal (`#161920`) tanpa geser kiri-kanan (*no horizontal scroll*).
  3. **Bagian Spesifikasi Desain**: Grid palet enam kolom sesuai kode hex beserta spesimen huruf `Playfair Display` dan `DM Sans`.

## 5. Alasan Satu Baris untuk Setiap Keputusan Utama (R-31)
- **Mengapa palet enam warna ini**: Obsidian (`#0D0F12`) dan Charcoal (`#161920`) menciptakan kontras ruang teater, Ivory (`#F4F1EA`) dan Crimson (`#C83E3B`) membedakan fisik kartu tiket dan penanda pembanding, sementara Amber (`#E5A93C`) mengunci fokus pada tombol utama dan nilai metrik tertinggi.
- **Mengapa pasangan huruf ini**: `Playfair Display` berukuran `88px` memberi karakter tajuk sinema yang tegas di bagian pembuka, sedangkan `DM Sans` menjaga ketajaman angka dan teks pada tabel berdensitas tinggi.
- **Mengapa tabel dibuat selebar halaman di atas latar Charcoal**: Menempatkan tabel secara *full-width* dengan pembungkusan teks otomatis menghilangkan *horizontal scrollbar* (geser kiri-kanan) sehingga seluruh kolom langsung terbaca utuh.
- **Mengapa hasil prediksi berbentuk tiket krem dengan stub Crimson**: Kontras permukaan krem (`#F4F1EA`) di atas halaman gelap (`#0D0F12`) dengan potongan stub merah (`#C83E3B`) menjadikan hasil proyeksi titik fokus instan yang menyerupai tiket bioskop fisik.
- **Mengapa tanpa emoji dekoratif (`R-04`)**: Menghapus emoji menjaga wibawa editorial jurnal film dan menyerahkan hierarki sepenuhnya pada tipografi, warna aksen, dan struktur tata letak.
