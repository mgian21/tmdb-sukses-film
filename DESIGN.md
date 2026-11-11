# Design Direction: Proyeksi Box Office TMDB 5000

## 1. Identitas & Karakter Produk
- **Produk**: Aplikasi web interaktif berbasis Machine Learning untuk memproyeksikan apakah sebuah produksi film akan balik modal (*Box Office Hit*, `revenue > budget`) atau merugi (*Box Office Flop*) berdasarkan arsip **TMDB 5000 Movie Dataset**.
- **Mood & Referensi Visual**: Ruang pemutaran teater sinema (*screening room*) yang dipadukan dengan jurnal editorial analisis *Box Office* klasik dan potongan tiket bioskop (*ticket stub*).
- **Design Read**:
  > Reading this as: Interactive ML Box Office analytical tool for film students and cinema analysts, in an editorial screening-room visual language, dial ENERGY 2 / RHYTHM 2 / MOTION 1.
- **Dial Liveliness**: `Dial: ENERGY 2 / RHYTHM 2 / MOTION 1`

## 2. Sistem Palet Warna (R-29: 2 Core + 1 Accent)
- **Core 1 - Latar & Permukaan (Neutral Base)**:
  - *Theater Obsidian* (`#0D0F12`): Latar utama halaman yang meniru suasana ruang bioskop redup agar fokus mata tertuju pada layar data.
  - *Screen Charcoal* (`#161920`): Permukaan panel simulasi, tabel, dan kanvas grafik.
  - *Frame Slate* (`#262B36`): Garis pembatas struktural dan perforasi tiket.
- **Core 2 - Tipografi & Konten (WCAG AA Verified)**:
  - *Warm Film Ivory* (`#F4F1EA`): Teks utama dan judul (rasio kontras `16.4:1` di atas `#0D0F12` dan `14.2:1` di atas `#161920`).
  - *Projector Silver* (`#B8B2A6`): Teks sekunder, label sumbu, dan keterangan (rasio kontras `9.2:1` di atas `#0D0F12` dan `8.0:1` di atas `#161920`).
- **Deliberate Accent (1 Warna Aksen Utama)**:
  - *Marquee Amber Gold* (`#E5A93C`): Digunakan secara terfokus pada tombol aksi utama (*Hitung Proyeksi Box Office*), garis aksen tajuk *playbill*, dan penanda nilai tertinggi (teks gelap `#0D0F12` di atas tombol `#E5A93C` memiliki rasio kontras `9.5:1`).
  - Penanda Semantik Data: *Emerald Reel* (`#4CAF7D`) untuk status untung/sukses dan *Crimson Curtain* (`#D9534F`) untuk status rugi/tidak sukses.

## 3. Tipografi (R-06)
- **Judul & Tajuk Editorial**: `'Playfair Display', Georgia, serif`. Dipilih untuk menghadirkan karakter katalog festival film dan tajuk jurnal sinema.
- **Teks Antarmuka & Kontrol**: `'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif`. Dipilih demi keterbacaan tinggi pada kontrol *slider*, tabel perbandingan model, dan angka statistik.
- **Angka & Metrik**: Menggunakan `font-variant-numeric: tabular-nums` agar nilai dolar anggaran, persentase akurasi, dan skor F1 sejajar rapi.

## 4. Alasan Satu Baris untuk Setiap Keputusan Utama (R-31)
- **Mengapa palet warna ini**: Warna dasar gelap (`#0D0F12`) dan gading hangat (`#F4F1EA`) mereplikasi kenyamanan ruang teater sinema sekaligus menjaga kontras baca di atas standar WCAG AA (`> 8:1`), sementara emas *marquee* (`#E5A93C`) mengarahkan fokus ke aksi proyeksi utama.
- **Mengapa tipografi ini**: Kombinasi serif editorial (`Playfair Display`) pada tajuk dan sans-serif bersih (`Plus Jakarta Sans`) pada komponen data membedakan narasi sinema dari angka evaluasi teknis.
- **Mengapa tata letak ini**: Menempatkan **Studio Proyeksi Box Office** langsung di bagian utama (di atas tab evaluasi dan arsip) memprioritaskan keputusan utama pengguna, yaitu menguji skenario film, sebelum menelusuri tabel pembuktian model.
- **Mengapa variasi ritme dan spasi ini (`RHYTHM 2`)**: Pembagian kolom asimetris (`1.15 : 0.85` pada simulator dan `1.2 : 0.8` pada evaluasi model) mencegah kebosanan grid seragam dan memberi ruang lebih besar pada elemen berdensitas data tinggi.
- **Mengapa motif potongan tiket (*ticket stub*)**: Garis ganda di bagian tajuk dan garis putus-putus (*perforated rule*) pada panel hasil prediksi memberikan identitas khas bioskop tanpa perlu dekorasi berlebihan atau gambar generik.
- **Mengapa tanpa emoji dekoratif (`R-04`)**: Menghapus emoji dari judul, tab, dan pesan hasil menghilangkan kesan template buatan AI serta menjaga fokus pada ketegasan tipografi dan bukti data.
