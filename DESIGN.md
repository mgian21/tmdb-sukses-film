# Design Direction: Proyeksi Box Office TMDB 5000 (Noir & Emerald)

## 1. Identitas & Karakter Produk
- **Produk**: Aplikasi analisis dan simulasi *Machine Learning* untuk memproyeksikan apakah produksi sebuah film akan balik modal (*Box Office Hit*, `revenue > budget`) atau merugi (*Box Office Flop*) berdasarkan arsip **TMDB 5000 Movie Dataset**.
- **Mood & Referensi Visual**: **Noir & Emerald (Nuansa Bioskop Klasik / Retro Prestisius)**. Menggabungkan kedalaman ruang teater klasik berwarna hijau-hitam pinus (*Deep Pine Black* & *Dark Forest*) dengan kertas tiket hangat (*Warm Paper*), kilau kuningan teater (*Antique Gold*), hijau zamrud (*Emerald*), dan aksen oranye karat (*Rust Orange*).
- **Design Read**:
  > Reading this as: Interactive ML Box Office analytical tool for film students and cinema analysts, in a Noir & Emerald retro-prestigious cinema visual language, dial ENERGY 2 / RHYTHM 2 / MOTION 1.
- **Dial Liveliness**: `Dial: ENERGY 2 / RHYTHM 2 / MOTION 1`

## 2. Sistem Palet Warna: Noir & Emerald (R-29 & R-25 WCAG AA Verified)
1. **Dark Base: Deep Pine Black (`#0B0E0D`)**: Latar utama halaman dan warna teks kontras tinggi di atas kartu tiket krem maupun tombol emas.
2. **Surface / Card: Dark Forest (`#141A17`)**: Permukaan bagian tabel selebar halaman, pita ringkasan arsip, dan kanvas grafik Matplotlib.
3. **Main Text & Ticket Surface: Warm Paper (`#F2EFE9`)**: Teks utama di atas latar gelap (rasio kontras `16.8:1` terhadap `#0B0E0D` dan `15.1:1` terhadap `#141A17`) serta latar fisik kartu tiket hasil proyeksi.
4. **Muted Text: Sage Gray (`#8F9E96`)**: Teks pendukung, label sekunder, dan keterangan sumbu grafik (rasio kontras `6.74:1` terhadap `#0B0E0D` dan `6.03:1` terhadap `#141A17`).
5. **Primary Accent: Antique Gold (`#D4AF37`) & Emerald (`#10B981`)**:
   - *Antique Gold* (`#D4AF37`): Tombol aksi utama (`#0B0E0D` di atas `#D4AF37` memiliki kontras `9.1:1`), garis tajuk atas, dan sorotan nilai terbaik (`★`) pada tabel evaluasi model.
   - *Emerald* (`#10B981`): Penanda semantik keberhasilan *Box Office Hit (Untung)* pada tabel arsip, grafik distribusi balik modal, dan lencana status untung.
6. **Secondary Accent: Rust Orange (`#C2410C`)**: Digunakan secara disiplin untuk tiga penanda penting: **stub tiket** prediksi (kontras teks `#F2EFE9` di atas `#C2410C` adalah `5.50:1`), **label `"PEMBANDING"`**, dan **garis bawah tautan** navigasi bagian (serta penanda status rugi/data belum lengkap).

## 3. Tipografi (R-06)
- **Judul Utama (`Playfair Display`)**: `'Playfair Display', Georgia, serif` berukuran `88px` pada tajuk utama untuk menghadirkan wibawa poster sinema klasik dan jurnal film prestisius.
- **Tabel & Data (`DM Sans`)**: `'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif` dengan `font-variant-numeric: tabular-nums` untuk menjaga keterbacaan tabel selebar halaman, angka dolar, dan metrik evaluasi.

## 4. Komposisi & Gerakan (`Energy 2 / Rhythm 2 / Motion 1`)
- **Tanpa animasi saat halaman dimuat (`MOTION 1`)**: Gerakan dibatasi hanya pada transisi *hover* pendek (`0.15s ease`) untuk tombol, tautan bernavigasi garis bawah Rust Orange, dan baris tabel.
- **Pergantian komposisi per bagian (`RHYTHM 2`)**:
  1. **Bagian Utama (`#studio-proyeksi`)**: Dua kolom asimetris (`1.12 : 0.88`) yang menyandingkan parameter produksi film di kiri dengan kartu tiket *Warm Paper* (`#F2EFE9`) ber-stub *Rust Orange* (`#C2410C`) di kanan.
  2. **Bagian Evaluasi Model (`#evaluasi-model`)**: Tabel perbandingan metrik selebar halaman di atas latar *Dark Forest* (`#141A17`) diikuti dua kolom diagnostik (Matriks Evaluasi & 8 Fitur Terpenting).
  3. **Bagian Arsip TMDB 5000 (`#arsip-tmdb`)**: Grafik distribusi historis diikuti tabel eksplorasi arsip selebar halaman di atas latar *Dark Forest* (`#141A17`) tanpa *horizontal scroll*.

## 5. Alasan Satu Baris untuk Setiap Keputusan Utama (R-31)
- **Mengapa palet Noir & Emerald ini**: *Deep Pine Black* (`#0B0E0D`) dan *Dark Forest* (`#141A17`) membangun suasana teater klasik yang teduh di mata, sementara *Antique Gold* (`#D4AF37`), *Emerald* (`#10B981`), dan *Rust Orange* (`#C2410C`) membedakan fungsi aksi utama, status keuntungan, dan penanda pembanding secara tegas.
- **Mengapa pasangan huruf ini**: `Playfair Display` (`88px`) menghadirkan karakter editorial sinema retro pada tajuk, sedangkan `DM Sans` memastikan kepadatan angka di tabel tetap jernih.
- **Mengapa tabel dibuat selebar halaman di atas latar Dark Forest**: Lebar penuh (`width: 100%; table-layout: fixed`) dengan angka dolar ringkas menghilangkan *horizontal scrollbar* sehingga pengguna dapat memindai seluruh kolom sekaligus.
- **Mengapa kartu tiket menggunakan latar Warm Paper dengan stub Rust Orange**: Kontras kertas hangat (`#F2EFE9`) di atas latar gelap (`#0B0E0D`) dengan potongan stub `#C2410C` menjadikan hasil proyeksi titik fokus utama yang menyerupai tiket bioskop fisik.
- **Mengapa bagian sampel palet dihapus dari UI**: Sesuai prinsip *Content-Driven Composition* (`C-3`), aplikasi hanya menampilkan fitur yang melayani tujuan analisis pengguna, sementara spesifikasi desain didokumentasikan di `DESIGN.md`.
