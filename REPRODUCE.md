# Reproduksi eksperimen

## Pemeriksaan hasil tersimpan

Jalankan `python scripts/verify_experiments.py`. Program menghitung ulang AUROC
dan penemuan pada sepuluh urutan pertama dari prediksi 48 foto, mengulang
perbandingan 43 foto dalam komunitas yang sama, serta membentuk ulang batch
50 foto dari katalog dan kebijakan yang dibekukan. Pengujian aplikasi berada
di `solution/evidence_intake/tests`; uji HTTP:

```sh
python scripts/test_inspection_planner_http_20260928.py --base-url http://127.0.0.1:8879 --output outputs/http-test
```

## Data, model, dan urutan analisis

1. Sediakan data BDC resmi secara lokal. Semua masukan citra dicocokkan melalui
   SHA-256 dan pemetaan foto kanonik. Jangan mengganti nama/foto tanpa memperbarui
   manifest. Model, revisi, hash, preprocessing dan pooling dicatat dalam
   `experiments/model_manifest.json`; setiap bank berbeda disimpan terpisah.
2. Ekstraksi CUDA tersedia dalam `scripts/cloud_dinov3_global_20260928.py`,
   `cloud_siglip2_so400m_naflex_20260928.py`, `cloud_sam3_dinov3_region_bank_20260928.py`,
   dan skrip crop region. Awalan cloud hanya menunjukkan konteks eksekusi CUDA.
   Untuk skrip SigLIP/SAM, variabel `BDC_SOURCE_ZIP` menerima arsip resmi lokal;
   penyimpanan Hub bersifat privat dan memerlukan akses pemilik. Parameter
   tetap dalam skrip merupakan bagian protokol asli. Tidak ada kredensial tertanam.
3. Bentuk graf global dengan `build_multiscale_collection_graph_20260928.py`
   (`--dino-dir`, `--siglip-dir`, `--manifest`, `--image-root`).
   Graf region, penyaring objek, sensus lengkap, dan konsensus memakai skrip
   `cloud_region_family_graph`, `build_region_object_filter`,
   `build_region_family_census`, dan `refine_region_support_consensus_r3`.
   Berkas berakhiran 20260928 adalah implementasi yang dipakai.
4. Evaluasi korespondensi memakai `evaluate_multiview_region_reranker_20260928.py`.
   Target semantik hanya untuk evaluasi; prediksi berasal dari fitur.
   `evaluate_exposure_state_holdout_20260928.py --help` menjelaskan masukan
   fitur, label pengembangan, label evaluasi beku, keluarga, serta pHash.
   `analyze_within_community_exposure_20260928.py` menjalankan kontrol komunitas.
5. Ulangi perbandingan jadwal dari fitur DINOv3 yang sama:

```sh
python scripts/evaluate_inspection_planner_20260928.py --features cache/dinov3/features.npy --feature-index cache/dinov3/index.csv
```

6. Visual analitis dengan foto dan mask asli:

```sh
python scripts/figure_inspection_atlas_20260928.py --image-root /path/to/BDC/train --proposals /path/to/proposals.jsonl.gz --output outputs/figures
```

## Pembanding RGB–radiograf

Manifest 842 citra dan kunci 421 pasangan terdapat dalam
`experiments/sensor_bridge_20260927`. Data dan lisensinya:
https://doi.org/10.5281/zenodo.18022530.
Gunakan empat arsip RGB/HQ train/test dan checkpoint DINOv2 yang hash-nya cocok.

```sh
python scripts/extract_xbat_frozen.py --archives /path/to/archives --manifest experiments/sensor_bridge_20260927/xbat_images_with_crops.json --checkpoint /path/to/model.safetensors --output cache/xbat
python scripts/test_xbat_paired_view_mapping_20260927.py --feature-dir cache/xbat --output outputs/xbat
```

Skrip ekstraksi portabel mempertahankan mekanisme preprocessing asli; port ini
telah diperiksa sintaksnya, tetapi ekstraksi GPU tidak diulang. Hasil tersimpan
berasal dari bank asli. Pembagian 330/91 mengikuti spesimen; set uji telah
dieksplorasi sebelumnya. Interpretasi dibatasi pada korespondensi fitur.

## Ruang lingkup

## Pemeriksaan cakupan lintas komunitas

```sh
python scripts/evaluate_inspection_context_coverage.py
```

Perhitungan memakai urutan inspeksi tersimpan dan katalog yang sama. Pada 50
foto, 25 keluarga melintasi komunitas untuk urutan discovery, dibandingkan lima
pada wakil global. `pair_review_manifest.json` menetapkan 25 pasangan secara
deterministik; `pair_adjudication.csv` menyimpan interpretasi visual setelah
pemeriksaan foto dan mask. Sepuluh pasangan didukung, lima ambigu, dan sepuluh
ditolak. Label ini tidak dipakai untuk membuat urutan. Hasil tersebut bukan
benchmark ketepatan semantik antarpenjadwal, karena pasangan pembanding tidak
diadjudikasi dalam pemeriksaan lanjutan ini.

## Batas interpretasi

Eksperimen `context_pair_selection_r1` mempertahankan 50 foto dan 25 keluarga
yang sama. `python scripts/evaluate_context_pair_selection.py` menggunakan
84 baris fitur tersimpan untuk menilai 81 pasangan lintas komunitas. Berkas
seleksi mencakup aturan pertama dijumpai, DINO foreground terdekat, dan
kesepakatan peringkat tiga pandangan. Lokasi bank privat dibaca dari `.local`
seperti eksperimen ekstraksi lainnya; tidak diperlukan ekstraksi ulang.
`python scripts/summarize_context_pair_selection.py` menghitung ulang hasil
dari 37 pasangan unik yang seluruhnya diperiksa melalui foto asli dan mask.
Ketiganya mendukung 10/25 pasangan pada tingkat pola visual umum. Aturan baru
tidak digunakan sebagai validasi otomatis. `verify_experiments.py` memeriksa
kelayakan pasangan, peringkat, pilihan, serta penyebut dan hitungan hasil dari
berkas tersimpan tanpa bank fitur besar. Figur kondisi memakai pasangan P05,
yang menampilkan televisi dan perangkat genggam dengan pola retak terlihat.

Hasil bersifat transduktif pada koleksi BDC. Label visual merupakan adjudikasi
satu model multimodal. Data dan cache besar harus disediakan secara privat
untuk menjalankan ulang ekstraksi; verifikasi hasil dan aplikasi dapat berjalan
dengan data turunan yang disertakan. Tidak ada ukuran waktu operator, dampak
daur ulang lapangan, atau generalisasi foto baru. Semua model mengikuti lisensi
upstream; SAM diatribusikan kepada Meta dan sumber checkpoint mirror dicatat.

## Kontrol gaya foto dan adegan berulang

`python scripts/evaluate_exposure_style_control.py` menghitung ulang AUROC dan bootstrap berpasangan dari 48 prediksi beku. Label gaya dan pengelompokan adegan tersimpan pada `experiments/final_study_20260928/exposure_style_control_r1/`; tidak ada pelatihan ulang. Empat adegan masing-masing mempunyai tiga turunan foto, sehingga tersisa 40 wakil. Pemeriksaan gaya menghasilkan 30 wakil beradegan; pengeluaran satu kemungkinan kemiripan sumber yang belum pasti menghasilkan 29. Penilaian visual bersifat adaptif oleh satu pemeriksa berbasis model dan tidak membuktikan independensi sumber.

## Umpan balik sesi inspeksi

`python scripts/evaluate_inspection_feedback.py` membandingkan dua skenario tercatat pada batch lima foto yang sama: seluruhnya dicatat, atau foto kedua dikeluarkan. Riwayat cakupan menjadi 19 dan 16 keluarga; kedua skenario memilih lima foto berikutnya yang sama, dengan cakupan gabungan terencana 29 keluarga. Hasil ini memeriksa pembaruan riwayat dan pengecualian, bukan perubahan urutan yang dipaksakan atau manfaat operator. Seluruh state dibuat pada basis data temporer terpisah dari sesi aplikasi.


## Dekomposisi cakupan keluarga

`python scripts/evaluate_inspection_coverage_decomposition.py` menguraikan urutan
pilihan foto yang telah dibekukan menurut kestabilan partisi, frekuensi keluarga,
dan risiko latar. Pada 50 foto, keluarga stabil yang tercakup berjumlah 59 untuk
pemilihan adaptif, 27 untuk wakil global, dan 42 untuk bobot statis. Analisis ini
bersifat deskriptif pada koleksi yang sama; kestabilan bukan ketepatan semantik.
Seluruh strata disimpan, termasuk kelompok jarang dan risiko latar tinggi.

## Ekspor hasil inspeksi

`python scripts/verify_inspection_export.py` memeriksa ekspor dua batch dari
katalog lengkap pada basis data temporer. Sepuluh foto dan 55 rekaman region
dicocokkan dengan lokasi sumber, SHA-256, bounding box, dan hash mask. Status
foto yang belum diperiksa tetap `pending`; hasil yang dikeluarkan dan catatan
foto pada batch sebelumnya tetap tersedia setelah sesi dibuka kembali.
Ini pengujian fungsi ekspor dan persistensi, bukan validasi operator.
