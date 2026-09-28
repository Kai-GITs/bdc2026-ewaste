# Anatomi Laten Limbah Elektronik

Kode eksperimen dan aplikasi inspeksi SATRIA DATA 2026, kelompok **SD2026040000363**.
Nuril Izza Ahmady; Kalfin Jefwin Setiawan Gultom.

## Analisis

Graf citra utuh memetakan 3.931 foto kanonik ke 15 komunitas. Graf lokal
menghasilkan 129 keluarga region, termasuk 67 keluarga lintas komunitas.
Partisi otomatis dipertahankan terpisah dari penamaan dan adjudikasi visual.
Probe status perakitan membandingkan citra utuh dengan ringkasan keluarga.
Penyusun batch mengoptimalkan tambahan cakupan keluarga pada anggaran inspeksi
yang sama: 50 foto mencakup 77 keluarga, dibandingkan 36 melalui wakil komunitas.
Ukuran ini menilai eksplorasi, bukan ketepatan semantik atau hasil pemilahan fisik.

## Mulai

```sh
python -m pip install -r requirements-release.txt
python scripts/verify_experiments.py
python -m pytest solution/evidence_intake/tests -q
python -m solution.evidence_intake.server --db evidence.sqlite3 --catalog solution/evidence_intake/demo/collection_catalog.json --image-root /path/to/BDC/train --inspection-policy experiments/final_study_20260928/region_family_census_r2/family_census.csv --priorities experiments/final_study_20260928/exposure_state_holdout_r1/blind_test_predictions.csv --port 8879
```

Aplikasi menyimpan sesi inspeksi dalam SQLite. Setiap foto mempunyai hasil
pemeriksaan dan catatan; sesi dapat dilanjutkan setelah browser ditutup. Batch
berikutnya hanya dibuat setelah seluruh foto selesai diperiksa, tanpa mengulang
foto yang sudah ditinjau. Foto yang dikeluarkan tidak menambah cakupan. Versi
sesi mencegah dua tab saling menimpa hasil. Bukti foto/region dan dossier dapat
diekspor dalam JSON/CSV/PDF. Inferensi otomatis untuk foto baru belum dievaluasi.

Lihat [REPRODUCE.md](REPRODUCE.md) untuk tahapan, masukan, dan batas reproduksi.
Identitas bobot dan preprocessing ada di `experiments/model_manifest.json`.
Foto sumber, bobot, cache fitur besar, dan kredensial tidak disertakan.
Repositori ini berisi kode penelitian, data turunan, dan aplikasi.
