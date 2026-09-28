# Telusur bukti region E-Waste

Aplikasi lokal untuk katalog 3.931 foto kanonik, 6.093 region, dan 129 keluarga.
Pengguna memilih region, membandingkan kandidat pada foto lain, mencatat keputusan,
menyimpan revisi dossier dalam SQLite, dan mengekspor JSON/CSV/PDF.

```sh
python -m solution.evidence_intake.server --db evidence.sqlite3 --catalog solution/evidence_intake/demo/collection_catalog.json --image-root /path/to/BDC/train --priorities experiments/final_study_20260928/exposure_state_holdout_r1/blind_test_predictions.csv --port 8878
```

Buka http://127.0.0.1:8878/. Opsi prioritas menampilkan 48 foto evaluasi menurut
skor konteks perakitan. Label tidak memengaruhi urutan dan tidak diberikan melalui
API prioritas. Skor merupakan skor peringkat, bukan probabilitas terkalibrasi.
Pencocokan unggahan menggunakan SHA-256 bank koleksi; tidak ada inferensi foto baru.
Berkas frozen_batch.json adalah fixture pengujian lama, bukan keseluruhan katalog.

Pada mode inspeksi, hasil sesi dapat diunduh sebagai CSV atau JSON. Ekspor memuat
foto dari batch terdahulu dan batch aktif, lokasi serta hash sumber, keputusan,
catatan, waktu pemeriksaan, dan status foto yang masih menunggu pemeriksaan.
JSON juga menyertakan ID region, bounding box, dan hash mask dari katalog.
Keanggotaan keluarga merupakan hasil model; menyelesaikan pemeriksaan foto tidak
otomatis mengonfirmasi identitas seluruh komponennya.

```sh
python -m pytest solution/evidence_intake/tests -q
python scripts/test_evidence_intake_workflow_20260928.py --base-url http://127.0.0.1:8878 --image-root /path/to/BDC/train --output-dir /path/to/test-output
```

Status keputusan dapat ditetapkan oleh pengguna setelah memeriksa citra. Status
aplikasi tidak membuktikan adanya penilaian ahli dalam eksperimen. Evaluasi paper
menggunakan adjudikasi satu agen dan kohort koleksi yang telah dianalisis.
