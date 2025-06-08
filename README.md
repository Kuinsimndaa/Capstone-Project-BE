# Capstone Project - Stress Detection API

API ini memungkinkan pengguna untuk melakukan deteksi tingkat stres berdasarkan analisis teks menggunakan model LSTM yang telah dilatih.

---

## Endpoint

```
POST https://web-production-8699.up.railway.app/predict
```

---

## Request

### Method

```
POST
```

### Headers

| Key          | Value            |
| ------------ | ---------------- |
| Content-Type | application/json |

### Body (JSON)

```json
{
  "text": "aku capek banget hari ini bener-bener stres karena kerjaan numpuk"
}
```

---

## Response

### Contoh Response `200 OK`

```json
{
  "prediction": "Negative",
  "stress_percent": 94.3
}
```

### Keterangan:

| Field            | Tipe     | Deskripsi                                     |
| ---------------- | -------- | --------------------------------------------- |
| `prediction`     | `string` | Hasil klasifikasi: `Positive` atau `Negative` |
| `stress_percent` | `float`  | Persentase tingkat stres (0 - 100)            |

---

## Error Handling

### Contoh Response jika input kosong

```json
{
  "status": "error",
  "message": "No text provided"
}
```

---

## Contoh Testing via Postman

- Method: `POST`
- URL: `https://web-production-8699.up.railway.app/predict`
- Body:

```json
{
  "text": "saya stres dengan tugas kuliah"
}
```

---

## Catatan Tambahan

- Model ini hanya mendukung input dalam Bahasa Indonesia.
- Hasil berupa `Negative` menandakan kemungkinan stres tinggi.
- Nilai `stress_percent` tidak mutlak, namun merupakan skor dari model LSTM hasil pembelajaran dataset.
