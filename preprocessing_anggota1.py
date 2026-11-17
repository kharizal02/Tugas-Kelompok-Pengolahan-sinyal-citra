import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
from scipy.io import wavfile


# =========================
# KONFIGURASI
# =========================

FS_TARGET = 16000
BIT_DEPTH = 16

POWER_MS = 10
NOISE_MS = 500
THRESHOLD_K = 5.0

FRAME_MS = 25
OVERLAP_MS = 10

FOLDER_WAV = Path("data_wav")
FOLDER_HASIL = Path("hasil_preprocessing")


# =========================
# BACA WAV
# =========================

def baca_wav(path):
    fs, data = wavfile.read(path)

    if fs != FS_TARGET:
        raise ValueError(f"{path.name}: sample rate harus 16000 Hz")

    if data.dtype != np.int16:
        raise ValueError(f"{path.name}: harus PCM 16-bit")

    if data.ndim != 1:
        raise ValueError(f"{path.name}: harus mono")

    sinyal = data.astype(np.float64) / 32768.0

    return fs, sinyal


# =========================
# LEVEL CLIPPING
# =========================

def level_clipping(sinyal, fs):
    panjang_blok = round(fs * POWER_MS / 1000)
    jumlah_blok = int(np.ceil(len(sinyal) / panjang_blok))

    power = []

    for i in range(jumlah_blok):
        awal = i * panjang_blok
        akhir = min(awal + panjang_blok, len(sinyal))

        blok = sinyal[awal:akhir]
        power.append(np.mean(blok ** 2))

    power = np.array(power)

    jumlah_noise = round(NOISE_MS / POWER_MS)
    noise = power[:jumlah_noise]

    threshold_noise = (
    np.mean(noise)
    + THRESHOLD_K * np.std(noise)
    )

    threshold_signal = (
        0.01 * np.max(power)
    )

    threshold = max(
        threshold_noise,
        threshold_signal
    )
    print(f"Threshold       : {threshold:.8e}")

    aktif = np.where(power > threshold)[0]

    if len(aktif) == 0:
        raise ValueError("Suara aktif tidak ditemukan")

    # margin 50 ms sebelum, 100 ms sesudah
    blok_awal = max(0, aktif[0] - 5)
    blok_akhir = min(jumlah_blok - 1, aktif[-1] + 10)

    sample_awal = blok_awal * panjang_blok
    sample_akhir = min(
        len(sinyal),
        (blok_akhir + 1) * panjang_blok
    )

    hasil = sinyal[sample_awal:sample_akhir]

    waktu_power = (
        np.arange(len(power))
        * panjang_blok / fs
    )

    return hasil, power, waktu_power, threshold


# =========================
# FRAME BLOCKING
# =========================

def frame_blocking(sinyal, fs):
    panjang_frame = round(fs * FRAME_MS / 1000)
    overlap = round(fs * OVERLAP_MS / 1000)

    shift = panjang_frame - overlap

    jumlah_frame = int(
        np.ceil(
            max(0, len(sinyal) - panjang_frame)
            / shift
        )
    ) + 1

    panjang_total = (
        (jumlah_frame - 1) * shift
        + panjang_frame
    )

    padding = panjang_total - len(sinyal)

    if padding > 0:
        sinyal = np.pad(sinyal, (0, padding))

    frames = []

    for i in range(jumlah_frame):
        awal = i * shift
        akhir = awal + panjang_frame

        frames.append(sinyal[awal:akhir])

    return np.array(frames), panjang_frame, overlap, shift


# =========================
# GRAFIK
# =========================

def simpan_grafik(nama, sinyal, fs, clipped, power, waktu_power, threshold, frames):
    waktu = np.arange(len(sinyal)) / fs

    plt.figure(figsize=(10, 4))
    plt.plot(waktu, sinyal)
    plt.title(f"Sinyal Asli - {nama}")
    plt.xlabel("Waktu (detik)")
    plt.ylabel("Amplitudo")
    plt.grid()
    plt.tight_layout()
    plt.savefig(FOLDER_HASIL / f"{nama}_01_asli.png")
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(waktu_power, power)
    plt.axhline(threshold, linestyle="--")
    plt.title(f"Power dan Threshold - {nama}")
    plt.xlabel("Waktu (detik)")
    plt.ylabel("Power")
    plt.grid()
    plt.tight_layout()
    plt.savefig(FOLDER_HASIL / f"{nama}_02_power.png")
    plt.close()

    waktu_clip = np.arange(len(clipped)) / fs

    plt.figure(figsize=(10, 4))
    plt.plot(waktu_clip, clipped)
    plt.title(f"Level Clipping - {nama}")
    plt.xlabel("Waktu (detik)")
    plt.ylabel("Amplitudo")
    plt.grid()
    plt.tight_layout()
    plt.savefig(FOLDER_HASIL / f"{nama}_03_clipping.png")
    plt.close()

    plt.figure(figsize=(10, 4))

    waktu_frame = np.arange(frames.shape[1]) / fs

    for i in range(min(5, len(frames))):
        plt.plot(
            waktu_frame,
            frames[i],
            label=f"Frame {i + 1}"
        )

    plt.title(f"Frame Blocking - {nama}")
    plt.xlabel("Waktu dalam frame (detik)")
    plt.ylabel("Amplitudo")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(FOLDER_HASIL / f"{nama}_04_frame.png")
    plt.close()


# =========================
# PROSES
# =========================

def proses(path):
    nama = path.stem

    fs, sinyal = baca_wav(path)

    clipped, power, waktu_power, threshold = level_clipping(
        sinyal,
        fs
    )

    frames, panjang_frame, overlap, shift = frame_blocking(
        clipped,
        fs
    )

    simpan_grafik(
        nama,
        sinyal,
        fs,
        clipped,
        power,
        waktu_power,
        threshold,
        frames
    )

    pd.DataFrame({
        "amplitudo": sinyal
    }).to_excel(
        FOLDER_HASIL / f"{nama}_sinyal_asli.xlsx",
        index=False
    )

    pd.DataFrame({
        "amplitudo": clipped
    }).to_excel(
        FOLDER_HASIL / f"{nama}_clipping.xlsx",
        index=False
    )

    pd.DataFrame(
        frames.T,
        columns=[
            f"Frame_{i+1}"
            for i in range(len(frames))
        ]
    ).to_excel(
        FOLDER_HASIL / f"{nama}_frame.xlsx",
        index=False
    )

    np.savez(
        FOLDER_HASIL / f"{nama}_frames.npz",
        frames=frames,
        fs=fs
    )

    print(f"\n{path.name}")
    print(f"Durasi asli     : {len(sinyal)/fs:.3f} s")
    print(f"Durasi clipping : {len(clipped)/fs:.3f} s")
    print(f"Threshold       : {threshold:.8e}")
    print(f"Jumlah frame    : {len(frames)}")
    print(f"Panjang frame   : {panjang_frame} sample")
    print(f"Overlap         : {overlap} sample")
    print(f"Frame shift     : {shift} sample")


# =========================
# MAIN
# =========================

FOLDER_WAV.mkdir(exist_ok=True)
FOLDER_HASIL.mkdir(exist_ok=True)

files = sorted(FOLDER_WAV.glob("*.wav"))

print(f"Jumlah WAV: {len(files)}")

for file in files:
    try:
        proses(file)

    except Exception as e:
        print(f"GAGAL {file.name}: {e}")

print("\nSelesai.")