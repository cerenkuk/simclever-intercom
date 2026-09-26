"""
P3.0 Donanım Envanteri Tespiti (CPU / GPU / VRAM / RAM / CUDA)
SimClever Intercom - Paket 3 ASR Seçimi ve Optimizasyonu
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import platform
import psutil
import json

def get_hardware_info():
    info = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "cuda_available": False,
        "gpu_count": 0,
        "gpus": []
    }

    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["gpu_count"] = torch.cuda.device_count()
            for i in range(info["gpu_count"]):
                props = torch.cuda.get_device_properties(i)
                info["gpus"].append({
                    "id": i,
                    "name": props.name,
                    "vram_total_gb": round(props.total_memory / (1024 ** 3), 2),
                    "cuda_capability": f"{props.major}.{props.minor}"
                })
    except ImportError:
        info["torch_installed"] = False
        info["note"] = "PyTorch not installed. CUDA detection skipped."

    return info

def print_summary(info):
    print("=" * 60)
    print("SIMCLEVER INTERCOM - P3.0 DONANIM ENVANTERİ RAPORU")
    print("=" * 60)
    print(f"İşletim Sistemi : {info['system']} {info['release']} ({info['machine']})")
    print(f"İşlemci (CPU)   : {info['processor']}")
    print(f"Çekirdek Sayısı : {info['cpu_count_physical']} Fiziksel / {info['cpu_count_logical']} Mantıksal")
    print(f"Toplam RAM      : {info['total_ram_gb']} GB")
    print(f"CUDA Desteği    : {'EVET' if info['cuda_available'] else 'HAYIR'}")
    
    if info['gpus']:
        for gpu in info['gpus']:
            print(f"GPU #{gpu['id']}          : {gpu['name']} ({gpu['vram_total_gb']} GB VRAM)")
    else:
        print("GPU             : Harici NVIDIA GPU tespit edilmedi / PyTorch CUDA aktif değil.")
    print("=" * 60)

if __name__ == "__main__":
    hw_info = get_hardware_info()
    print_summary(hw_info)
    
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "reports")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "hardware_inventory.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(hw_info, f, ensure_ascii=False, indent=2)
    print(f"Envanter raporu kaydedildi: {out_path}")
