"""Uğurcan DOĞAN — Fashion-MNIST CNN ödevi."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="CNN katmanları ve Fashion-MNIST deneyleri")
    parser.add_argument("--layers-only", action="store_true", help="Yalnızca katman örneklerini çalıştır")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "sonuclar",
                        help="Çıktı klasörü (mevcut ölçümleri korumak için farklı klasör seçin)")
    args = parser.parse_args()
    from kodlar.katmanlar import incele
    sonuc = incele(args.output_dir)
    if not args.layers_only:
        from kodlar.egitim import deneyleri_calistir
        deneyleri_calistir(sonuc, args.output_dir)


if __name__ == "__main__":
    main()
