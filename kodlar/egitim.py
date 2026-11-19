"""Fashion-MNIST üzerinde aynı ayarlarla iki CNN deneyinin yürütülmesi."""
import csv
import json
import time
from pathlib import Path

import numpy as np
from .ortam import OUT, plt, tf


def veriyi_hazirla(images):
    """28×28 uint8 görüntülere kanal ekler ve pikselleri [0, 1] aralığına alır."""
    images = np.asarray(images)
    if images.ndim != 3 or images.shape[1:] != (28, 28):
        raise ValueError("Girdi boyutu (örnek sayısı, 28, 28) olmalıdır.")
    if images.dtype != np.uint8:
        raise ValueError("Ham görüntüler uint8 türünde olmalıdır.")
    return images.astype(np.float32)[..., np.newaxis] / 255.0


def model_olustur(filters):
    """Yönergedeki mimariyi değiştirmeden 8 veya 16 ilk katman filtresi kullanır."""
    if type(filters) is not int or filters not in (8, 16):
        raise ValueError("İlk katman için filtre sayısı 8 veya 16 olmalıdır.")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(filters, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def egitim_grafigi(history, filters, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = range(1, len(history["accuracy"]) + 1)
    for ax, key, title in zip(axes, ("accuracy", "loss"), ("Doğruluk", "Kayıp")):
        ax.plot(epochs, history[key], label="Eğitim")
        ax.plot(epochs, history["val_" + key], label="Doğrulama")
        ax.set(xlabel="Epoch", title=title, xticks=list(epochs))
        ax.legend()
        ax.grid(alpha=0.2)
    fig.suptitle(f"İlk katmanda {filters} filtre")
    fig.tight_layout()
    fig.savefig(output_dir / f"egitim_{filters}.png", dpi=180)
    plt.close(fig)


def deneyleri_calistir(katman_sonuclari, output_dir=OUT):
    """Beş epoch eğitim, test değerlendirmesi ve ölçüm dosyalarını üretir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (xt, yt), (xe, ye) = tf.keras.datasets.fashion_mnist.load_data()
    xt, xe = veriyi_hazirla(xt), veriyi_hazirla(xe)
    results = []
    for filters in (8, 16):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42)
        model = model_olustur(filters)
        with (output_dir / f"model_{filters}.txt").open("w", encoding="utf-8") as f:
            model.summary(print_fn=lambda line: f.write(line + "\n"))
        start = time.perf_counter()
        history = model.fit(
            xt, yt, epochs=5, batch_size=64, validation_split=0.1, verbose=2
        ).history
        seconds = time.perf_counter() - start
        loss, accuracy = model.evaluate(xe, ye, verbose=0)
        results.append({
            "filters": filters,
            "train_accuracy": history["accuracy"][-1],
            "validation_accuracy": history["val_accuracy"][-1],
            "test_accuracy": accuracy,
            "test_loss": loss,
            "seconds": seconds,
            "parameters": model.count_params(),
            "history": history,
        })
        model.save(output_dir / f"model_{filters}.keras")
        egitim_grafigi(history, filters, output_dir)
    (output_dir / "sonuclar.json").write_text(
        json.dumps(dict(experiments=results, **katman_sonuclari),
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fields = ["filters", "train_accuracy", "validation_accuracy", "test_accuracy",
              "test_loss", "seconds", "parameters"]
    with (output_dir / "karsilastirma.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return results
