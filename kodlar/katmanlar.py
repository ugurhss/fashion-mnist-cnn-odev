"""Konvolüsyon, Max Pooling ve aktivasyonların küçük örneklerle incelenmesi."""
import json
import platform
from pathlib import Path

import numpy as np
from .ortam import OUT, plt, tf


def katman_hesaplari():
    """Dosya yazmadan yönergedeki örnekleri hesaplar."""
    tf.keras.utils.set_random_seed(42)
    conv = tf.keras.layers.Conv2D(
        16, (3, 3), activation="relu", kernel_initializer="he_normal"
    )
    giris = tf.random.normal((1, 128, 128, 1))
    cikis = conv(giris)
    agirliklar, bias = conv.get_weights()
    matris = np.array(
        [[1, 3, 2, 4], [5, 6, 1, 2], [7, 2, 8, 3], [4, 1, 5, 9]],
        dtype=np.float32,
    )
    pooling = tf.keras.layers.MaxPooling2D((2, 2), strides=2)(
        matris.reshape(1, 4, 4, 1)
    ).numpy()[0, :, :, 0]
    relu = tf.nn.relu([-5., -2., -1., 0., 2., 4., 6.]).numpy()
    softmax = tf.nn.softmax([[2., 1., 0.1]]).numpy()[0]
    sonuc = {
        "conv_input": list(giris.shape),
        "conv_output": list(cikis.shape),
        "weight_shape": list(agirliklar.shape),
        "bias_shape": list(bias.shape),
        "first_filter": agirliklar[:, :, 0, 0].tolist(),
        "pooling": pooling.tolist(),
        "relu": relu.tolist(),
        "softmax": softmax.tolist(),
        "tensorflow": tf.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    return sonuc, agirliklar


def incele(output_dir=OUT):
    """Katman sonuçlarını ve ortak renk ölçekli 16 filtre görselini kaydeder."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sonuc, agirliklar = katman_hesaplari()
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(
            agirliklar[:, :, 0, i], cmap="coolwarm",
            vmin=agirliklar.min(), vmax=agirliklar.max(),
        )
        ax.set_title(f"Filtre {i + 1}")
        ax.axis("off")
    fig.suptitle("16 filtrenin başlangıç ağırlıkları")
    fig.tight_layout()
    fig.savefig(output_dir / "filtreler.png", dpi=180)
    plt.close(fig)
    np.save(output_dir / "filtre_agirliklari.npy", agirliklar)
    (output_dir / "katman_ciktilari.txt").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(sonuc, ensure_ascii=False, indent=2))
    return sonuc
