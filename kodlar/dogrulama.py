"""Kayıtlı modellerin test sonuçlarını tekrar doğrular. Eğitim yapmaz."""
import json
import numpy as np
from .egitim import veriyi_hazirla
from .ortam import OUT, tf


def dogrula():
    recorded = json.loads((OUT / "sonuclar.json").read_text(encoding="utf-8"))
    (_, _), (images, labels) = tf.keras.datasets.fashion_mnist.load_data()
    if len(labels) != 10000:
        raise ValueError("Test kümesi tam olarak 10.000 görüntü içermelidir.")
    images = veriyi_hazirla(images)
    checks = []
    for record in recorded["experiments"]:
        model = tf.keras.models.load_model(OUT / f"model_{record['filters']}.keras")
        loss, accuracy = model.evaluate(images, labels, batch_size=32, verbose=0)
        probabilities = model.predict(images, batch_size=64, verbose=0)
        if (not np.isfinite(probabilities).all()
                or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6)
                or not ((probabilities >= 0) & (probabilities <= 1)).all()):
            raise ValueError("Model geçerli sınıf olasılıkları üretmedi.")
        if (abs(loss - record["test_loss"]) >= 1e-5
                or abs(accuracy - record["test_accuracy"]) >= 1e-6):
            raise ValueError("Tekrar ölçülen değerler kayıtlı test sonuçlarıyla uyuşmuyor.")
        predicted = probabilities.argmax(axis=1)
        matrix = np.zeros((10, 10), dtype=int)
        np.add.at(matrix, (labels, predicted), 1)
        if matrix.sum() != 10000 or abs(np.trace(matrix) / 10000 - accuracy) >= 1e-6:
            raise ValueError("Karışıklık matrisi ile doğruluk ölçümü uyuşmuyor.")
        checks.append({
            "filters": record["filters"], "samples": len(labels),
            "test_loss": loss, "test_accuracy": accuracy,
            "correct": int(np.trace(matrix)), "confusion_matrix": matrix.tolist(),
            "matches_recorded_results": True,
        })
    report = {
        "description": "Saved-model re-evaluation on the complete, untouched Fashion-MNIST test split. No retraining.",
        "checks": checks,
    }
    (OUT / "dogrulama.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    for check in checks:
        print(f"{check['filters']} filters: {check['correct']}/10000 correct; recorded results match.")
    return report


if __name__ == "__main__":
    dogrula()
