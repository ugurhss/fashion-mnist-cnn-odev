"""Matematik, veri işleme ve model davranışı için çevrimdışı kontroller."""
import unittest
import numpy as np
from kodlar.egitim import model_olustur, veriyi_hazirla
from kodlar.katmanlar import katman_hesaplari
from kodlar.ortam import tf


class LayerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result, cls.weights = katman_hesaplari()

    def test_convolution_dimensions(self):
        self.assertEqual(self.result['conv_output'], [1, 126, 126, 16])
        self.assertEqual(self.weights.shape, (3, 3, 1, 16))
        self.assertEqual(self.result['bias_shape'], [16])
        self.assertTrue(np.isfinite(self.weights).all())
        self.assertGreater(float(self.weights.std()), 0)

    def test_pooling_matches_hand_calculation(self):
        np.testing.assert_array_equal(self.result['pooling'], [[6, 4], [7, 9]])

    def test_relu_preserves_positive_values(self):
        np.testing.assert_array_equal(self.result['relu'], [0, 0, 0, 0, 2, 4, 6])

    def test_softmax_matches_independent_numpy_formula(self):
        scores = np.array([2., 1., .1], dtype=np.float64)
        expected = np.exp(scores - scores.max())
        expected /= expected.sum()
        np.testing.assert_allclose(self.result['softmax'], expected, rtol=1e-6)
        self.assertAlmostEqual(sum(self.result['softmax']), 1, places=6)
        self.assertEqual(int(np.argmax(self.result['softmax'])), 0)


class DataTests(unittest.TestCase):
    def test_normalization_and_channel_axis(self):
        raw = np.zeros((2, 28, 28), dtype=np.uint8)
        raw[1] = 255
        result = veriyi_hazirla(raw)
        self.assertEqual(result.shape, (2, 28, 28, 1))
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(float(result.min()), 0)
        self.assertEqual(float(result.max()), 1)
        self.assertEqual(int(raw[1, 0, 0]), 255)

    def test_invalid_images_are_rejected(self):
        for raw in [np.zeros((28, 28), dtype=np.uint8),
                    np.zeros((1, 27, 28), dtype=np.uint8),
                    np.zeros((1, 28, 28), dtype=np.float32)]:
            with self.subTest(shape=raw.shape, dtype=raw.dtype):
                with self.assertRaises(ValueError):
                    veriyi_hazirla(raw)


class ModelTests(unittest.TestCase):
    def test_architecture_and_parameter_counts(self):
        for filters, expected in [(8, 27562), (16, 28794)]:
            with self.subTest(filters=filters):
                model = model_olustur(filters)
                self.assertEqual(model.count_params(), expected)
                self.assertEqual([tuple(layer.output.shape[1:]) for layer in model.layers],
                                 [(26, 26, filters), (13, 13, filters), (11, 11, 16),
                                  (5, 5, 16), (400,), (64,), (10,)])
                self.assertEqual(model.loss, 'sparse_categorical_crossentropy')
                self.assertIsInstance(model.optimizer, tf.keras.optimizers.Adam)

    def test_invalid_filter_count(self):
        for value in [0, -1, 7, 32, True, None, 8.0, "8"]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    model_olustur(value)

    def test_forward_pass_is_a_probability_distribution(self):
        model = model_olustur(8)
        output = model(np.ones((3, 28, 28, 1), dtype=np.float32)).numpy()
        self.assertEqual(output.shape, (3, 10))
        self.assertTrue(np.isfinite(output).all())
        self.assertTrue(((output >= 0) & (output <= 1)).all())
        np.testing.assert_allclose(output.sum(axis=1), 1, atol=1e-6)

    def test_training_step_updates_weights(self):
        tf.keras.utils.set_random_seed(42)
        model = model_olustur(8)
        before = [w.copy() for w in model.get_weights()]
        x = np.random.default_rng(42).random((4, 28, 28, 1), dtype=np.float32)
        metrics = model.train_on_batch(x, np.array([0, 1, 2, 3]), return_dict=True)
        self.assertTrue(np.isfinite(metrics['loss']))
        self.assertTrue(any(not np.array_equal(x, y)
                            for x, y in zip(before, model.get_weights())))


if __name__ == '__main__':
    unittest.main()
