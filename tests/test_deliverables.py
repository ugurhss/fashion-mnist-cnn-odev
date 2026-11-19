"""Kayıtlı ölçümler, model dosyaları ve teslim belgelerinin tutarlılığı."""
import csv
import json
from pathlib import Path
import unittest
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
from kodlar.ortam import ROOT, tf


class DeliverableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = ROOT / 'sonuclar'
        cls.data = json.loads((cls.out / 'sonuclar.json').read_text(encoding='utf-8'))

    def test_experiments_and_epoch_history(self):
        self.assertEqual([r['filters'] for r in self.data['experiments']], [8, 16])
        for row in self.data['experiments']:
            for key in ['accuracy', 'val_accuracy', 'loss', 'val_loss']:
                values = np.array(row['history'][key])
                self.assertEqual(len(values), 5)
                self.assertTrue(np.isfinite(values).all())
                self.assertTrue((values >= 0).all())
                if 'accuracy' in key:
                    self.assertTrue((values <= 1).all())
            self.assertEqual(row['train_accuracy'], row['history']['accuracy'][-1])
            self.assertEqual(row['validation_accuracy'], row['history']['val_accuracy'][-1])
            self.assertTrue(0 <= row['test_accuracy'] <= 1)
            self.assertGreater(row['seconds'], 0)
            self.assertGreaterEqual(row['test_loss'], 0)

    def test_csv_and_json_agree(self):
        with (self.out / 'karsilastirma.csv').open(newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 2)
        for csv_row, record in zip(rows, self.data['experiments']):
            for key, value in csv_row.items():
                self.assertAlmostEqual(float(value), record[key], places=9)

    def test_initial_filters_match_recorded_values(self):
        weights = np.load(self.out / 'filtre_agirliklari.npy')
        self.assertEqual(weights.shape, (3, 3, 1, 16))
        np.testing.assert_allclose(weights[:, :, 0, 0], self.data['first_filter'])

    def test_saved_models_match_architecture(self):
        for record in self.data['experiments']:
            model = tf.keras.models.load_model(self.out / f"model_{record['filters']}.keras")
            self.assertEqual(model.count_params(), record['parameters'])
            self.assertEqual(model.layers[0].filters, record['filters'])
            result = model(np.zeros((1, 28, 28, 1), dtype=np.float32)).numpy()
            self.assertTrue(np.isfinite(result).all())
            np.testing.assert_allclose(result.sum(), 1, atol=1e-6)

    def test_full_evaluation_evidence(self):
        evidence = json.loads((self.out / 'dogrulama.json').read_text(encoding='utf-8'))
        self.assertEqual(len(evidence['checks']), 2)
        for check, record in zip(evidence['checks'], self.data['experiments']):
            matrix = np.asarray(check['confusion_matrix'])
            self.assertEqual(matrix.shape, (10, 10))
            self.assertTrue((matrix >= 0).all())
            self.assertEqual(int(matrix.sum()), 10000)
            self.assertEqual(int(np.trace(matrix)), check['correct'])
            self.assertEqual(check['filters'], record['filters'])
            self.assertAlmostEqual(np.trace(matrix) / 10000, record['test_accuracy'], places=6)
            self.assertAlmostEqual(check['test_loss'], record['test_loss'], places=5)

    def test_office_files_and_font(self):
        for filename in ['Ugurcan_DOGAN_Rapor.docx', 'Ugurcan_DOGAN_Sunum.pptx']:
            with self.subTest(filename=filename), ZipFile(ROOT / filename) as z:
                self.assertIsNone(z.testzip())
                for name in z.namelist():
                    if not name.endswith('.xml'):
                        continue
                    tree = ET.fromstring(z.read(name))
                    if name.startswith(('ppt/slides/slide', 'ppt/charts/chart')):
                        for node in tree.iter():
                            if 'typeface' in node.attrib:
                                self.assertEqual(node.attrib['typeface'], 'Times New Roman')
                    if name in ['word/styles.xml', 'word/document.xml', 'word/footer1.xml']:
                        for node in tree.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}rFonts'):
                            for key in ['ascii', 'hAnsi']:
                                value = node.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}' + key)
                                if value:
                                    self.assertEqual(value, 'Times New Roman')
                if filename.endswith('.pptx'):
                    slides = [n for n in z.namelist()
                              if n.startswith('ppt/slides/slide') and n.endswith('.xml')]
                    self.assertEqual(len(slides), 16)

    def test_graphics_are_readable_images(self):
        from PIL import Image
        for filename in ['filtreler.png', 'egitim_8.png', 'egitim_16.png']:
            with Image.open(self.out / filename) as image:
                image.verify()
            with Image.open(self.out / filename) as image:
                self.assertGreaterEqual(image.width, 1000)
                self.assertGreaterEqual(image.height, 700)


if __name__ == '__main__':
    unittest.main()
