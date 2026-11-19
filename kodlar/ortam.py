"""ServBay ile uyumlu çalışma yolları ve TensorFlow ortamı."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "sonuclar"
os.environ.setdefault("KERAS_HOME", str(ROOT / ".keras"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

plt.rcParams["font.family"] = "Times New Roman"
# Modüller model oluşturmadan önce iş parçacığı sınırlarını ayarlar.
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)
