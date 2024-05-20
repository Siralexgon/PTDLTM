import keras_nlp
from keras_nlp.api.models import DebertaV3Preprocessor
from keras_nlp.api.models import DebertaV3Classifier
import keras
import keras.api.backend as K
import tensorflow as tf

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split

# NOTE: Tham khảo : https://www.kaggle.com/code/awsaf49/aes-2-0-kerasnlp-starter#%F0%9F%8D%BD%EF%B8%8F-%7C-Preprocessing
# NOTE: đọc hết TODO

# TODO: đổi path file csv
data_path = ".csv"
# Sử dụng model DeBERTa-V3-Extra-Small-English
model_preset = "deberta_v3_extra_small_en"
max_epochs = 10
batch_size = 8
shufle = 1000
lr = 0.0000003


# Label giá trị trị từ 1-6 (cột score) là dữ liệu categorical.
# Nhưng để làm bài này, label phải là dữ liệu ordinary
# Biến đổi label CATEGORICAL SANG ORDINARY bằng cách :

# Encode label thành một vector nhị phân có độ dài 6, bit i bằng 1 nghĩa là đã thỏa yêu cầu i
# Các giá trị trong vector CHO BIẾT XÁC SUẤT VỊ TRÍ i THỎA YÊU CẦU i.
# Từ đó có thể sử dụng CROSS-ENTROPY để sử dụng hàm lỗi so sánh sự tương đồng giữa 2 phân phối.

# Từ trên ta tạo tính ORDINARY như sau:

# Bài được chấm điểm 1 <= i <= 6 nghĩa là đã thỏa mãn yêu cầu i, VÀ CŨNG THỎA TẤT CẢ CÁC YÊU CẦU TRƯỚC ĐÓ (<= i),
# nên các bit trong vector nhãn sẽ = 1 ở các vị trí <= i, và 0 ở các vị trí còn lại.
# Cách làm này giúp có TÍNH SO SÁNH, vì một khi được điểm j >= i, thì các bit 1 của điểm i cũng = 1 ở trong j


def make_ordinary(y):
    n = y.shape[0]
    z = np.zeros((n, 6), "float32")
    for i in range(n):
        s = y[i]
        z[i, :s] = 1
    return z


# Load dữ liệu từ file csv và tạo tf.data.Dataset cho dữ liệu train và dữ liệu validation
# Dataset load dữ liệu theo lô (và cache để tiết kiệm bộ nhớ trong)
# và hoán vị dữ liệu sau mỗi vòng lặp cho THUẬT TOÁN SGD
def create_dataset():

    df = pd.read_csv(data_path)

    X_train_df, X_val_df, y_train_df, y_val_df = train_test_split(
        df["full_text"], df["score"], test_size=0.2, stratify=df["score"]
    )
    X_train = X_train_df.toList()
    X_val = X_val_df.toList()
    y_train = make_ordinary(y_train_df)
    y_val = make_ordinary(y_val_df)

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .cache()
        .shuffle(shufle, seed=0)
        .batch(batch_size)
    )
    opt = tf.data.Options()
    opt.experimental_deterministic = False
    train_ds = train_ds.with_options(opt)

    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val)).cache().batch(batch_size)
    )

    return train_ds, val_ds


# TODO: Code DebertaV3Classifier.from_preset tự động thực hiện tiền sử lý đầu vào,
# đọc phần Preprocessing để viết lại phần tiền xử lý


# Load model đã được cấu hình sẵn
# Như đã nói trên sử đầù ra là một vector độ dài 6, mỗi vị trí cho biết xác suất thỏa yêu cầu i,
# Đầu ra của mạng sử dụng sigmoid cho ra giá trị 0->1, chuẩn hóa thành xác suất cho mỗi vị trí i.
# Sử dụng hàm Binary Cross-Entropy, vì ta tính xác suất mỗi vị trí (mỗi vị trí đúng (1) đến sai (0)),
# sau đó keras.losses.BinaryCrossentropy tổng hợp Loss tại mỗi vị trí làm Loss chung
# Sử dụng Adam optimizer kết hợp SGD và momentum
def create_DebertaV3_model():
    debertaV3 = DebertaV3Classifier.from_preset(model_preset, num_classes=6)

    inputs = debertaV3.input
    outputs = debertaV3(inputs)

    prob_outputs = keras.layers.Activation("sigmoid")(outputs)
    model = keras.Model(inputs, prob_outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=keras.losses.BinaryCrossentropy(),
    )

    return model


train_ds, val_ds = create_dataset()
model = create_DebertaV3_model()


# TODO: chạy code thử xem có lỗi gì không
# TODO: viết thêm callback sau mỗi vòng lặp để tính độ chính xác/ lỗi sau mỗi vòng lặp

# Train model
model.fit(
    train_ds,
    epochs=max_epochs,
    validation_data=val_ds,
)

# TODO: Viết test cho model đã train
