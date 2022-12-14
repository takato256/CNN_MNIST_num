import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import streamlit as st

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TFメッセージ非表示


# 特徴マップ可視化
def feature_vi(model, input_shape, test_img):
    # モデル再構築
    x = tf.keras.Input(shape=input_shape)
    model_vi = tf.keras.Model(inputs=x, outputs=model.call(x))

    # ネットワーク構成出力
    model_vi.summary()
    print("")

    # レイヤー情報を取得
    feature_vi = []
    feature_vi.append(model_vi.get_layer('input_1'))
    feature_vi.append(model_vi.get_layer('conv2d'))
    feature_vi.append(model_vi.get_layer('max_pooling2d'))
    feature_vi.append(model_vi.get_layer('conv2d_1'))
    feature_vi.append(model_vi.get_layer('max_pooling2d_1'))

    # データランダム抽出
    idx = int(np.random.randint(0, len(test_img), 1))
    img = test_img[idx]
    img = img[None, :, :, :]

    for i in range(len(feature_vi) - 1):
        # 特徴マップ取得
        feature_model = tf.keras.Model(inputs=feature_vi[0].input, outputs=feature_vi[i + 1].output)
        feature_map = feature_model.predict(img)
        feature_map = feature_map[0]
        feature = feature_map.shape[2]
        
        # 層の名前を表示
        layer_names = ["畳み込み層 1", "プーリング層 1", "畳み込み層 2", "プーリング層 2"]
        st.markdown("### {}".format(layer_names[i]))

        # 出力
        for j in range(feature):
            plt.subplots_adjust(wspace=0.4, hspace=0.8)
            plt.subplot(int(feature / 6 + 1), 6, j + 1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(feature_map[:, :, j])
        st.pyplot()
