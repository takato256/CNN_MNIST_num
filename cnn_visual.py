import tensorflow as tf
import tensorflow.keras.layers as kl
import numpy as np
import feature_visual
import filter_visual
import argparse as arg
import os
import streamlit as st

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TFメッセージ非表示
st.set_option('deprecation.showPyplotGlobalUse', False) # streamlit上での警告を非表示


# CNN
class CNN(tf.keras.Model):
    def __init__(self, n_out, input_shape):
        super().__init__()

        self.conv1 = kl.Conv2D(16, 4, activation='relu', input_shape=input_shape)
        self.conv2 = kl.Conv2D(32, 4, activation='relu')
        self.conv3 = kl.Conv2D(64, 4, activation='relu')

        self.mp1 = kl.MaxPool2D((2, 2), padding='same')
        self.mp2 = kl.MaxPool2D((2, 2), padding='same')
        self.mp3 = kl.MaxPool2D((2, 2), padding='same')

        self.flt = kl.Flatten()

        self.link = kl.Dense(1024, activation='relu')
        self.link_class = kl.Dense(n_out, activation='softmax')

    def call(self, x):
        h1 = self.mp1(self.conv1(x))
        h2 = self.mp2(self.conv2(h1))
        h3 = self.mp3(self.conv3(h2))
        
        h4 = self.link(self.flt(h3))

        return self.link_class(h4)


# 学習
class trainer(object):
    def __init__(self, n_out, input_shape):
        self.model = CNN(n_out, input_shape)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                            metrics=['accuracy'])

    def train(self, train_img, train_lab, batch_size, epochs, input_shape, test_img):
        # 学習
        with st.spinner("学習中‥"):
            self.model.fit(train_img, train_lab, batch_size=batch_size, epochs=epochs)


        # 特徴マップ可視化
        with st.spinner("特徴マップを読み込み中‥"):
            feature_visual.feature_vi(self.model, input_shape, train_img)
        
        # フィルタ可視化
        with st.spinner("フィルタを読み込み中‥"):
            filter_visual.filter_vi(self.model)


def main():
    st.markdown("## 畳み込みニューラルネットワーク(CNN)")
    st.markdown("このサイトでは数字を学習させることができます")
    st.markdown("学習開始ボタンを押す度に、違う数字が表示されます")
    with st.sidebar:
        st.markdown("※注意")
        st.markdown("学習中は値を変更しないでください")
        batch_size = st.number_input("ミニバッチサイズ", 1, 1000, 256, step=1)
        epoch_num = st.number_input("エポック数", 1, 100, 3, step=1)
        st.markdown("ミニバッチサイズとは？")
        st.markdown("""データセットをいくつかのグループに分けた、各グループの数のことです。
                    例えば、1000件のデータセットを200件ずつに分けた場合、ミニバッチサイズは200になります。
                    バッチサイズが大きくなると、特徴量が平均化され、特徴量が失われる可能性があります。""")
        st.markdown("エポック数とは？")
        st.markdown("""エポック数は学習回数とも言われます。
                    エポック数が少ないと適切に学習する前に学習が"終わってしまいます。
                    逆に大きすぎると、過学習を起こしてしまいます。""")
        
    if st.button("学習開始"):
        # コマンドラインオプション作成
        parser = arg.ArgumentParser(description='CNN Feature-map & Filter Visualization')
        parser.add_argument('--batch_size', '-b', type=int, default=batch_size,
                            help='ミニバッチサイズの指定(デフォルト値=256)')
        parser.add_argument('--epoch', '-e', type=int, default=epoch_num,
                            help='学習回数の指定(デフォルト値=10)')
        args = parser.parse_args()

        # データセット取得、前処理
        with st.spinner("データを取得・前処理中"):
            (train_img, train_lab), (test_img, _) = tf.keras.datasets.mnist.load_data()
            train_img = tf.convert_to_tensor(train_img, np.float32)
            train_img /= 255
            train_img = train_img[:, :, :, np.newaxis]

            test_img = tf.convert_to_tensor(test_img, np.float32)
            test_img /= 255
            test_img = train_img[:, :, :, np.newaxis]

        # 学習開始
        input_shape = (28, 28, 1)
    
        Trainer = trainer(10, input_shape)
        Trainer.train(train_img, train_lab, batch_size=args.batch_size,
                        epochs=args.epoch, input_shape=input_shape, test_img=test_img)


if __name__ == '__main__':
    main()
