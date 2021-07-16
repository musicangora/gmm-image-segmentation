# gmm-image-segmentation
混合ガウス分布にギブスサンプリングを用いて画像のセグメンテーションを行う。
 
## Sample Output
|![](sample_outputs/deer.png)|![](sample_outputs/deer_overlay.png)|
|:---:|:---:|
|クラスタリング後|入力画像と出力画像の比較|

## Sample Animation
![](sample_result/animation.gif)

## 使い方
- このリポジトリをクローンして、`./image`下に推論したい画像を追加する
    - 画像のサイズが大きいとメモリが足りなくなる
    - 内部で画像を1/2にリサイズしているが、数百ピクセル程度の画像を推奨
- 表示される指示に従って、拡張子まで含めた画像のファイル名、クラスタ数、試行回数を入力する
- 推論が終了するとK-Meansによるクラスタリング結果とGMMによるクラスタリング結果の比較画像が保存される

```.bash
$ python main.py 
```

```.bash
$ 画像のファイル名を入力してください：deer.jpg

$ クラス数を入力してください：3

$ 最大試行回数を入力してください：15
 iter: 15 [====================>](100％)
```

## Sample Result
### K-Means法によるクラスタリングとギブスサンプリングによるクラスタリングの比較
![](sample_result/k-means_gibbs_comparison.png)

### 初期の分布と推論後の分布の比較
- 点線：初期の分布
- 実線：推論後の分布

|![](sample_result/RGhist.png)|![](sample_result/GBhist.png)|
|:---:|:---:|
|Rチャンネル, Gチャンネルのヒストグラム|Gチャンネル, Bチャンネルのヒストグラム|

|![](sample_result/RGchannels.png)|![](sample_result/GBchannels.png)|
|:---:|:---:|
|x軸：Rチャンネル、y軸：Gチャンネル|x軸：Gチャンネル、y軸：Bチャンネル|

## プログラムの流れ
1. 画像の読み込みと前処理
    - 入力画像のリサイズと画素値を0.0~1.0の範囲に正規化
    - 画像を各チャンネルごとに標準化(平均0, 標準偏差1)
2. KMeans法を用いてパラメータμ、Σ、πを初期化
3. ギブスサンプリングによる推論
4. クラスタリングの結果の可視化
    - 各ピクセルごとに割り当てられたクラスタに応じて着色

## パラメータ
- クラスタ数：K = 3
- データの次元数：D = 3(RGBの3チャンネルのため)
- データ数：N = height x width
- 試行回数：MaxIter = 50
- μの事前分布：β = 1.0、m = [0, 0, 0]
- Λの事前分布：W = 3 * I, ν = 3
- πの事前分布：α = [2, 2, 2]

## アルゴリズム
疑似コード ([3]アルゴリズム4.5 ガウス混合モデルのためのギブスサンプリング)

```
パラメータのサンプルμ, Λ, πに初期値を設定
for i=1, ..., MAXITER do
    for n=1, ..., N do
        式(4.93)を用いてs_nをサンプル
    end for
    for k=1, ..., K do
        式(4.102)を用いてΛ_kをサンプル
        式(4.98)を用いてμ_kをサンプル
    end for
    式(4.44)を用いてπをサンプル
end for
```

### 式
- <img src="https://latex.codecogs.com/svg.image?\bg_white&space;\;&space;\mathbf{s}_n&space;\sim&space;\mathrm{Cat}(\mathbf{s}_n|\eta_n)&space;\;" title="\bg_white \; \mathbf{s}_n \sim \mathrm{Cat}(\mathbf{s}_n|\eta_n) \;" /> &ensp; (4.93)
- <img src="https://latex.codecogs.com/svg.image?\bg_white&space;\;&space;\mathbf{\Lambda}_k&space;\sim&space;\mathcal{W}(\mathbf{\Lambda}_k|\hat\nu_k,&space;\hat{\mathbf{W}}_k)&space;\;" title="\bg_white \; \mathbf{\Lambda}_k \sim \mathcal{W}(\mathbf{\Lambda}_k|\hat\nu_k, \hat{\mathbf{W}}_k) \;" /> &ensp; (4.102)
- <img src="https://latex.codecogs.com/svg.image?\bg_white&space;\;&space;\mathbf{\mu}_k&space;\sim&space;\mathcal{N}(\mathbf{\mu}_k|\hat{\mathbf{m}}_k,&space;(\hat{\beta}_k\mathbf{\Lambda}_k)^{-1})&space;\;" title="\bg_white \; \mathbf{\mu}_k \sim \mathcal{N}(\mathbf{\mu}_k|\hat{\mathbf{m}}_k, (\hat{\beta}_k\mathbf{\Lambda}_k)^{-1}) \;" /> &ensp; (4.98)
- <img src="https://latex.codecogs.com/svg.image?\bg_white&space;\;&space;\mathbf{\pi}&space;\sim&space;\mathrm{Dir}(\mathbf{\pi}|\hat{\mathbf{\alpha}})&space;\;" title="\bg_white \; \mathbf{\pi} \sim \mathrm{Dir}(\mathbf{\pi}|\hat{\mathbf{\alpha}}) \;" /> &ensp; (4.44)


## Λの事前分布のハイパーパラメータW = I * param
- I：D x Dの単位行列
- param：ハイパーパラメータ

paramの値によってセグメンテーションの結果が変化する。今回は、param = 3.0以上でうまくいく事がわかったため、param = 3.0で実験を行っている。

|![](optimize_param/result_0.005.png)|![](optimize_param/result_0.1.png)|![](optimize_param/result_1.0.png)|
|:---:|:---:|:---:|
|![](optimize_param/plot_0.005.png)|![](optimize_param/plot_0.1.png)|![](optimize_param/plot_1.0.png)|
|param = 0.005|param = 0.1|param = 1.0|

|![](optimize_param/result_2.0.png)|![](optimize_param/result_5.0.png)|![](optimize_param/result_10.0.png)|
|:---:|:---:|:---:|
|![](optimize_param/plot_2.0.png)|![](optimize_param/plot_5.0.png)|![](optimize_param/plot_10.0.png)|
|param = 2.0|param = 5.0|param = 10.0|


## 参考
[1] [【Python】4.4.2：ガウス混合モデルにおける推論：ギブスサンプリング【緑ベイズ入門のノート】](https://www.anarchive-beta.com/entry/2020/11/28/210948)

[2] [laituan245/image-segmentation-GMM](https://github.com/laituan245/image-segmentation-GMM/blob/master/main.py)

[3] [機械学習スタートアップシリーズ ベイズ推論による機械学習入門 (KS情報科学専門書) 単行本（ソフトカバー） – 2017/10/21
須山 敦志  (著), 杉山 将 (監修)](https://www.amazon.co.jp/dp/4061538322/)


## サンプルデータセット
[image-segmentation-GMM/images/](https://github.com/laituan245/image-segmentation-GMM/tree/master/images)
