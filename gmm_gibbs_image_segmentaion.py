# -*- coding: utf-8 -*-
""" Image Segmentaion using Gibbs Sampling to Gaussian Mixture Models

## 参考
- [【Python】4.4.2：ガウス混合モデルにおける推論：ギブスサンプリング【緑ベイズ入門のノート】](https://www.anarchive-beta.com/entry/2020/11/28/210948)
- [laituan245/image-segmentation-GMM](https://github.com/laituan245/image-segmentation-GMM/blob/master/main.py)
"""

import numpy as np
from scipy.stats import multivariate_normal, wishart, dirichlet  # 多次元ガウス分布、ウィシャート分布、ディリクレ分布
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image


# 画像の読み込み
img = np.asarray(Image.open('./deer.jpg').resize((150, 100)), dtype=np.float32)
img_h, img_w, img_c = img.shape
img_px = img.reshape(-1, img_c)/255.0

# 画像の標準化
_mean = np.mean(img_px, axis=0, keepdims=True)
_std = np.mean(img_px, axis=0, keepdims=True)
img_px = (img_px - _mean) / _std

# 初期値を設定

# クラスタ数
K = 3
# データの次元数
D = 3  # RGBなので
# データ数
N = img_px.shape[0]

# 事前分布の設定
# muの事前分布のパラメータを指定
beta = 1.0
m_d = np.repeat(0.0, D)

# lambdaの事前分布のパラメータを指定
w_dd = np.identity(D)*3  # この値を大きくするとクラスタリングがうまくいく、1以上で正しく動き3程度が丁度いい、5以上はあまり変わらない
nu = D

# piの事前分布のパラメータを指定
alpha_k = np.repeat(2.0, K)

# GMMの初期値を求めるためにKMeansを使う場合
kmeans = KMeans(n_clusters=K)
labels = kmeans.fit_predict(img_px)
init_mu = kmeans.cluster_centers_
init_pi, init_cov = [], []
for i in range(K):
    datas = np.array([img_px[j, :] for j in range(len(labels)) if labels[j] == i]).T
    init_cov.append(np.cov(datas))
    init_pi.append(datas.shape[1]/float(len(labels)))
init_pi = np.array(init_pi)
init_cov = np.array(init_cov)

# mu, lambda, piをサンプリングして初期化する場合
mu_kd = np.empty((K, D))
lambda_kdd = np.empty((K, D, D))
for k in range(K):
    # クラスタkの精度行列をサンプル
    lambda_kdd[k] = wishart.rvs(df=nu, scale=w_dd, size=1)

    # クラスタkの平均をサンプル
    mu_kd[k] = np.random.multivariate_normal(
        mean=m_d, cov=np.linalg.inv(beta*lambda_kdd[k])
    ).flatten()

# 混合比率をサンプル
pi_k = dirichlet.rvs(alpha=alpha_k, size=1).flatten()

# KMeansで初期化する場合は、パラメータの初期値を上書き
mu_kd = init_mu
lambda_kdd = init_cov
pi_k = init_pi

# 作図用のx軸のxの値を作成
x_1_line = np.linspace(
    np.min(mu_kd[:, 0] -3*np.sqrt(lambda_kdd[:, 0, 0])),
    np.max(mu_kd[:, 0] +3*np.sqrt(lambda_kdd[:, 0, 0])),
    num=300
)

# 作図用のy軸のxの値を作成
x_2_line = np.linspace(
    np.min(mu_kd[:, 1] -3*np.sqrt(lambda_kdd[:, 1, 1])),
    np.max(mu_kd[:, 1] +3*np.sqrt(lambda_kdd[:, 1, 1])),
    num=300
)


# 作図用の格子状の点を作成
x_1_grid, x_2_grid = np.meshgrid(x_1_line, x_2_line)

# 作図用のxの点を作成
x_point = np.stack([x_1_grid.flatten(), x_2_grid.flatten()], axis=1)

# 作図用に各次元の要素数を保存
x_dim = x_1_grid.shape



# KMeansによる初期の分布を可視化
# 3次元データなので、R, Gに相当するインデックス0, 1を抜き出して可視化する
init_model_RG = 0
for k in range(K):
    # クラスタkの分布の確率密度を計算
    tmp_density = multivariate_normal.pdf(
        x=x_point, mean=mu_kd[k, :2], cov=lambda_kdd[k, :2, :2]
    )

    # K個の分布の加重平均を計算
    init_model_RG += pi_k[k] * tmp_density

init_model_GB = 0
for k in range(K):
    # クラスタkの分布の確率密度を計算
    tmp_density = multivariate_normal.pdf(
        x=x_point, mean=mu_kd[k, 1:], cov=lambda_kdd[k, 1:, 1:]
    )

    # K個の分布の加重平均を計算
    init_model_GB += pi_k[k] * tmp_density

# 計算に必要な各関数を定義
def calc_sn_param(x, mu, lam, pi):
    eta = np.zeros((N, K))
    # 潜在変数の事後分布のパラメータを計算(4.94)
    for k in range(K):
        tmp_eta_n = np.diag(
            -0.5*(x - mu[k]).dot(lam[k]).dot((x - mu[k]).T)
        ).copy()
        tmp_eta_n += 0.5*np.log(np.linalg.det(lam[k])+1e-7)
        tmp_eta_n += np.log(pi[k] + 1e-7)
        eta[:, k] = np.exp(tmp_eta_n)
    eta /= np.sum(eta, axis=1, keepdims=True)  # 正規化

    return eta


def sample_s_n(eta):
    # 潜在変数をサンプリング(4.93)
    return np.random.multinomial(n=1, pvals=eta, size=1).flatten()


def calc_mu_param(x, s_n):
    beta_hat = np.zeros(K)
    m_hat = np.zeros((K, D))
    for k in range(K):
        # muの事後分布のパラメータを計算(4.99)
        beta_hat[k] = np.sum(s_n[:, k]) + beta
        m_hat[k] = np.sum(s_n[:, k]*x.T, axis=1)
        m_hat[k] += beta * m_d
        m_hat[k] /= beta_hat[k]

    return beta_hat, m_hat


def calc_lam_param(x, s_n, beta_hat, m_hat):
    w_hat = np.zeros((K, D, D))
    nu_hat = np.zeros(K)
    for k in range(K):
        # lambdaの事後分布のパラメータを計算(4.103)
        tmp_w_dd = np.dot((s_n[:, k]*x.T), x)
        tmp_w_dd += beta * np.dot(m_d.reshape(D, 1), m_d.reshape(1, D))
        tmp_w_dd -= beta_hat[k]*np.dot(m_hat[k].reshape(D, 1), m_hat[k].reshape(1, D))
        tmp_w_dd += np.linalg.inv(w_dd)
        
        w_hat[k] = np.linalg.inv(tmp_w_dd)
        nu_hat[k] = np.sum(s_nk[:, k]) + nu

    return w_hat, nu_hat


def sample_lam(nu_hat, w_hat):
    # 式(4.102)を用いてlambda_kをサンプル
    return wishart.rvs(size=1, df=nu_hat, scale=w_hat)


def sample_mu(m_hat, beta_hat, lam):
    # 式(4.98)を用いてmu_kをサンプル
    return np.random.multivariate_normal(mean=m_hat, cov=np.linalg.inv(beta_hat*lam), size=1).flatten()


def calc_pi_param(s_n):
    # 混合比率のパラメータを計算(4.45)
    return np.sum(s_n, axis=0) + alpha_k


def sample_pi(alpha):
    # 式(4.44)を用いてpiをサンプル
    return dirichlet.rvs(size=1, alpha=alpha).flatten()

# 推論

# 試行回数を指定
MAXITER = 50

# パラメータの初期化
eta_nk = np.zeros((N, K))
s_nk = np.zeros((N, K))
beta_hat_k = np.zeros(K)
m_hat_kd = np.zeros((K, D))
w_hat_kdd = np.zeros((K, D, D))
nu_hat_k = np.zeros(K)
alpha_hat_k = np.zeros(K)

# ギブスサンプリング
for i in range(MAXITER):
    # 潜在変数の事後分布のパラメータを計算(4.94)
    eta_nk = calc_sn_param(img_px, mu_kd, lambda_kdd, pi_k)
    
    for n in range(N):
        # 式4.93を用いてs_nをサンプル
        s_nk[n] = sample_s_n(eta_nk[n])

    # muの事後分布のパラメータを計算(4.99)
    beta_hat_k, m_hat_kd = calc_mu_param(img_px, s_nk)
    # lambdaの事後分布のパラメータを計算(4.103)
    w_hat_kdd, nu_hat_k = calc_lam_param(img_px, s_nk, beta_hat_k, m_hat_kd)

    for k in range(K):
        # 式(4.102)を用いてlambda_kをサンプル
        lambda_kdd[k] = sample_lam(nu_hat_k[k], w_hat_kdd[k])
        # 式(4.98)を用いてmu_kをサンプル
        mu_kd[k] = sample_mu(m_hat_kd[k], beta_hat_k[k], lambda_kdd[k])

    # 混合比率のパラメータを計算(4.45)
    alpha_hat_k = calc_pi_param(s_nk)

    # 式(4.44)を用いてpiをサンプル
    pi_k = sample_pi(alpha_hat_k)


    # 値を記録
    _, s_n = np.where(s_nk == 1)

    # 実行状況の表示
    print("\r iter: %d [%s>%s](%d％)"%(i+1,"="*int(np.round((i+1)/MAXITER*20, 1)), "_"*(20-int(np.round((i+1)/MAXITER*20, 1))),np.round((i+1)/MAXITER*100, 1)), end="")

# クラスタリングの結果の表示
fig = plt.figure(figsize=(12, 4))
plt.suptitle("Compare KMeans v.s. GMM", fontsize=16)

plt.subplot(1, 2, 1)
plt.imshow(labels.reshape(img_h, img_w), vmin=0, vmax=K-1, cmap="brg")
plt.axis("off")
plt.title("KMeans labels")

plt.subplot(1, 2, 2)
plt.imshow(s_n.reshape(img_h, img_w), vmin=0, vmax=K-1, cmap="brg")
plt.axis("off")
plt.title("GMM Inference")

plt.show()

# 最後にサンプルしたパラメータによる混合分布を計算
# これも同様にR, Gに相当するインデックス0, 1を抜き出して可視化

res_density_RG = 0
for k in range(K):
    # クラスタkの分布の確率密度を計算
    tmp_density = multivariate_normal.pdf(
        x=x_point, mean=mu_kd[k, :2], cov=np.linalg.inv(lambda_kdd[k, :2, :2])
    )

    # K個の分布の加重平均を計算
    res_density_RG += tmp_density * pi_k[k]


res_density_GB = 0
for k in range(K):
    # クラスタkの分布の確率密度を計算
    tmp_density = multivariate_normal.pdf(
        x=x_point, mean=mu_kd[k, 1:], cov=np.linalg.inv(lambda_kdd[k, 1:, 1:])
    )

    # K個の分布の加重平均を計算
    res_density_GB += tmp_density * pi_k[k]


# 最終的な分布を作図
plt.figure(figsize=(12, 8))
plt.contour(x_1_grid, x_2_grid, init_model_RG.reshape(x_dim), 
           alpha=0.5, linestyles='dashed') # KMeansによる初期の分布
plt.scatter(x=init_mu[:, 0], y=init_mu[:, 1], 
            color='blue', s=100, marker='x')  # KMeansによる初期の平均
plt.scatter(x=mu_kd[:, 0], y=mu_kd[:, 1], 
            color='red', s=100, marker='x')  # 推論後の平均
plt.contour(x_1_grid, x_2_grid, res_density_RG.reshape(x_dim)) # サンプルによる分布:(塗りつぶし)
plt.xlabel("$x_1(R)$")
plt.ylabel("$x_2(G)$")
plt.suptitle('Gaussian Mixture Model: Gibbs Sampling', fontsize=16)
plt.show()

# 最終的な分布を作図
plt.figure(figsize=(12, 8))
plt.contour(x_1_grid, x_2_grid, init_model_GB.reshape(x_dim), 
           alpha=0.5, linestyles='dashed') # KMeansによる初期の分布
plt.scatter(x=init_mu[:, 1], y=init_mu[:, 2], 
            color='blue', s=100, marker='x')  # KMeansによる初期の平均
plt.scatter(x=mu_kd[:, 1], y=mu_kd[:, 2], 
            color='red', s=100, marker='x')  # 推論後の平均
plt.contour(x_1_grid, x_2_grid, res_density_GB.reshape(x_dim)) # サンプルによる分布:(塗りつぶし)
plt.xlabel("$x_1(G)$")
plt.ylabel("$x_2(B)$")
plt.suptitle('Gaussian Mixture Model: Gibbs Sampling', fontsize=16)
plt.show()

plt.imshow(s_n.reshape(img_h, img_w), vmin=0, vmax=K-1, cmap="brg")
plt.axis("off")
plt.show()