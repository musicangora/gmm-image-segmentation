# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 13:52:42 2021

@author: miiya
"""

import numpy as np

from scipy.stats import wishart, dirichlet  # ウィシャート分布、ディリクレ分布
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image


# Define function for calculation
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
        nu_hat[k] = np.sum(s_n[:, k]) + nu

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



if __name__ == '__main__':
    # Load image
    image_name = input("画像のファイル名を入力してください：")
    image_path = "./images/%s"%image_name
    image = Image.open(image_path)
    image = image.resize((image.width // 2, image.height // 2))
    image = np.asarray(image, dtype=np.float32)
    image_height, image_width, image_channels = image.shape
    image_pixels = image.reshape(-1, image_channels)/255.0
    
    # Normalization
    _mean = np.mean(image_pixels,axis=0,keepdims=True)
    _std = np.std(image_pixels,axis=0,keepdims=True)
    image_pixels = (image_pixels - _mean) / _std
    
    # Input number of classes
    K = int(input("クラス数を入力してください："))
    
    # Define initial parameter
    D = image_channels  # データの次元数
    N = image_pixels.shape[0]  # データ数
    
    # prior distribution: mu
    beta = 1.0
    m_d = np.repeat(0.0, D)
    
    # prior distribution: Lambda
    w_dd = np.identity(D)*3
    nu = D
    
    # prior distribution: pi
    alpha_k = np.repeat(2.0, K)
    
    
    # Apply K-Means to find the initial weights and
    # covariance matrices for GMM
    kmeans = KMeans(n_clusters=K)
    labels = kmeans.fit_predict(image_pixels)
    mu_kd = kmeans.cluster_centers_
    pi_k, lambda_kdd = [], []
    for i in range(K):
        datas = np.array([image_pixels[j, :] for j in range(len(labels)) if labels[j] == i]).T
        lambda_kdd.append(np.cov(datas))
        pi_k.append(datas.shape[1]/float(len(labels)))
    pi_k = np.array(pi_k)
    lambda_kdd = np.array(lambda_kdd)
    
    
    # Input max iteration
    MAXITER = int(input("最大試行回数を入力してください："))
    
    # Initialize parameters
    eta_nk = np.zeros((N, K))
    s_nk = np.zeros((N, K))
    beta_hat_k = np.zeros(K)
    m_hat_kd = np.zeros((K, D))
    w_hat_kdd = np.zeros((K, D, D))
    nu_hat_k = np.zeros(K)
    alpha_hat_k = np.zeros(K)
    
    # Gibbs Sampling
    for i in range(MAXITER):
        # 潜在変数の事後分布のパラメータを計算(4.94)
        eta_nk = calc_sn_param(image_pixels, mu_kd, lambda_kdd, pi_k)
        
        for n in range(N):
            # 式4.93を用いてs_nをサンプル
            s_nk[n] = sample_s_n(eta_nk[n])
    
        # muの事後分布のパラメータを計算(4.99)
        beta_hat_k, m_hat_kd = calc_mu_param(image_pixels, s_nk)
        # lambdaの事後分布のパラメータを計算(4.103)
        w_hat_kdd, nu_hat_k = calc_lam_param(image_pixels, s_nk, beta_hat_k, m_hat_kd)
    
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


    # Show Result
    # クラスタリングの結果の表示
    fig = plt.figure(figsize=(12, 4))
    plt.suptitle("Compare KMeans v.s. GMM", fontsize=16)
    
    plt.subplot(1, 2, 1)
    plt.imshow(labels.reshape(image_height, image_width), vmin=0, vmax=K-1, cmap="brg")
    plt.axis("off")
    plt.title("KMeans labels")
    
    plt.subplot(1, 2, 2)
    plt.imshow(s_n.reshape(image_height, image_width), vmin=0, vmax=K-1, cmap="brg")
    plt.axis("off")
    plt.title("GMM Inference")
    
    plt.savefig("result_%s_K%d.png"%(image_name.split(".")[0], K))
    
    
