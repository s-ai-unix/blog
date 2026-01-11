---
title: "MathJax 测试页面"
date: 2024-01-11T10:00:00+08:00
draft: false
description: "测试 MathJax 数学公式渲染"
categories: ["测试"]
tags: ["MathJax", "测试"]
mathjax: true
---

## MathJax 公式渲染测试

这是一个测试页面,用于验证 MathJax 是否正常工作。

### 行内公式测试

这是一些行内公式的测试:
- 欧拉公式: $e^{i\pi} + 1 = 0$
- 质能方程: $E = mc^2$
- 勾股定理: $a^2 + b^2 = c^2$
- 积分: $\int_0^1 x^2 dx = \frac{1}{3}$
- 求和: $\sum_{i=1}^{n} i = \frac{n(n+1)}{2}$

### 块级公式测试

一些重要的数学公式:

**二次公式**:
$$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$$

**微积分基本定理**:
$$\int_a^b f(x) dx = F(b) - F(a)$$

**正态分布**:
$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

**傅里叶变换**:
$$\hat{f}(\xi) = \int_{-\infty}^{\infty} f(x) e^{-2\pi i x \xi} dx$$

**矩阵运算**:
$$\begin{pmatrix} a & b \\ c & d \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} ax + by \\ cx + dy \end{pmatrix}$$

### 下标和上标测试

- 向量: $\vec{v} = (v_1, v_2, \ldots, v_n)$
- 矩阵: $A = [a_{ij}]_{i,j=1}^{n}$
- 偏导数: $\frac{\partial f}{\partial x_i}$
- 梯度: $\nabla f = \left(\frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_n}\right)$

### 希腊字母测试

- $\alpha, \beta, \gamma, \delta, \epsilon, \zeta, \eta, \theta, \iota, \kappa, \lambda, \mu$
- $\nu, \xi, \pi, \rho, \sigma, \tau, \upsilon, \phi, \chi, \psi, \omega$
- 大写: $\Gamma, \Delta, \Theta, \Lambda, \Sigma, \Phi, \Psi, \Omega$

### 特殊符号测试

- 集合: $\in, \notin, \subset, \subseteq, \cup, \cap, \emptyset$
- 逻辑: $\forall, \exists, \Rightarrow, \Leftrightarrow, \neg, \vee, \wedge$
- 箭头: $\to, \rightarrow, \Rightarrow, \leftarrow, \leftrightarrow$
- 关系: $\leq, \geq, \neq, \approx, \equiv, \sim$
- 其他: $\infty, \partial, \nabla, \pm, \mp$

### 分数和根号测试

- 分数: $\frac{a}{b}, \frac{1}{2}, \frac{x^2+1}{x-1}$
- 根号: $\sqrt{x}, \sqrt[3]{x}, \sqrt{n+1}$
- 组合: $\binom{n}{k}, \binom{n}{k}p^k(1-p)^{n-k}$

### 极限和级数测试

- 极限: $\lim_{x \to \infty} \frac{1}{x} = 0$
- 级数求和: $\sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6}$
- 无穷乘积: $\prod_{n=2}^{\infty} \frac{n^2}{n^2-1} = 2$

如果所有公式都能正确渲染,说明 MathJax 配置成功!
