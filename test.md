# test

## 1

OK**ssss**

$$
\begin{align*}
abcdefghijklmnopqrstuvwxyz\\
\mathbb{abcdefghijklmnopqrstuvwxyz}\\
ABCDEFGHIJKLMNOPQRSTUVWXYZ\\
\mathbb{ABCDEFGHIJKLMNOPQRSTUVWXYZ}
\end{align*}
$$


$$
\begin{align*}
    p(\mathbf{z_0},u_0)&=p(u_0|\mathbf{z_0})p(\mathbf{z_0})\\
    &=\frac{1}{kq(\mathbf{z_0})}\times q(\mathbf{z_0})\\
    &=1/k
\end{align*}
$$

$$
\begin{align*}
    p(z_0|accpet)&=\int p(z_0,u|accpet)du\\
    &=\int \frac{p(z_0,u,accept)}{P(accept)}du\\
    &=\frac{1}{P(accept)}\int_{u\leq \tilde{p}(z_0)} p(z_0,u)du\\
    &=\frac{\tilde{p}(z_0)/k}{\int \tilde{p}(z)dz/k}\\
    &=p(z_0)\\
    p(z)=\frac{\tilde{p}(z)}{\int \tilde{p}(z)dz}
\end{align*}
$$