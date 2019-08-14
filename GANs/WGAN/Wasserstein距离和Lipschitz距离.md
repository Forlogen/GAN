## Wasserstein 距离

Wasserstein距离又称Earth-Move距离（EM距离），用来衡量两个分布之间的距离，它的定义式如下
$$
W(P_{1},P_{2})=\inf \limits_{\gamma \sim \prod(P_{1},P_{2})} E_{(x,y)\sim \gamma}[||x-y||]
$$
其中$\prod(P_{1},P_{2})$ 是$P_{1}$ 和$P_{2}$ 所有可能的联合分布的集合。对于每一个可能的联合分布$\gamma$，可以从中采样$(x,y) \sim \gamma$ ,得到一个样本$x$ 和$y$ ,并计算出这对样本的距离$||x-y||$ ,所以可以计算该联合分布下，样本对距离的期望值$E_{(x,y)\sim \gamma}[||x-y||]$ 。在所有可能的联合分布中能够对这个期望取到的下确界$\inf \limits_{\gamma \sim \prod(P_{1},P_{2})} E_{(x,y)\sim \gamma}[||x-y||]$ 就是Wasserstein距离。

直观上可以把求期望的过程理解为在联合分布$\gamma$ 这个路径规划下把土堆$P_{1}$移到土堆$P_{2}$ 所需的消耗，而Wasserstein距离就是在最优路径规划下的最小消耗，因此它又叫做Earth-Move距离。如下所示：

![1554965233640](C:\Users\dyliang\AppData\Roaming\Typora\typora-user-images\1554965233640.png)

我们将$p_{r}$ 代表的土堆变成$p_{\theta}$ 代表的土堆的最小消耗就是所要求的Wasserstein距离。

Wasserstein距离相比于$KL$ 散度和$JS$ 散度的优势在于，即使两个分布的支撑集没有重叠或是重叠部分非常的少，仍能反映两个分布的远近情况。而$ JS$ 散度在此情况下为一个常量，$KL$ 散度可能无意义。

同时根据Kantorovich-Rubinstein对偶原理，可以得到Wasserstein距离的等价形式如下所示：
$$
W(P_{1},P_{2})=\sup \limits_{||f||_{L}\leq 1}E_{x\sim P_{1}}[f(x)]-E_{x\sim P_{2}}[f(x)]
$$

## Lipschitz 连续定义

如果有函数$f(x)$ 。存在一个常量$K$ ,使得对$f(x)$ 定义域上的任意两个值满足如下的条件：
$$
|f(x_{1})-f(x_{2})|\leq |x_{1}-x_{2}|\ast K
$$
就称函数$f(x)$ 满足Lipschitz连续条件，并称$K$ 为$f(x)$ 的Lipschitz常数。它比一致连续要强，因为它限制了函数的局部变动幅度不能超过某常量。