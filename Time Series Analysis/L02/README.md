# **多元回归模型 (Multiple Regression Model) - Chapter 2 笔记**

## **1. 回归分析的目标**
### **定义**
回归分析的目标是探索**两个或多个变量**之间的关系，从而通过已知变量的值来预测另一个变量的值。

### **案例 1：Verizon Wireless 的客户流失预测**
#### **第一阶段**
- 通过客户当前套餐、通话记录、服务请求数据，建立预测哪些客户可能离开的模型。

#### **第二阶段**
- 使用预测模型选取高流失风险的客户，并提供不同的套餐优惠。
- 追踪客户接受优惠的情况，并基于数据进一步优化模型。

#### **结果**
- 2% 的客户流失率降至 1.5%（减少 25%）。
- 每年挽回客户流失价值 **\$700M**。
- 目标邮件营销成本降低 60%。

---

## **2. 医学案例：胆固醇水平预测**
### **数据集**
24 名高胆固醇患者的**血浆总胆固醇水平 (mg/ml)** 记录：3.5, 1.9, 4.0, 2.6, 4.5, ...

### **方法 1：使用均值预测**
$$
\hat{Y} = \frac{1}{n} \sum_{i=1}^{n} Y_i
$$

### **方法 2：使用回归预测**
引入 **年龄** 变量，建立线性回归模型：
$$
Y = \beta_0 + \beta_1 X + \epsilon
$$

发现 **年龄与胆固醇水平存在线性关系**，比单纯使用均值预测更准确。

---

## **3. 回归的基本概念**
### **变量类型**
- **定量变量 (Quantitative Variables)**：可用数值表示，如**年龄、收入、时间**等。
- **定性变量 (Qualitative Variables)**：不可用数值直接表示，如**性别、教育水平、犯罪类型**等。

### **变量分类**
- **因变量 (Response Variable, Y)**：需要预测的变量（又称**输出变量**）。
- **自变量 (Predictor Variables, X)**：用于预测因变量的变量（又称**输入变量**）。

### **两种回归关系**
1. **确定性关系 (Functional Relationship)**：
   \[
   Y = 2X
   \]
2. **统计关系 (Statistical Relationship)**：
   \[
   Y = E[Y] + \epsilon
   \]
   其中，\( E[Y] \) 是回归模型的**期望值**。

---

## **4. 多元线性回归 (Multiple Linear Regression, MLR)**
### **数学模型**
\[
Y_i = \beta_0 + \beta_1 X_{i1} + \beta_2 X_{i2} + \dots + \beta_p X_{ip} + \epsilon_i
\]
其中：
- \( Y_i \)：因变量
- \( X_{i1}, X_{i2}, ..., X_{ip} \)：自变量
- \( \beta_0, \beta_1, ..., \beta_p \)：回归系数
- \( \epsilon_i \)：误差项

### **矩阵形式**
\[
Y = X\beta + \epsilon, \quad \epsilon \sim N(0, \sigma^2 I)
\]
其中：
- \( X \) 是 **设计矩阵 (design matrix)**。
- \( \beta \) 是 **回归系数向量**。

---

## **5. 估计回归系数**
### **最小二乘法 (Least Squares Estimation, LSE)**
目标是**最小化残差平方和 (Sum of Squared Errors, SSE)**：
\[
\sum_{i=1}^{n} (Y_i - \beta_0 - \beta_1 X_{i1} - \dots - \beta_p X_{ip})^2
\]
通过求导，得出最优解：
\[
\hat{\beta} = (X'X)^{-1} X'Y
\]

---

## **6. 误差方差估计**
### **方差估计公式**
\[
s^2 = \frac{SSE}{n - p - 1}
\]
其中：
- **SSE** 是误差平方和。
- \( n - p - 1 \) 是自由度。

---

## **7. 评价回归模型**
### **残差 (Residuals)**
计算方法：
\[
e_i = Y_i - \hat{Y}_i
\]
**残差分析**用于检测模型拟合的质量。

### **决定系数 \( R^2 \)**
衡量自变量对因变量的解释能力：
\[
R^2 = \frac{SSR}{SYY}
\]
**调整后 \( R^2 \)**：
\[
R_a^2 = 1 - \frac{SSE / (n - p - 1)}{SYY / (n - 1)}
\]

---

## **8. 假设检验**
### **F 检验**
计算 F 统计量：
\[
F = \frac{MS_{Reg}}{MS_{Error}}
\]
若 \( p\text{-value} < 0.05 \)，则回归模型显著。

### **t 检验（个别回归系数检验）**
计算 t 统计量：
\[
t^* = \frac{\hat{\beta}_j}{\text{SE}(\hat{\beta}_j)}
\]
若 \( |t^*| > t_{\alpha/2, n-p-1} \)，则拒绝 \( H_0 \)。

---

## **9. 逻辑回归 (Logistic Regression)**
适用于**因变量是二分类变量 (Binary Outcome)** 的情况：
\[
\log \frac{p}{1 - p} = \beta_0 + \beta_1 X_1 + \dots + \beta_p X_p + \epsilon
\]

**估计公式**：
\[
p_i = \frac{e^{\beta_0 + \beta_1 X_1 + \dots + \beta_p X_p}}{1 + e^{\beta_0 + \beta_1 X_1 + \dots + \beta_p X_p}}
\]

---

## **10. 代码实现**
### **多元回归 (MLR)**
```r
mlr <- lm(y ~ x1 + x2 + x3, data=ozone)
summary(mlr)
anova(mlr)
```

### 逻辑回归

```R
model <- glm(Win ~ Distance, data = train, family="binomial")
```

