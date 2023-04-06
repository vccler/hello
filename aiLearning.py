# 导入必要的库和模块
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载数据
data = pd.read_csv('credit.csv')

# 数据预处理
data = data.dropna()  # 删除缺失值
X = data.drop('Attrition_Flag', axis=1).drop('Gender', axis=1).drop('Education_Level', axis=1)\
    .drop('Marital_Status', axis=1).drop('Income_Category', axis=1).drop('Card_Category', axis=1)
y = data['Total_Trans_Amt']  # 标签

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 构建决策树模型
model = DecisionTreeClassifier(max_depth=3)  # 设置树的最大深度为3
model.fit(X_train, y_train)  # 拟合模型

# 在测试集上评估模型
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy: ", acc)

