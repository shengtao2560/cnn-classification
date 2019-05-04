# cnn-classification
使用卷积神经网络进行中文文本分类.

## 环境

- Python 3.7
- TensorFlow 1.3
- numpy
- scikit-learn
- scipy

## 预处理

- process.py 从`scrapiesrestb.xls`中读取经人工标注的数据库，打上分类标签，随机按比例处理为三种数据集.

## 数据集

- drug.train.txt: 训练集
- drug.val.txt: 验证集
- drug.test.txt: 测试集

按10:1:2划分

## 训练与测试

运行 `python run_cnn.py train`开始训练.

运行 `python run_cnn.py test` 在测试集上进行测试.

## 可视化

- tb.bat 运行后http://localhost:6006 连接tensorboard

## 预测

- predict.py 一个简单的预测Demo
