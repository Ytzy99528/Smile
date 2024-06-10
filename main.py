import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, choice):
        predictions = []
        for x_test in X:
            if choice == 1:
                # 欧氏距离
                distances = [np.linalg.norm(x_train - x_test) for x_train in self.X_train]
            elif choice == 2:
                # 曼哈顿距离
                distances = [sum(abs(x_train - x_test)) for x_train in self.X_train]
            else:
                # 切比雪夫距离
                distances = [max(abs(x_train - x_test)) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]      # 返回排序后的索引
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            most_common = max(set(k_nearest_labels), key=k_nearest_labels.count)    # key:辅助求最大值
            predictions.append(most_common)
        return predictions


def rgb_to_gray(rgb_image):
    # 初始化灰度图像数组
    gray_image = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.uint8)

    for i in range(rgb_image.shape[0]):
        for j in range(rgb_image.shape[1]):
            # 获取当前像素的RGB值
            r, g, b = rgb_image[i, j]
            # 计算灰度值
            gray = int(0.2989 * r + 0.5870 * g + 0.1140 * b)
            # 设置灰度图像的像素值
            gray_image[i, j] = gray
    return gray_image


# 交叉验证
def cross_validate(X, y, k_values, cv=5):
    fold_size = len(X) // cv
    k_scores = []
    X = np.array(X)
    y = np.array(y)
    for k in k_values:
        scores = []
        for fold in range(cv):
            # 创建训练集和验证集
            start, end = fold * fold_size, (fold + 1) * fold_size
            X_train = np.concatenate((X[:start], X[end:]))
            y_train = np.concatenate((y[:start], y[end:]))
            X_val = X[start:end]
            y_val = y[start:end]

            # 训练模型
            knn = KNN(k=k)
            knn.fit(X_train, y_train)
            predictions = knn.predict(X_val, choice=1)

            # 计算准确率
            accuracy = np.mean(predictions == y_val)
            scores.append(accuracy)
        k_scores.append(np.mean(scores))

    # 选择最佳k值
    best_k = k_values[np.argmax(k_scores)]
    return best_k, k_scores


# 算法评价
def classification_report(y_true, y_pred):
    # 初始化混淆矩阵的四个参数
    TP = 0  # 真正例
    TN = 0  # 真负例

    # 计算混淆矩阵的参数
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            TP += 1
        if y_true[i] == y_pred[i]:
            TN += 1
    # 计算准确率
    accuracy = TP / (TP + TN)

    # 打印结果
    print(f'Accuracy: {accuracy}')


# 第一步 切分训练集和测试集
X = []  # 定义图像名称
Y = []  # 定义图像分类类标

for i in range(1, 4):
    # 遍历文件夹，读取图片
    for f in os.listdir("./photo/%s" % i):
        # 获取图像名称
        X.append("./photo/" + str(i) + "/" + str(f))
        # 获取图像标签即为文件夹名称
        Y.append(i)

X = np.array(X)
Y = np.array(Y)
# 随机率为100% 选取其中的30%作为测试集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=2)
print(len(X_train), len(X_test), len(y_train), len(y_test))


# 第二步 图像读取及转换为像素直方图
# 训练集
XX_train = []
for i in X_train:
    # 读取图像
    image = cv2.imread(i)
    image = rgb_to_gray(image)
    # 图像像素大小一致
    img = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
    # 计算图像直方图并存储至X数组
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # hist = cv2.calcHist([img], [0, 1], None, [256, 256], [0.0, 255.0, 0.0, 255.0])
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    XX_train.append(((hist / 255).flatten()))

# 测试集
XX_test = []
for i in X_test:
    image = cv2.imread(i)
    image = rgb_to_gray(image)
    # 图像像素大小一致
    img = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # 计算图像直方图并存储至X数组
    # hist = cv2.calcHist([img], [0, 1], None, [256, 256], [0.0, 255.0, 0.0, 255.0])

    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.plot((hist / 255).flatten())
    plt.show()
    XX_test.append(((hist / 255).flatten()))


# 假设 X 是您的特征数据，y 是您的标签
k_values = range(1, 31)
best_k, k_scores = cross_validate(XX_train, y_train, k_values)
print(f'最佳的k值为: {best_k}')


clf = KNN(k=3)
clf.fit(XX_train, y_train)
predictions_labels = clf.predict(XX_test, choice=1)

print('预测结果:')
print(predictions_labels)

print('算法评价:')
print((classification_report(y_test, predictions_labels)))

# 第三步 输出测试集图片及预测结果
k = 0
while k < len(X_test):
    # 读取图像
    print(X_test[k])
    image = cv2.imread(X_test[k])
    print(predictions_labels[k])
    # 显示图像
    cv2.imshow("img", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    k = k + 1