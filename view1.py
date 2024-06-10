import networkx as nx
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimSun']


# 创建一个有向图
G = nx.DiGraph()

# 添加节点
nodes = ["数据", "灰度化", "尺寸标准化", "噪声去除", "图像直方图", "KNN", "分类结果"]
G.add_nodes_from(nodes)

# 添加边
edges = [("数据", "灰度化"), ("灰度化", "尺寸标准化"), ("尺寸标准化", "噪声去除"),
         ("噪声去除", "图像直方图"), ("图像直方图", "KNN"), ("KNN", "分类结果")]
G.add_edges_from(edges)

# 绘制图形
pos = nx.spring_layout(G, seed=42)
nx.draw_networkx_nodes(G, pos, node_color="pink", node_size=500)
nx.draw_networkx_edges(G, pos, arrows=True)
nx.draw_networkx_labels(G, pos)

plt.savefig("图1.jpg", format='jpg')
plt.axis("off")
plt.show()

