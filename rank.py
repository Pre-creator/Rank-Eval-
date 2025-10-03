import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs

class HybridRankingSystem:
    def __init__(self, comparison_matrix, weights=[1/3, 1/3, 1/3]):
        """
        初始化混合排名系统
        :param comparison_matrix: 比较矩阵 M[i][j] = 1 表示 Fi > Fj，否则为 0
        :param weights: 三种方法的权重 [elo_weight, hodge_weight, centrality_weight]
        """
        self.M = np.array(comparison_matrix)
        self.n = self.M.shape[0]  # 方法数量
        self.weights = weights
        self.elo_scores = None
        self.hodge_scores = None
        self.centrality_scores = None
        self.final_scores = None
        self.ranking = None
        
    def _normalize_scores(self, scores):
        """将分数归一化到 [0, 1] 范围"""
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score == min_score:
            return np.ones_like(scores) * 0.5
        return (scores - min_score) / (max_score - min_score)
    
    def compute_elo_scores(self, initial_score=1500, K=32, iterations=10):
        """
        计算Elo分数
        :param initial_score: 初始分数
        :param K: 调整系数
        :param iterations: 迭代次数
        """
        scores = np.ones(self.n) * initial_score
        
        for _ in range(iterations):
            new_scores = np.copy(scores)
            # 遍历所有比较对
            for i in range(self.n):
                for j in range(self.n):
                    if i != j:
                        # 实际结果: Fi > Fj 则为1，否则为0
                        outcome = self.M[i][j]
                        
                        # 计算预期概率
                        diff = scores[j] - scores[i]
                        expected = 1 / (1 + 10 **(diff / 400))
                        
                        # 更新分数
                        new_scores[i] += K * (outcome - expected)
            
            scores = new_scores
        
        self.elo_scores = self._normalize_scores(scores)
        return self.elo_scores
    
    def compute_hodge_rank(self):
        """
        计算HodgeRank分数（简化实现）
        核心思想：将比较矩阵视为有向图，通过求解最小二乘问题找到全局一致的排名
        """
        # 构建有向图
        G = nx.DiGraph()
        for i in range(self.n):
            G.add_node(i)
            for j in range(self.n):
                if i != j and self.M[i][j] == 1:
                    G.add_edge(j, i, weight=1)  # i > j 表示从j到i有一条边
        
        # 构建拉普拉斯矩阵
        laplacian = nx.laplacian_matrix(G).toarray()
        # 添加一个约束条件以避免零解
        laplacian[-1, :] = 1
        laplacian[-1, -1] = 1
        
        # 构建右侧向量
        b = np.zeros(self.n)
        b[-1] = 0.5  # 最后一个节点的约束值
        
        # 求解线性方程组 Lx = b
        scores = np.linalg.lstsq(laplacian, b, rcond=None)[0]
        
        self.hodge_scores = self._normalize_scores(scores)
        return self.hodge_scores
    
    def compute_centrality_scores(self, centrality_type='eigenvector'):
        """
        计算中心性分数
        :param centrality_type: 中心性类型 'degree', 'betweenness', 'eigenvector'
        """
        # 构建有向图
        G = nx.DiGraph()
        for i in range(self.n):
            G.add_node(i)
            for j in range(self.n):
                if i != j and self.M[i][j] == 1:
                    G.add_edge(i, j)  # i > j 表示从i到j有一条边
        
        if centrality_type == 'degree':
            centrality = nx.degree_centrality(G)
        elif centrality_type == 'betweenness':
            centrality = nx.betweenness_centrality(G)
        elif centrality_type == 'eigenvector':
            centrality = nx.eigenvector_centrality(G, max_iter=1000)
        else:
            raise ValueError("不支持的中心性类型")
        
        # 转换为数组形式
        scores = np.array([centrality[i] for i in range(self.n)])
        self.centrality_scores = self._normalize_scores(scores)
        return self.centrality_scores
    
    def compute_final_ranking(self, elo_iterations=10, centrality_type='eigenvector'):
        """计算最终排名"""
        # 确保所有分数都已计算
        if self.elo_scores is None:
            self.compute_elo_scores(iterations=elo_iterations)
        if self.hodge_scores is None:
            self.compute_hodge_rank()
        if self.centrality_scores is None:
            self.compute_centrality_scores(centrality_type)
        
        # 加权融合
        self.final_scores = (
            self.weights[0] * self.elo_scores +
            self.weights[1] * self.hodge_scores +
            self.weights[2] * self.centrality_scores
        )
        
        # 生成排名 (从高到低)
        self.ranking = np.argsort(-self.final_scores)
        return self.ranking, self.final_scores

# 示例用法
if __name__ == "__main__":
    # 示例比较矩阵: 5个方法之间的比较结果
    # M[i][j] = 1 表示方法i优于方法j，否则为0
    comparison_matrix = [
        [0, 1, 1, 0, 1],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 0, 1],
        [1, 1, 1, 0, 1],
        [0, 0, 0, 0, 0]
    ]
    
    # 创建排名系统实例，设置权重
    ranking_system = HybridRankingSystem(
        comparison_matrix, 
        weights=[0.4, 0.3, 0.3]  # Elo占40%，HodgeRank占30%，中心性占30%
    )
    
    # 计算最终排名
    ranking, scores = ranking_system.compute_final_ranking(
        elo_iterations=15, 
        centrality_type='eigenvector'
    )
    
    # 输出结果
    print("各方法的最终得分:", scores)
    print("排名 (从高到低):", ranking + 1)  # +1 是为了从1开始编号
    
    # 输出各方法的单独得分，供参考
    print("\n各方法的Elo得分:", ranking_system.elo_scores)
    print("各方法的HodgeRank得分:", ranking_system.hodge_scores)
    print("各方法的中心性得分:", ranking_system.centrality_scores)
