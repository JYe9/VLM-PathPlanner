# visualization.py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Wedge
from typing import List, Tuple
from rrt_planner import Node, SectorConfig

class Visualizer:
    def __init__(self, bounds: Tuple[float, float, float, float]):
        self.bounds = bounds
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.sector_patches = []
        
    def setup_plot(self):
        """设置基础绘图环境"""
        self.ax.set_xlim(self.bounds[0], self.bounds[2])
        self.ax.set_ylim(self.bounds[1], self.bounds[3])
        self.ax.grid(True)
        self.ax.set_aspect('equal')
        
    def draw_obstacles(self, obstacles: List[Tuple[float, float, float]]):
        """绘制障碍物"""
        for ox, oy, r in obstacles:
            obstacle = Circle((ox, oy), r, fill=True, color='red', alpha=0.3)
            self.ax.add_patch(obstacle)
            
    def draw_nodes(self, nodes: List[Node]):
        """绘制RRT树节点和边"""
        # 绘制节点之间的连接
        for node in nodes[1:]:  # 跳过起始节点
            if node.parent_id is not None:
                parent = nodes[node.parent_id]
                self.ax.plot([parent.x, node.x], [parent.y, node.y], 
                           'g-', alpha=0.5, linewidth=0.5)
        
        # 绘制所有节点
        node_x = [node.x for node in nodes]
        node_y = [node.y for node in nodes]
        self.ax.plot(node_x, node_y, 'g.', markersize=2)
        
    def draw_path(self, path: List[Node]):
        """绘制最终路径"""
        path_x = [node.x for node in path]
        path_y = [node.y for node in path]
        self.ax.plot(path_x, path_y, 'b-', linewidth=2, label='Final Path')
        
    def draw_sectors(self, current_node: Node, sectors: List[SectorConfig]):
        """绘制采样扇区"""
        # 清除之前的扇区
        for patch in self.sector_patches:
            patch.remove()
        self.sector_patches.clear()
        
        # 绘制新的扇区
        for sector in sectors:
            # 创建扇形
            wedge = Wedge(
                (current_node.x, current_node.y),
                r=2.0,  # 扇区半径
                theta1=sector.center_angle - sector.span_angle/2,
                theta2=sector.center_angle + sector.span_angle/2,
                alpha=sector.priority * 0.3,
                color='yellow'
            )
            self.sector_patches.append(wedge)
            self.ax.add_patch(wedge)
            
    def draw_start_goal(self, start: Node, goal: Node):
        """绘制起点和终点"""
        self.ax.plot(start.x, start.y, 'go', markersize=10, label='Start')
        self.ax.plot(goal.x, goal.y, 'ro', markersize=10, label='Goal')
        self.ax.legend()
        
    def update(self):
        """更新显示"""
        self.fig.canvas.draw()
        plt.pause(0.01)
        
    def save(self, filename: str):
        """保存图像"""
        plt.savefig(filename)
        
    def clear(self):
        """清除当前图像"""
        self.ax.clear()
        self.setup_plot()