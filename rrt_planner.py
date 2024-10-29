# rrt_planner.py
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Node:
    x: float
    y: float
    parent_id: Optional[int] = None

@dataclass
class SectorConfig:
    center_angle: float  # 扇形中心角度
    span_angle: float   # 扇形张角
    priority: float     # 优先级 (0-1)

class ImprovedRRT:
    def __init__(self, start: Tuple[float, float], goal: Tuple[float, float], 
                 bounds: Tuple[float, float, float, float],
                 obstacles: List[Tuple[float, float, float]],  # (x, y, radius)
                 step_size: float = 0.5,
                 max_iterations: int = 5000):
        self.start = Node(*start)
        self.goal = Node(*goal)
        self.bounds = bounds
        self.obstacles = obstacles
        self.step_size = step_size
        self.max_iterations = max_iterations
        
        self.nodes: List[Node] = [self.start]
        self.current_sectors: List[SectorConfig] = []
        
    def sample_in_sector(self, sector: SectorConfig) -> Tuple[float, float]:
        """在给定扇形区域内采样"""
        while True:
            # 随机采样距离和角度
            distance = np.random.uniform(0, self.step_size)
            base_angle = sector.center_angle
            angle_range = sector.span_angle / 2
            angle = base_angle + np.random.uniform(-angle_range, angle_range)
            
            # 计算采样点坐标
            x = self.nodes[-1].x + distance * np.cos(angle)
            y = self.nodes[-1].y + distance * np.sin(angle)
            
            # 检查是否在边界内
            if (self.bounds[0] <= x <= self.bounds[2] and 
                self.bounds[1] <= y <= self.bounds[3]):
                return x, y
    
    def is_collision_free(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        """检查路径是否无碰撞"""
        for ox, oy, r in self.obstacles:
            # 简化的碰撞检测：检查路径上的几个点是否与障碍物相交
            t = np.linspace(0, 1, 10)
            path_x = x1 + (x2 - x1) * t
            path_y = y1 + (y2 - y1) * t
            
            for px, py in zip(path_x, path_y):
                if np.sqrt((px - ox)**2 + (py - oy)**2) <= r:
                    return False
        return True
    
    def find_nearest_node(self, point: Tuple[float, float]) -> Node:
        """找到最近的节点"""
        distances = [(node.x - point[0])**2 + (node.y - point[1])**2 
                    for node in self.nodes]
        return self.nodes[np.argmin(distances)]
    
    def update_sectors(self, sectors: List[SectorConfig]):
        """更新采样扇区配置"""
        self.current_sectors = sectors
    
    def plan(self, multimodal_feedback) -> Optional[List[Node]]:
        """执行路径规划"""
        for i in range(self.max_iterations):
            # 根据当前扇区配置选择一个扇区进行采样
            if self.current_sectors:
                weights = [s.priority for s in self.current_sectors]
                chosen_sector = np.random.choice(self.current_sectors, p=weights/np.sum(weights))
                sampled_point = self.sample_in_sector(chosen_sector)
            else:
                # 如果没有扇区配置，进行全局采样
                sampled_point = (
                    np.random.uniform(self.bounds[0], self.bounds[2]),
                    np.random.uniform(self.bounds[1], self.bounds[3])
                )
            
            # 找到最近节点
            nearest_node = self.find_nearest_node(sampled_point)
            
            # 检查新节点是否可达
            if self.is_collision_free(nearest_node.x, nearest_node.y,
                                    sampled_point[0], sampled_point[1]):
                new_node = Node(sampled_point[0], sampled_point[1], 
                              len(self.nodes) - 1)
                self.nodes.append(new_node)
                
                # 检查是否可以连接到目标
                if self.is_collision_free(new_node.x, new_node.y,
                                        self.goal.x, self.goal.y):
                    self.goal.parent_id = len(self.nodes) - 1
                    self.nodes.append(self.goal)
                    return self.extract_path()
                
                # 获取多模态模型反馈并更新扇区
                feedback = multimodal_feedback(new_node, self.goal, self.obstacles)
                self.update_sectors(feedback)
        
        return None
    
    def extract_path(self) -> List[Node]:
        """提取最终路径"""
        path = []
        current_node = self.nodes[-1]  # 目标节点
        
        while current_node.parent_id is not None:
            path.append(current_node)
            current_node = self.nodes[current_node.parent_id]
        
        path.append(self.start)
        return path[::-1]