# main.py
import yaml
import os
import time
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

from rrt_planner import ImprovedRRT, Node
from multimodal_model import MultiModalInteraction
from visualization import Visualizer
from prompt_templates import PromptTemplate

def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # 加载配置
    config = load_config('config.yaml')
    
    # 设置环境参数
    env_config = config['environment']
    bounds = (
        env_config['bounds']['x_min'],
        env_config['bounds']['y_min'],
        env_config['bounds']['x_max'],
        env_config['bounds']['y_max']
    )
    start = (env_config['start_point']['x'], env_config['start_point']['y'])
    goal = (env_config['goal_point']['x'], env_config['goal_point']['y'])
    obstacles = env_config['obstacles']
    
    # 初始化组件
    rrt_planner = ImprovedRRT(
        start=start,
        goal=goal,
        bounds=bounds,
        obstacles=obstacles,
        step_size=config['rrt_params']['step_size'],
        max_iterations=config['rrt_params']['max_iterations']
    )
    
    multimodal = MultiModalInteraction(config['model_params'])
    visualizer = Visualizer(bounds)
    prompt_manager = PromptTemplate()
    
    # 设置可视化环境
    visualizer.setup_plot()
    visualizer.draw_obstacles(obstacles)
    visualizer.draw_start_goal(Node(*start), Node(*goal))
    
    # 定义反馈函数
    def feedback_function(current_node: Node, goal_node: Node, 
                        obstacles: List[Tuple[float, float, float]]):
        # 获取多模态模型的建议
        sectors_data = multimodal.get_feedback(
            current_node=current_node,
            goal_node=goal_node,
            nodes=rrt_planner.nodes,  # 传入所有已探索的节点
            obstacles=obstacles,
            bounds=bounds
        )
        
        # 将返回的数据转换为SectorConfig对象
        sectors = multimodal._convert_to_sector_config(sectors_data)
        
        # 可视化当前状态
        if config['visualization']['show_sectors']:
            visualizer.draw_sectors(current_node, sectors)
            visualizer.update()  # 实时更新显示
        
        return sectors
    
    # 执行路径规划
    start_time = time.time()
    path = rrt_planner.plan(feedback_function)
    end_time = time.time()
    
    if path:
        print(f"Path found in {end_time - start_time:.2f} seconds!")
        # 绘制最终结果
        visualizer.draw_nodes(rrt_planner.nodes)
        visualizer.draw_path(path)
        
        # 保存结果
        if not os.path.exists(config['visualization']['save_path']):
            os.makedirs(config['visualization']['save_path'])
        visualizer.save(
            os.path.join(
                config['visualization']['save_path'],
                f'result_{time.strftime("%Y%m%d_%H%M%S")}.png'
            )
        )
        plt.show()
    else:
        print("No path found!")

if __name__ == "__main__":
    main()