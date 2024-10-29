# prompt_templates.py
from typing import Dict, List, Tuple
import json

class PromptTemplate:
    def __init__(self):
        self.templates = {
            'default': """
            在2D路径规划任务中:
            当前位置: ({current_x:.2f}, {current_y:.2f})
            目标位置: ({goal_x:.2f}, {goal_y:.2f})
            距离目标: {distance_to_goal:.2f}
            
            环境信息:
            {obstacles_description}
            
            请提供以下建议:
            1. 主要探索方向（角度）
            2. 建议的扇区划分（2-3个）
            3. 每个扇区的优先级（0-1）
            4. 建议的采样步长
            
            输出格式要求：JSON格式
            {{
                "sectors": [
                    {{
                        "center_angle": float,
                        "span_angle": float,
                        "priority": float
                    }}
                ],
                "suggested_step_size": float,
                "confidence": float
            }}
            """,
            
            'obstacle_dense': """
            在障碍物密集区域的路径规划任务中:
            当前位置: ({current_x:.2f}, {current_y:.2f})
            目标位置: ({goal_x:.2f}, {goal_y:.2f})
            距离目标: {distance_to_goal:.2f}
            
            密集障碍物信息:
            {obstacles_description}
            
            请特别注意:
            1. 建议较小的采样步长
            2. 多个窄扇区
            3. 避障优先级
            
            {format_requirements}
            """,
            
            'near_goal': """
            接近目标的路径规划任务中:
            当前位置: ({current_x:.2f}, {current_y:.2f})
            目标位置: ({goal_x:.2f}, {goal_y:.2f})
            距离目标: {distance_to_goal:.2f}
            
            周围障碍物:
            {obstacles_description}
            
            请特别注意:
            1. 精确的方向控制
            2. 较小的扇区角度
            3. 较高的优先级分配
            
            {format_requirements}
            """
        }
        
        self.format_requirements = """
        请以JSON格式返回建议:
        {
            "sectors": [
                {
                    "center_angle": float,
                    "span_angle": float,
                    "priority": float
                }
            ],
            "suggested_step_size": float,
            "confidence": float
        }
        """
        
    def get_template(self, scenario: str) -> str:
        """获取特定场景的模板"""
        return self.templates.get(scenario, self.templates['default'])
    
    def format_obstacles_description(self, 
                                  obstacles: List[Tuple[float, float, float]],
                                  current_pos: Tuple[float, float]) -> str:
        """格式化障碍物描述"""
        descriptions = []
        for ox, oy, r in obstacles:
            dist = ((ox - current_pos[0])**2 + (oy - current_pos[1])**2)**0.5
            angle = np.degrees(np.arctan2(oy - current_pos[1], 
                                        ox - current_pos[0]))
            descriptions.append(
                f"- 障碍物位于 ({ox:.2f}, {oy:.2f}), "
                f"距离 {dist:.2f}, 方向 {angle:.1f}度, 半径 {r:.2f}"
            )
        return "\n".join(descriptions)
    
    def create_prompt(self, 
                     current_pos: Tuple[float, float],
                     goal_pos: Tuple[float, float],
                     obstacles: List[Tuple[float, float, float]],
                     scenario: str = 'default') -> str:
        """创建完整的prompt"""
        # 计算到目标的距离
        distance_to_goal = ((goal_pos[0] - current_pos[0])**2 + 
                           (goal_pos[1] - current_pos[1])**2)**0.5
        
        # 获取并填充模板
        template = self.get_template(scenario)
        obstacles_description = self.format_obstacles_description(obstacles, 
                                                               current_pos)
        
        return template.format(
            current_x=current_pos[0],
            current_y=current_pos[1],
            goal_x=goal_pos[0],
            goal_y=goal_pos[1],
            distance_to_goal=distance_to_goal,
            obstacles_description=obstacles_description,
            format_requirements=self.format_requirements
        )