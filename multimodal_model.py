# multimodal_model.py
import base64
import io
import requests
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from dataclasses import dataclass
from PIL import Image

@dataclass
class ModelFeedback:
    sectors: List[Dict]
    confidence: float
    suggested_step_size: float

class MultiModalInteraction:
    def __init__(self, model_config: dict):
        self.config = model_config
        self.api_key = model_config['api_key']
        self.api_base = "https://api.openai.com/v1/chat/completions"
        self.history = []
        
    def create_scene_image(self, current_node, goal_node, nodes, obstacles, bounds) -> Image:
        """创建当前场景的图像"""
        # 创建新的图像
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        
        # 绘制所有节点和连接
        for node in nodes[1:]:
            if node.parent_id is not None:
                parent = nodes[node.parent_id]
                ax.plot([parent.x, node.x], [parent.y, node.y], 
                       'g-', alpha=0.5, linewidth=0.5)
        
        # 绘制障碍物
        for ox, oy, r in obstacles:
            circle = plt.Circle((ox, oy), r, color='red', alpha=0.3)
            ax.add_patch(circle)
        
        # 绘制起点和终点
        ax.plot(nodes[0].x, nodes[0].y, 'go', markersize=10, label='Start')
        ax.plot(goal_node.x, goal_node.y, 'ro', markersize=10, label='Goal')
        
        # 突出显示当前节点
        ax.plot(current_node.x, current_node.y, 'bo', markersize=8, label='Current')
        
        # 添加网格和图例
        ax.grid(True)
        ax.legend()
        ax.set_aspect('equal')
        
        # 将图像转换为PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        
        return Image.open(buf)
    
    def encode_image(self, image: Image) -> str:
        """将PIL Image转换为base64编码"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def create_prompt(self, current_node, goal_node) -> str:
        """创建用于查询多模态模型的prompt"""
        return f"""
        你现在是一个路径规划专家，我向你展示了一个2D环境中的路径规划场景。
        图中:
        - 绿点表示起点
        - 红点表示终点
        - 蓝点表示当前位置
        - 红色圆形区域表示障碍物
        - 绿色细线表示已探索的路径
        
        请基于当前场景，提供以下建议：
        1. 分析从当前位置（蓝点）到目标（红点）的最佳路径方向
        2. 推荐2-3个优先探索的扇形区域，包括：
           - 扇形的中心角度（0-360度，0度为正x轴）
           - 扇形的张角（建议范围）
           - 每个扇形区域的探索优先级（0-1之间）
        3. 建议的步长大小（考虑障碍物密度）
        
        请按以下JSON格式返回建议：
        {
            "sectors": [
                {
                    "center_angle": float,  // 扇形中心角度
                    "span_angle": float,    // 扇形张角
                    "priority": float       // 优先级(0-1)
                }
            ],
            "suggested_step_size": float,   // 建议步长
            "confidence": float             // 建议置信度(0-1)
        }
        """
    
    def get_feedback(self, current_node, goal_node, nodes, obstacles, bounds) -> List[Dict]:
        """获取多模态模型的反馈"""
        # 生成场景图像
        scene_image = self.create_scene_image(
            current_node, goal_node, nodes, obstacles, bounds)
        base64_image = self.encode_image(scene_image)
        
        # 准备API请求
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.create_prompt(current_node, goal_node)
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 1000
        }
        
        try:
            response = requests.post(self.api_base, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # 解析模型返回的JSON响应
            feedback_str = result['choices'][0]['message']['content']
            feedback_dict = eval(feedback_str)  # 注意：实际使用时应该用json.loads并添加错误处理
            
            return feedback_dict['sectors']
            
        except Exception as e:
            print(f"Error getting model feedback: {e}")
            # 返回默认的扇区设置
            return [
                {
                    "center_angle": np.degrees(np.arctan2(
                        goal_node.y - current_node.y,
                        goal_node.x - current_node.x
                    )),
                    "span_angle": 45,
                    "priority": 0.8
                }
            ]
    
    def _convert_to_sector_config(self, sectors: List[Dict]) -> List[SectorConfig]:
        """将API返回的扇区信息转换为SectorConfig对象"""
        return [SectorConfig(
            center_angle=sector['center_angle'],
            span_angle=sector['span_angle'],
            priority=sector['priority']
        ) for sector in sectors]