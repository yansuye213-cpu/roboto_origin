import pygame
import math
import random
import os
import signal
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(SCRIPT_DIR, "assets")


def asset_path(filename):
    return os.path.join(ASSET_DIR, filename)


# 初始化Pygame
pygame.init()

# 设置HDMI输出 - 使用全屏模式
screen_info = pygame.display.Info()
screen_width, screen_height = screen_info.current_w, screen_info.current_h
screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
pygame.mouse.set_visible(False)  # 隐藏光标
pygame.display.set_caption("DOLY Eye HDMI")

# 尝试加载眼睛资源，如果失败则使用替代图形
try:
    # 检查文件是否存在
    if not os.path.exists(asset_path('Eye_1_White.png')):
        raise FileNotFoundError("Eye_1_White image not found")
    if not os.path.exists(asset_path('Eye_1_Top.png')):
        raise FileNotFoundError("Eye_1_Top image not found")

    if not os.path.exists(asset_path('Eye_2_Iris.png')):
        raise FileNotFoundError("Eye_2_Iris image not found")
    if not os.path.exists(asset_path('Eye_3_Pupil.png')):
        raise FileNotFoundError("Eye_3_Pupil image not found")
    if not os.path.exists(asset_path('Eye_4_Highlight.png')):
        raise FileNotFoundError("Eye_4_Highlight image not found")
    if not os.path.exists(asset_path('Eye_5_Reflection.png')):
        raise FileNotFoundError("Eye_5_Reflection image not found")

    Eye_1_White_img = pygame.image.load(asset_path('Eye_1_White.png')).convert_alpha() # 使用convert_alpha以支持透明度
    Eye_1_Top_img = pygame.image.load(asset_path('Eye_1_Top.png')).convert_alpha()
    Eye_2_Iris_img = pygame.image.load(asset_path('Eye_2_Iris.png')).convert_alpha()
    Eye_3_Pupil_img = pygame.image.load(asset_path('Eye_3_Pupil.png')).convert_alpha()
    Eye_4_Highlight_img = pygame.image.load(asset_path('Eye_4_Highlight.png')).convert_alpha()
    Eye_5_Reflection_img = pygame.image.load(asset_path('Eye_5_Reflection.png')).convert_alpha()
    print("Successfully loaded eye assets")
except Exception as e:
    print(f"Error loading images: {e}. Using fallback graphics.")
    # 创建替代图形
    Eye_1_White_img = pygame.Surface((300, 300), pygame.SRCALPHA)
    pygame.draw.circle(Eye_1_White_img, (255, 255, 255), (150, 150), 150)  # 白色眼球

    Eye_1_Top_img = pygame.Surface((300, 300), pygame.SRCALPHA)
    pygame.draw.circle(Eye_1_Top_img, (0, 0, 0, 60), (150, 150), 140)  # 半透明黑色阴影
    
    Eye_2_Iris_img = pygame.Surface((120, 120), pygame.SRCALPHA)
    pygame.draw.circle(Eye_2_Iris_img, (200, 200, 200), (150, 100), 60)  # 虹膜
    
    Eye_3_Pupil_img = pygame.Surface((120, 120), pygame.SRCALPHA)
    pygame.draw.circle(Eye_3_Pupil_img, (0, 0, 0), (60, 60), 60)  # 黑色瞳孔
    
    Eye_4_Highlight_img = pygame.Surface((120, 120), pygame.SRCALPHA)
    pygame.draw.circle(Eye_4_Highlight_img, (200, 200, 200), (150, 100), 60)  # 白色高光

    Eye_5_Reflection_img = pygame.Surface((120, 120), pygame.SRCALPHA)
    pygame.draw.circle(Eye_5_Reflection_img, (255, 255, 255), (150, 100), 60)  # 白色反光

# 尝试加载眼皮资源
upper_eyelid_images = []  # 上眼皮图片
lower_eyelid_images = []  # 下眼皮图片
eyelid_types = ["Almond Shape", "Cat Eye Shape", "Round Shape"]

try:
    # 假设你有三对眼皮图片，命名为 upper_eyelid1.png, lower_eyelid1.png, 等等
    for i in range(3):
        upper_eyelid_path = asset_path(f'upper_eyelid{i+1}.png')
        lower_eyelid_path = asset_path(f'lower_eyelid{i+1}.png')
        
        if not os.path.exists(upper_eyelid_path) or not os.path.exists(lower_eyelid_path):
            raise FileNotFoundError(f"Eyelid images {i+1} not found")
        
        # 加载上眼皮图片
        upper_eyelid_img = pygame.image.load(upper_eyelid_path).convert_alpha()
        upper_eyelid_img = pygame.transform.scale(upper_eyelid_img, (Eye_1_White_img.get_width(), Eye_1_White_img.get_height()//2))
        upper_eyelid_images.append(upper_eyelid_img)
        
        # 加载下眼皮图片
        lower_eyelid_img = pygame.image.load(lower_eyelid_path).convert_alpha()
        lower_eyelid_img = pygame.transform.scale(lower_eyelid_img, (Eye_1_White_img.get_width(), Eye_1_White_img.get_height()//2))
        lower_eyelid_images.append(lower_eyelid_img)
        
        print(f"Successfully loaded eyelid set {i+1}")
        
except Exception as e:
    print(f"Error loading eyelid images: {e}. Using fallback eyelids.")
    # 创建替代眼皮图形
    for i in range(3):
        # 创建上眼皮
        upper_eyelid = pygame.Surface((300, 150), pygame.SRCALPHA)
        if i == 0:  # 杏仁形
            pygame.draw.ellipse(upper_eyelid, (30, 30, 30), (50, 50, 200, 100))
        elif i == 1:  # 猫眼形
            pygame.draw.ellipse(upper_eyelid, (30, 30, 30), (25, 50, 250, 100))
        else:  # 圆形
            pygame.draw.ellipse(upper_eyelid, (30, 30, 30), (0, 50, 300, 100))
        upper_eyelid_images.append(upper_eyelid)
        
        # 创建下眼皮
        lower_eyelid = pygame.Surface((300, 150), pygame.SRCALPHA)
        if i == 0:  # 杏仁形
            pygame.draw.ellipse(lower_eyelid, (30, 30, 30), (50, -50, 200, 100))
        elif i == 1:  # 猫眼形
            pygame.draw.ellipse(lower_eyelid, (30, 30, 30), (25, -50, 250, 100))
        else:  # 圆形
            pygame.draw.ellipse(lower_eyelid, (30, 30, 30), (0, -50, 300, 100))
        lower_eyelid_images.append(lower_eyelid)

# 定义状态和参数
current_state = "NORMAL"
blink_timer = 0
blink_duration = 0
is_blinking = False
breath_time = 0  # 呼吸时间计数
gaze_x, gaze_y = 0, 0  # 当前注视点
target_gaze_x, target_gaze_y = 0, 0  # 目标注视点

# 眼睛位置 - 在屏幕中央
eye_radius = min(screen_width, screen_height) // 4 
eye_pos = (screen_width // 2, screen_height // 2)

# 瞳孔移动范围限制
Eye_3_Pupil_move_range = eye_radius * 0.5

# 眼皮类型 (0: 杏仁形, 1: 猫眼形, 2: 圆形)
eyelid_type = 0

# 创建时钟对象来控制帧率
clock = pygame.time.Clock()
FPS = 60  # 设置为240 FPS以获得更流畅的动画

# 主循环
running = True


def request_shutdown(_signum, _frame):
    global running
    running = False


signal.signal(signal.SIGINT, request_shutdown)
signal.signal(signal.SIGTERM, request_shutdown)

while running:
    # 处理事件（如退出）
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False
        elif event.type == pygame.KEYDOWN:
            # 使用下方向键切换眼皮类型
            if event.key == pygame.K_DOWN:
                eyelid_type = (eyelid_type + 1) % 3  # 循环切换 0->1->2->0...
                print(f"Switched to {eyelid_types[eyelid_type]}")
    
    # --- 状态与行为逻辑 --- 
    # 根据当前状态更新参数
    if current_state == "NORMAL":
        # 随机眨眼触发 - 增加眨眼频率
        if not is_blinking and random.random() < 0.01:  # 1%的概率触发一次眨眼
            is_blinking = True
            blink_timer = 0
            blink_duration = random.uniform(0.5, 1)  # 随机眨眼持续时间
            print("Blink started!")
        
        # 每隔一段时间更新注视点
        if random.random() < 0.02:  # 2%的概率更新注视点
            target_gaze_x = random.uniform(-1, 1)
            target_gaze_y = random.uniform(-0.5, 0.5)
    # 平滑过渡到目标注视点
    gaze_x += (target_gaze_x - gaze_x) * 0.1
    gaze_y += (target_gaze_y - gaze_y) * 0.1
    
    # 更新眨眼计时器
    if is_blinking:
        blink_timer += 1/FPS
        if blink_timer >= blink_duration:
            is_blinking = False
            print("Blink completed!")
    
    # --- 渲染逻辑 ---
    screen.fill((30, 30, 30))  # 用深灰色清屏
    
    # 计算眨眼程度 (0.0-1.0, 0是全开，1是全闭)
    blink_progress = 0.0
    if is_blinking:
        # 创建一个平滑的眨眼动画（使用三次函数）
        t = blink_timer / blink_duration
        if t < 0.5:
            blink_progress = 2 * t  # 从0到1
        else:
            blink_progress = 2 * (1 - t)  # 从1到0
    

    
    # 缩放眼睛瞳孔
    # 随机缩放比例
    eye_radius_zoom_param = 1.0 + random.uniform(-0.001, 0.001)
    # 眼睛位置 - 在屏幕中央
    eye_radius_zoom = eye_radius * eye_radius_zoom_param
    # # scaled_Eye_1_White = pygame.transform.scale(Eye_1_White_img, (int(eye_radius), int(eye_radius)))
    # # scaled_Eye_3_Pupil = pygame.transform.scale(Eye_3_Pupil_img, (int(eye_radius_zoom*4), int(eye_radius_zoom*4)))
    # scaled_Eye_3_Pupil = pygame.transform.scale(Eye_3_Pupil_img, (int(eye_radius_zoom*3), int(eye_radius_zoom*3)))


    # 计算虹膜位置和缩放
    iris_offset_x = gaze_x * Eye_3_Pupil_move_range * 0.5 
    iris_offset_y = gaze_y * Eye_3_Pupil_move_range * 0.5
    iris_pos = (eye_pos[0] + iris_offset_x, eye_pos[1] + iris_offset_y)
    iris_scale = int(eye_radius_zoom * 2.8)
    scaled_Eye_2_Iris = pygame.transform.scale(Eye_2_Iris_img, (iris_scale, iris_scale))
    iris_rect = scaled_Eye_2_Iris.get_rect(center=iris_pos)
    
    # 计算阴影层缩放和位置（与虹膜同步）
    top_scale = int(eye_radius_zoom * 4.0)
    scaled_Eye_1_Top = pygame.transform.scale(Eye_1_Top_img, (top_scale, top_scale))
    # top_rect = scaled_Eye_1_Top.get_rect(center=iris_pos)
    # 计算top层中心点：eye_pos和iris_pos之间线性插值，权重0.5
    top_center_x = eye_pos[0] + (iris_pos[0] - eye_pos[0]) * 0.5
    top_center_y = eye_pos[1] + (iris_pos[1] - eye_pos[1]) * 0.5
    top_rect = scaled_Eye_1_Top.get_rect(center=(top_center_x, top_center_y))
        


    # 计算瞳孔位置（限制在合理范围内）
    Eye_3_Pupil_x = gaze_x * Eye_3_Pupil_move_range * 0.5 
    Eye_3_Pupil_y = gaze_y * Eye_3_Pupil_move_range * 0.5 
    pupil_pos = (eye_pos[0] + Eye_3_Pupil_x, eye_pos[1] + Eye_3_Pupil_y)

    # 呼吸时间推进
    breath_time += 1 / FPS
    # 呼吸缩放因子（范围约0.92~1.08，可自行调整幅度和速度）
    breath_scale = 1.0 + 0.02 * math.sin(breath_time * 1.8)  # 1.2为呼吸频率（越小越慢）
    # 计算瞳孔缩放（加入呼吸缩放）
    pupil_scale = int(eye_radius_zoom * 2.5 * breath_scale)
    scaled_Eye_3_Pupil = pygame.transform.scale(Eye_3_Pupil_img, (pupil_scale, pupil_scale))
    pupil_rect = scaled_Eye_3_Pupil.get_rect(center=pupil_pos)

    # 高光位置：在虹膜与瞳孔外径中点构成的圆轨迹上
    iris_radius = iris_scale // 2
    pupil_radius = pupil_scale // 2
    highlight_radius = (iris_radius + pupil_radius) // 2.5  # 取两者半径平均值

    # highlight_angle = math.radians(135)  # 你可以改为其它角度，或随时间/注视点变化

    # 计算高光角度与注视点的关系
    # 归一化 gaze_x, gaze_y 到 [-1, 1]
    norm_gaze_x = gaze_x
    norm_gaze_y = gaze_y * 2  # 因为y范围是-0.5~0.5，放大到-1~1
    # 计算注视方向的极角，atan2(y, x) 角度范围[-pi, pi]
    
    # 计算注视方向的极角
    gaze_angle = math.atan2(-norm_gaze_y, norm_gaze_x)  # [-pi, pi]

    # 将 gaze_angle 线性映射到 45°~135°（π/4~3π/4）
    # 先将 gaze_angle 映射到 [0, 1]
    t = (gaze_angle + math.pi) / (2 * math.pi)
    # 再映射到 [π/4, 3π/4]
    min_angle = math.pi / 4      # 45°
    max_angle = 3 * math.pi / 4  # 135°
    highlight_angle = min_angle + (max_angle - min_angle) * t

    # highlight_angle = math.atan2(-norm_gaze_y, norm_gaze_x)  # 负号让上为正角度

    highlight_pos = (
        int(iris_pos[0] + highlight_radius * math.cos(highlight_angle)),
        int(iris_pos[1] - highlight_radius * math.sin(highlight_angle))
    )
    highlight_scale = int(eye_radius_zoom * 0.8)
    scaled_Eye_4_Highlight = pygame.transform.scale(Eye_4_Highlight_img, (highlight_scale, highlight_scale))
    highlight_rect = scaled_Eye_4_Highlight.get_rect(center=highlight_pos)

    # 反光位置：相对于瞳孔位置有一个小偏移（比如向右下偏移）
    reflection_radius = (iris_radius + pupil_radius) // 2.5  # 取两者半径平均值

    # reflection_angle = math.radians(0)  # 你可以改为其它角度，或随时间/注视点变化
    reflection_angle = highlight_angle+3.14159  # 你可以改为其它角度，或随时间/注视点变化
    reflection_pos = (
        int(iris_pos[0] + reflection_radius * math.cos(reflection_angle)),
        int(iris_pos[1] - reflection_radius * math.sin(reflection_angle))
    )
    reflection_scale = int(eye_radius_zoom * 0.8)
    scaled_Eye_5_Reflection = pygame.transform.scale(Eye_5_Reflection_img, (reflection_scale, reflection_scale))
    reflection_rect = scaled_Eye_5_Reflection.get_rect(center=reflection_pos)

    # 绘制眼睛基底
    eye_rect = Eye_1_White_img.get_rect(center=eye_pos)
    # screen.blit(Eye_1_White_img, eye_rect)
    # eye_rect = scaled_Eye_1_White.get_rect(center=eye_pos)
    # screen.blit(scaled_Eye_1_White, eye_rect)
    # --- 绘制顺序 ---
    screen.blit(Eye_1_White_img, eye_rect)           # 白眼
    screen.blit(scaled_Eye_1_Top, top_rect)          # 眼白阴影（与虹膜同步）
    screen.blit(scaled_Eye_2_Iris, iris_rect)        # 虹膜
    screen.blit(scaled_Eye_3_Pupil, pupil_rect)      # 瞳孔
    screen.blit(scaled_Eye_4_Highlight, highlight_rect)  # 高光
    screen.blit(scaled_Eye_5_Reflection, reflection_rect) # 反光
    
    # 绘制瞳孔
    # Eye_3_Pupil_rect = Eye_3_Pupil_img.get_rect(center=(Eye_3_Pupil_x, Eye_3_Pupil_y))
    # screen.blit(Eye_3_Pupil_img, Eye_3_Pupil_rect)
    # Eye_3_Pupil_rect = scaled_Eye_3_Pupil.get_rect(center=(Eye_3_Pupil_x, Eye_3_Pupil_y))
    # screen.blit(scaled_Eye_3_Pupil, Eye_3_Pupil_rect)
    
    # 使用PNG图案绘制眼皮 - 上下滑移效果
    if blink_progress > 0:
        # 获取当前选择的上下眼皮图片
        current_upper_eyelid = upper_eyelid_images[eyelid_type]
        current_lower_eyelid = lower_eyelid_images[eyelid_type]
        
        # 计算上下眼皮的偏移量
        eyelid_offset = current_upper_eyelid.get_height() * blink_progress
        
        # 绘制上眼皮 - 向下移动
        upper_eyelid_rect = current_upper_eyelid.get_rect(
            centerx=eye_pos[0],
            bottom=eye_rect.top + eyelid_offset
        )
        screen.blit(current_upper_eyelid, upper_eyelid_rect)
        
        # 绘制下眼皮 - 向上移动
        lower_eyelid_rect = current_lower_eyelid.get_rect(
            centerx=eye_pos[0],
            top=eye_rect.bottom - eyelid_offset
        )
        screen.blit(current_lower_eyelid, lower_eyelid_rect)
    
    # # 添加一些调试信息
    # font = pygame.font.Font(None, 36)
    # state_text = font.render(f"State: {current_state}", True, (255, 255, 255))
    # screen.blit(state_text, (20, 20))
    
    # # 显示当前眼皮类型
    # eyelid_text = font.render(f"Eyelid Type: {eyelid_types[eyelid_type]} (Press DOWN to change)", True, (255, 255, 255))
    # screen.blit(eyelid_text, (20, 60))
    
    # # 显示眨眼状态
    # blink_status = "Blinking" if is_blinking else "Not Blinking"
    # blink_text = font.render(f"Blink Status: {blink_status}", True, (255, 255, 255))
    # screen.blit(blink_text, (20, 100))
    
    # 更新屏幕
    pygame.display.flip()
    
    # 控制帧率
    clock.tick(FPS)

# 退出程序
pygame.quit()
