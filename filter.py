import cv2
import tensorflow as tf
import numpy as np
import mediapipe as mp

# 加载模型
model = tf.keras.models.load_model('hand_gesture_model_optimized')

# 初始化MediaPipe手部检测
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    # 读取摄像头图像
    ret, frame = cap.read()

    # 使用MediaPipe检测手
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # 检查是否检测到手
    if results.multi_hand_landmarks:
        # 绘制手部关键点和线条
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # 获取手的边界框
        h, w, c = frame.shape
        hand_landmarks = results.multi_hand_landmarks[0]
        x_min = w
        x_max = 0
        y_min = h
        y_max = 0
        for landmark in hand_landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            x_min = min(x_min, x)
            x_max = max(x_max, x)
            y_min = min(y_min, y)
            y_max = max(y_max, y)
        
        # 获取手的边界框并稍微扩大一些
        padding = 20
        x_min = max(0, x_min - padding)
        x_max = min(w, x_max + padding)
        y_min = max(0, y_min - padding)
        y_max = min(h, y_max + padding)
        
        # 裁剪手的图像
        hand_img = frame[y_min:y_max, x_min:x_max]
        
        # 预处理图像以供模型使用
        img = cv2.resize(hand_img, (96, 96))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # 使用模型进行预测
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions[0])

           # 根据预测结果添加滤镜
        if predicted_class == 10:  # 绿色滤镜
            frame[:, :, 0] = 0
            frame[:, :, 2] = 0
        elif predicted_class == 1:  # 红色滤镜
            frame[:, :, 0] = 0
            frame[:, :, 1] = 0
            

        # 打印预测概率
        print(predictions[0])
        
        # 显示处理后的图像
        cv2.putText(frame, f'Predicted class: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 显示图像
    cv2.imshow('Hand Gesture Recognition', frame)

    # 按下'q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 关闭摄像头和OpenCV窗口
cap.release()
cv2.destroyAllWindows()
