# -*- coding: utf-8 -*-
"""
Created on Sun May 12 11:32:36 2024

@author: Hao
"""

import cv2
import dlib
import numpy as np

class face_emotion():
    
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()  # 初始化人臉檢測器
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # 初始化特徵點預測器
        self.cnt = 0
        
    def learning_face(self):
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # 顏色字典，用於儲存每種類別對應的顏色
        color_dict = {"Sleeping": (0, 0, 255), "Happy": (255, 187, 0), "Turned Right": (0, 255, 255), "Turned Left": (255, 255, 0), "Looking Forward": (0, 255, 0)}
        
        # 開啟鏡頭
        cap = cv2.VideoCapture(0)
        
        while True:
            # 讀取一幀
            ret, im_rd = cap.read()
            
            if not ret:
                break
            
            # 轉換成灰階圖像
            img_gray = cv2.cvtColor(im_rd, cv2.COLOR_BGR2GRAY)
            
            # 使用人臉檢測器偵測人臉
            faces = self.detector(img_gray, 0)
            
            # 如果偵測到人臉
            if len(faces) != 0:
                for d in faces:
                    # 獲取表情
                    emotion = self.detect_emotion(im_rd, d)
                    
                    # 根據表情選擇框線顏色
                    box_color = color_dict.get(emotion, (255, 255, 255))
                    
                    # 繪製矩形框
                    cv2.rectangle(im_rd, (d.left(), d.top()), (d.right(), d.bottom()), box_color, 2)
                    
                    # 繪製表情文字
                    cv2.putText(im_rd, emotion, (d.left(), d.bottom() + 20), font, 1, box_color, 2)
            
            # 顯示鏡頭即時影像
            cv2.imshow("Face Recognition", im_rd)
            
            # 按下 q 鍵退出循環
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 釋放鏡頭資源
        cap.release()
        cv2.destroyAllWindows()
    
    def detect_emotion(self, im_rd, face):
        shape = self.predictor(im_rd, face)
        
        # 將 shape 轉換為 NumPy 數組
        shape_np = np.array([[p.x, p.y] for p in shape.parts()])
        
        # 檢測眼睛是否閉著
        eye_aspect_ratio = self.eye_aspect_ratio(shape_np)
        
        # 分析表情
        mouth_width = (shape.part(54).x - shape.part(48).x) / (face.right() - face.left())
        mouth_height = (shape.part(66).y - shape.part(62).y) / (face.bottom() - face.top())
        
        brow_sum = 0
        frown_sum = 0
        for j in range(17, 21):
            brow_sum += (shape.part(j).y - face.top()) + (shape.part(j + 5).y - face.top())
            frown_sum += shape.part(j + 5).x - shape.part(j).x
            
        brow_height = (brow_sum / 10) / (face.bottom() - face.top())
        brow_width = (frown_sum / 5) / (face.right() - face.left())
        
        # 判斷表情
        if eye_aspect_ratio < 0.2 and mouth_height < 0.03:
            return "Sleeping"
        elif mouth_height >= 0.03:
            if eye_aspect_ratio >= 0.056:
                return "Happy"
            # elif 判斷是否還有其他表情...
        else:
            # 檢測眼睛和鼻子之間的相對位置
            nose_x = shape.part(30).x
            left_eye_x = shape.part(36).x
            right_eye_x = shape.part(45).x
            
            # 計算眼睛和鼻子之間的相對位置
            eye_nose_diff = abs(nose_x - (left_eye_x + right_eye_x) / 2)
            
            # 判斷是否在看前方
            if eye_nose_diff < 10:
                return "Looking Forward"
            else:
                # 檢測頭部轉向
                left_diff = abs(nose_x - left_eye_x)
                right_diff = abs(nose_x - right_eye_x)
            
                # 判斷頭部轉向
                if left_diff < right_diff:
                    return "Turned Right"
                else:
                    return "Turned Left"
    
    def eye_aspect_ratio(self, shape):
        # 定義用於計算眼睛長寬比的索引
        (left_eye_start, left_eye_end) = (36, 42)
        (right_eye_start, right_eye_end) = (42, 48)
    
        # 分別計算左右眼的關鍵點座標
        left_eye_pts = shape[left_eye_start:left_eye_end]
        right_eye_pts = shape[right_eye_start:right_eye_end]
    
        # 計算左眼的長寬比
        left_eye_ear = self.eye_aspect_ratio_calculator(left_eye_pts)
        
        # 計算右眼的長寬比
        right_eye_ear = self.eye_aspect_ratio_calculator(right_eye_pts)
    
        # 返回平均長寬比
        return (left_eye_ear + right_eye_ear) / 2
    
    def eye_aspect_ratio_calculator(self, eye_pts):
        # 分別計算眼睛的垂直距離和水平距離
        vertical_dist = np.linalg.norm(eye_pts[1] - eye_pts[5])
        horizontal_dist_1 = np.linalg.norm(eye_pts[0] - eye_pts[3])
        horizontal_dist_2 = np.linalg.norm(eye_pts[1] - eye_pts[4])
    
        # 計算長寬比
        ear = (vertical_dist) / ((horizontal_dist_1 + horizontal_dist_2) / 2)
    
        return ear
        
if __name__ == "__main__":
    my_face = face_emotion()
    my_face.learning_face()



