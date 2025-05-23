# obstacle_detector.py

import cv2
import numpy as np

class ObstacleDetector:
    def __init__(self):
        print("Engel Tespit Modülü Başlatılıyor...")
        self.turuncu_lower = np.array([5, 100, 100])
        self.turuncu_upper = np.array([20, 255, 255])
        self.sari_lower = np.array([20, 100, 100])
        self.sari_upper = np.array([35, 255, 255])

        self.min_contour_area_ratio = 0.005
        self.max_contour_area_ratio = 0.70
        self.approach_threshold_ratio_turuncu = 0.15
        self.approach_threshold_ratio_sari = 0.10

        self.roi_y_start_ratio = 0.45
        self.roi_y_end_ratio = 0.95
        self.roi_x_margin_ratio = 0.05

    def _get_roi_coords(self, frame_shape):
        h, w = frame_shape[:2]
        return int(h*self.roi_y_start_ratio), int(h*self.roi_y_end_ratio), \
               int(w*self.roi_x_margin_ratio), int(w*(1-self.roi_x_margin_ratio))

    def _is_obstacle_in_center_region(self, obs_center_x_in_roi, roi_w):
        return roi_w*0.25 < obs_center_x_in_roi < roi_w*0.75

    def _is_obstacle_on_left_of_center(self, obs_center_x_in_roi, roi_w):
        return obs_center_x_in_roi < roi_w * 0.55

    def find_obstacles(self, input_frame_rgb, lane_info=None): # Parametre adı güncellendi
        output_display_frame = input_frame_rgb.copy() # Gelen frame'in kopyası üzerine çiz
        
        hsv_original = cv2.cvtColor(input_frame_rgb, cv2.COLOR_RGB2HSV)

        ys, ye, xs, xe = self._get_roi_coords(input_frame_rgb.shape)
        roi_hsv_for_detection = hsv_original[ys:ye, xs:xe]
        
        if roi_hsv_for_detection.size == 0:
            return [], [], output_display_frame 
            
        roi_area_total = (ye - ys) * (xe - xs)
        roi_width_px = (xe - xs)

        cv2.rectangle(output_display_frame, (xs, ys), (xe, ye), (255, 255, 0), 1)

        turuncu_list, sari_list = [], []

        # Turuncu Engel
        mask_t = cv2.inRange(roi_hsv_for_detection, self.turuncu_lower, self.turuncu_upper)
        mask_t = cv2.erode(mask_t, None, iterations=1); mask_t = cv2.dilate(mask_t, None, iterations=2)
        contours_t, _ = cv2.findContours(mask_t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_t:
            area = cv2.contourArea(cnt)
            ratio = area/roi_area_total if roi_area_total > 0 else 0
            if self.min_contour_area_ratio < ratio < self.max_contour_area_ratio:
                x_r, y_r, w_r, h_r = cv2.boundingRect(cnt)
                gx, gy = x_r+xs, y_r+ys
                cx_roi = x_r+w_r//2
                
                is_appr = ratio > self.approach_threshold_ratio_turuncu
                is_in_lane = self._is_obstacle_in_center_region(cx_roi, roi_width_px)
                if is_in_lane:
                    turuncu_list.append({'rect':(gx,gy,w_r,h_r), 'area_ratio':ratio, 'is_approaching':is_appr, 'is_in_my_lane':is_in_lane})
                    clr = (0,100,255) if is_appr else (0,165,255)
                    cv2.rectangle(output_display_frame, (gx,gy), (gx+w_r,gy+h_r), clr, 2)
                    cv2.putText(output_display_frame, f"T{ratio:.2f}",(gx,gy-5), cv2.FONT_HERSHEY_SIMPLEX,0.4,clr,1)
        
        # Sarı Engel
        mask_s = cv2.inRange(roi_hsv_for_detection, self.sari_lower, self.sari_upper)
        mask_s = cv2.erode(mask_s, None, iterations=1); mask_s = cv2.dilate(mask_s, None, iterations=2)
        contours_s, _ = cv2.findContours(mask_s, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_s:
            area = cv2.contourArea(cnt)
            ratio = area/roi_area_total if roi_area_total > 0 else 0
            if self.min_contour_area_ratio < ratio < self.max_contour_area_ratio:
                x_r,y_r,w_r,h_r = cv2.boundingRect(cnt)
                gx,gy = x_r+xs, y_r+ys
                cx_roi = x_r+w_r//2

                is_on_l = self._is_obstacle_on_left_of_center(cx_roi, roi_width_px)
                is_appr_s = ratio > self.approach_threshold_ratio_sari
                if is_on_l:
                    sari_list.append({'rect':(gx,gy,w_r,h_r), 'area_ratio':ratio, 'is_on_left_lane':is_on_l, 'is_approaching':is_appr_s})
                    clr = (0,200,200) if is_appr_s else (0,255,255)
                    cv2.rectangle(output_display_frame, (gx,gy), (gx+w_r,gy+h_r), clr, 2)
                    cv2.putText(output_display_frame, f"S{ratio:.2f}",(gx,gy-5), cv2.FONT_HERSHEY_SIMPLEX,0.4,clr,1)
                    
        return turuncu_list, sari_list, output_display_frame

if __name__ == "__main__":
    from picamera2 import Picamera2
    import time # time importu eksikti test bloğunda
    CAM_WIDTH, CAM_HEIGHT = 320, 240
    picam2_test = Picamera2()
    config_test = picam2_test.create_video_configuration(main={"size": (CAM_WIDTH, CAM_HEIGHT), "format": "RGB888"})
    picam2_test.configure(config_test); picam2_test.start(); time.sleep(1)
    detector = ObstacleDetector()
    cv2.namedWindow("Obstacle Test", cv2.WINDOW_NORMAL)
    try:
        while True:
            frame = picam2_test.capture_array("main")
            if frame is None: continue
            _, _, display = detector.find_obstacles(frame) # frame'i direkt ver
            cv2.imshow("Obstacle Test", display)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    except Exception as e: print(f"Hata: {e}") # Hata mesajını yazdır
    finally: picam2_test.stop(); cv2.destroyAllWindows()