# obstacle_detector.py

import cv2
import numpy as np

class ObstacleDetector:
    def __init__(self):
        print("Engel Tespit Modülü Başlatılıyor...")
        # Turuncu renk için HSV aralıkları (Bu değerler deneme ile bulunmalıdır!)
        # Örnek değerler: (H: 0-179, S: 0-255, V: 0-255)
        self.turuncu_lower = np.array([5, 100, 100])
        self.turuncu_upper = np.array([20, 255, 255]) # Turuncunun tonuna göre ayarla

        # Sarı renk için HSV aralıkları
        self.sari_lower = np.array([20, 100, 100])
        self.sari_upper = np.array([35, 255, 255]) # Sarının tonuna göre ayarla

        # Engel tespiti için parametreler
        self.min_contour_area_ratio = 0.005 # Konturun, ROI alanının en az %0.5'i olması
        self.max_contour_area_ratio = 0.70
        self.approach_threshold_ratio_turuncu = 0.15 # Turuncu engel için ROI'nin %15'ini geçerse "yakın"
        self.approach_threshold_ratio_sari = 0.10 # Sarı engel için (daha uzaktan fark et)

        # Engellerin aranacağı ROI (görüntünün alt ve orta kısımları)
        self.roi_y_start_ratio = 0.45 # Görüntü yüksekliğinin %45'inden başlasın
        self.roi_y_end_ratio = 0.95   # Görüntü yüksekliğinin %95'ine kadar (en alt hariç)
        self.roi_x_margin_ratio = 0.05 # Kenarlardan %5 boşluk

    def _get_roi_coords(self, frame_shape):
        height, width = frame_shape[:2]
        y_start = int(height * self.roi_y_start_ratio)
        y_end = int(height * self.roi_y_end_ratio)
        x_start = int(width * self.roi_x_margin_ratio)
        x_end = int(width * (1 - self.roi_x_margin_ratio))
        return y_start, y_end, x_start, x_end

    def _is_obstacle_in_center_region(self, obstacle_center_x, roi_width_px):
        """Engelin ROI'nin merkez bölgesinde olup olmadığını kontrol eder (turuncu için)."""
        # ROI'nin %30 sol, %40 orta, %30 sağ gibi bölümlere ayırabiliriz.
        # Merkez bölge: roi_width_px * 0.3 ile roi_width_px * 0.7 arası
        center_region_start = roi_width_px * 0.25
        center_region_end = roi_width_px * 0.75
        # obstacle_center_x, ROI içindeki x koordinatı olmalı.
        if center_region_start < obstacle_center_x < center_region_end:
            return True
        return False

    def _is_obstacle_on_left_of_center(self, obstacle_center_x, roi_width_px):
        """Engelin ROI'nin merkezinin solunda olup olmadığını kontrol eder (sarı için)."""
        # Merkez bölge: roi_width_px * 0.5
        if obstacle_center_x < roi_width_px * 0.5: # ROI'nin sol yarısı
            return True
        return False

    def find_obstacles(self, frame_rgb, lane_info=None):
        debug_frame = frame_rgb.copy()
        hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)

        roi_y_start, roi_y_end, roi_x_start, roi_x_end = self._get_roi_coords(frame_rgb.shape)
        roi_frame_rgb = frame_rgb[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        hsv_roi = hsv[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        
        if roi_frame_rgb.size == 0: # ROI alanı boşsa işlem yapma
            return [], [], debug_frame
            
        roi_area_total = roi_frame_rgb.shape[0] * roi_frame_rgb.shape[1]
        roi_width_px = roi_frame_rgb.shape[1]


        cv2.rectangle(debug_frame, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (255, 255, 0), 1) # ROI Çizimi

        turuncu_engeller_list = []
        sari_engeller_list = []

        # 1. Turuncu Engel Tespiti
        mask_turuncu = cv2.inRange(hsv_roi, self.turuncu_lower, self.turuncu_upper)
        mask_turuncu = cv2.erode(mask_turuncu, None, iterations=1)
        mask_turuncu = cv2.dilate(mask_turuncu, None, iterations=2)
        contours_turuncu, _ = cv2.findContours(mask_turuncu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours_turuncu:
            area = cv2.contourArea(cnt)
            area_ratio_roi = area / roi_area_total if roi_area_total > 0 else 0

            if self.min_contour_area_ratio < area_ratio_roi < self.max_contour_area_ratio:
                x_roi, y_roi, w_roi, h_roi = cv2.boundingRect(cnt)
                # Global frame koordinatları
                gx, gy = x_roi + roi_x_start, y_roi + roi_y_start
                center_x_global, center_y_global = gx + w_roi // 2, gy + h_roi // 2
                center_x_roi = x_roi + w_roi //2 # ROI içindeki x merkezi

                is_approaching = area_ratio_roi > self.approach_threshold_ratio_turuncu
                # Turuncu engelin kendi şeridimizde (merkezde) olduğunu varsayalım
                # Daha iyi kontrol için _is_obstacle_in_center_region kullanılabilir
                is_in_my_lane = self._is_obstacle_in_center_region(center_x_roi, roi_width_px)


                if is_in_my_lane : # Sadece kendi şeridimizdeki turuncuları dikkate al
                    turuncu_engeller_list.append({
                        'rect': (gx, gy, w_roi, h_roi),
                        'area_ratio': area_ratio_roi,
                        'center_global': (center_x_global, center_y_global),
                        'is_approaching': is_approaching,
                        'is_in_my_lane': is_in_my_lane
                    })
                    color = (0, 100, 255) if is_approaching else (0, 165, 255)
                    cv2.rectangle(debug_frame, (gx, gy), (gx + w_roi, gy + h_roi), color, 2)
                    cv2.putText(debug_frame, f"T({area_ratio_roi:.2f})", (gx, gy - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # 2. Sarı Engel Tespiti
        mask_sari = cv2.inRange(hsv_roi, self.sari_lower, self.sari_upper)
        mask_sari = cv2.erode(mask_sari, None, iterations=1)
        mask_sari = cv2.dilate(mask_sari, None, iterations=2)
        contours_sari, _ = cv2.findContours(mask_sari, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours_sari:
            area = cv2.contourArea(cnt)
            area_ratio_roi = area / roi_area_total if roi_area_total > 0 else 0

            if self.min_contour_area_ratio < area_ratio_roi < self.max_contour_area_ratio:
                x_roi, y_roi, w_roi, h_roi = cv2.boundingRect(cnt)
                gx, gy = x_roi + roi_x_start, y_roi + roi_y_start
                center_x_global, center_y_global = gx + w_roi // 2, gy + h_roi // 2
                center_x_roi = x_roi + w_roi // 2

                # Sarı engelin sol şeritte (merkezin solunda) olup olmadığını kontrol et
                is_on_left = self._is_obstacle_on_left_of_center(center_x_roi, roi_width_px)
                is_approaching_sari = area_ratio_roi > self.approach_threshold_ratio_sari

                if is_on_left: # Sadece sol şeritteki sarıları dikkate al
                    sari_engeller_list.append({
                        'rect': (gx, gy, w_roi, h_roi),
                        'area_ratio': area_ratio_roi,
                        'center_global': (center_x_global, center_y_global),
                        'is_on_left_lane': is_on_left,
                        'is_approaching': is_approaching_sari
                    })
                    color = (0, 200, 200) if is_approaching_sari else (0, 255, 255)
                    cv2.rectangle(debug_frame, (gx, gy), (gx + w_roi, gy + h_roi), color, 2)
                    cv2.putText(debug_frame, f"S({area_ratio_roi:.2f})", (gx, gy - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        return turuncu_engeller_list, sari_engeller_list, debug_frame

if __name__ == "__main__":
    from picamera2 import Picamera2
    import time

    picam2_test = Picamera2()
    config_test = picam2_test.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
    picam2_test.configure(config_test)
    picam2_test.start()
    time.sleep(1)

    obstacle_detector = ObstacleDetector()
    print("Engel Tespiti Testi Başlatılıyor. Çıkmak için Ctrl+C.")

    try:
        while True:
            frame = picam2_test.capture_array("main")
            if frame is None: continue

            turuncu_obs, sari_obs, processed_frame = obstacle_detector.find_obstacles(frame.copy())

            for obs_t in turuncu_obs:
                if obs_t['is_in_my_lane'] and obs_t['is_approaching']:
                    print(f"Yaklaşan Turuncu Engel (kendi şeridinde): Alan Oranı={obs_t['area_ratio']:.2f}")
            for obs_s in sari_obs:
                if obs_s['is_on_left_lane']: # Zaten find_obstacles içinde bu kontrol var
                     print(f"Sarı Engel (sol şeritte): Alan Oranı={obs_s['area_ratio']:.2f}, Yaklaşıyor mu?: {obs_s['is_approaching']}")


            cv2.imshow("Engel Tespiti (RGB)", processed_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
    except KeyboardInterrupt:
        print("Program sonlandırılıyor.")
    finally:
        picam2_test.stop()
        cv2.destroyAllWindows()
        print("Test sonlandırıldı.")