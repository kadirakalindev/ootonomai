# obstacle_detector.py

import cv2
import numpy as np

class ObstacleDetector:
    def __init__(self):
        print("Engel Tespit Modülü Başlatılıyor...")
        # Turuncu renk için HSV aralıkları (AYARLANMALI)
        self.turuncu_lower = np.array([5, 90, 90])   # S ve V min biraz daha düşük olabilir
        self.turuncu_upper = np.array([25, 255, 255]) # H max biraz daha yüksek olabilir

        # Sarı renk için HSV aralıkları (AYARLANMALI)
        self.sari_lower = np.array([18, 90, 90])
        self.sari_upper = np.array([35, 255, 255])

        # Engel tespiti için parametreler
        self.min_contour_area_ratio = 0.003 # Daha küçük engelleri de yakalamak için
        self.max_contour_area_ratio = 0.60

        # YAKLAŞMA EŞİKLERİ (ÖNEMLİ - AYARLANMALI)
        self.approach_threshold_ratio_turuncu = 0.07 # Örn: ROI'nin %7'sini kaplarsa "yakın"
        self.approach_threshold_ratio_sari = 0.06    # Sarı için biraz daha erken

        # ROI (Engellerin aranacağı bölge - AYARLANMALI)
        self.roi_y_start_ratio = 0.40  # Görüntü yüksekliğinin %40'ından başlasın
        self.roi_y_end_ratio = 0.90    # Neredeyse en alta kadar (%90)
        self.roi_x_margin_ratio = 0.15 # Kenarlardan %15 içerde (şerit içine odaklan)
        print("ObstacleDetector: ROI ve yaklaşma eşikleri güncellendi.")


    def _get_roi_coords(self, frame_shape):
        height, width = frame_shape[:2]
        y_start = int(height * self.roi_y_start_ratio)
        y_end = int(height * self.roi_y_end_ratio)
        x_start = int(width * self.roi_x_margin_ratio)
        x_end = int(width * (1 - self.roi_x_margin_ratio))
        return y_start, y_end, x_start, x_end

    def _is_obstacle_in_center_region(self, obstacle_center_x_in_roi, roi_width_px):
        """Engelin ROI'nin merkezine yakın olup olmadığını kontrol eder."""
        # ROI genişliğinin %30 ile %70 arasındaki bölgesi merkez kabul edilebilir.
        # Bu, engelin kabaca şeridin ortasında olduğunu varsayar.
        # Daha hassas kontrol için şerit takip bilgisi (offset) kullanılabilir.
        center_region_start = roi_width_px * 0.30
        center_region_end = roi_width_px * 0.70
        if center_region_start < obstacle_center_x_in_roi < center_region_end:
            return True
        return False

    def _is_obstacle_on_left_of_center(self, obstacle_center_x_in_roi, roi_width_px):
        """Engelin ROI'nin merkezinin solunda olup olmadığını kontrol eder."""
        # ROI genişliğinin %0 ile %55 arasındaki bölgesi sol kabul edilebilir.
        if obstacle_center_x_in_roi < roi_width_px * 0.55:
            return True
        return False

    def find_obstacles(self, input_frame_rgb, lane_info=None):
        output_display_frame = input_frame_rgb.copy()
        hsv_original = cv2.cvtColor(input_frame_rgb, cv2.COLOR_RGB2HSV)

        ys, ye, xs, xe = self._get_roi_coords(input_frame_rgb.shape)
        roi_hsv_for_detection = hsv_original[ys:ye, xs:xe]
        
        if roi_hsv_for_detection.size == 0:
            return [], [], output_display_frame 
            
        roi_area_total = (ye - ys) * (xe - xs)
        roi_width_px = (xe - xs)

        # ROI'yi output_display_frame üzerine çiz
        cv2.rectangle(output_display_frame, (xs, ys), (xe, ye), (200, 200, 0), 1) # ROI rengi

        turuncu_list, sari_list = [], []

        # Turuncu Engel Tespiti
        mask_t = cv2.inRange(roi_hsv_for_detection, self.turuncu_lower, self.turuncu_upper)
        kernel_morph = np.ones((3,3), np.uint8) # Daha küçük kernel
        mask_t = cv2.erode(mask_t, kernel_morph, iterations=1)
        mask_t = cv2.dilate(mask_t, kernel_morph, iterations=2) # Dilate biraz daha fazla
        contours_t, _ = cv2.findContours(mask_t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours_t:
            area = cv2.contourArea(cnt)
            ratio = area / roi_area_total if roi_area_total > 0 else 0

            if self.min_contour_area_ratio < ratio < self.max_contour_area_ratio:
                x_r, y_r, w_r, h_r = cv2.boundingRect(cnt)
                # Global koordinatlar (output_display_frame için)
                gx, gy = x_r + xs, y_r + ys
                # ROI içindeki merkez x (konum kontrolü için)
                center_x_roi = x_r + w_r // 2
                
                is_approaching = ratio > self.approach_threshold_ratio_turuncu
                is_in_my_lane = self._is_obstacle_in_center_region(center_x_roi, roi_width_px)

                if is_in_my_lane: # Sadece kendi şeridimizdeki (merkezdeki) turuncuları al
                    turuncu_list.append({
                        'rect': (gx, gy, w_r, h_r),
                        'area_ratio': ratio,
                        'is_approaching': is_approaching,
                        'is_in_my_lane': is_in_my_lane # Bu flag main.py'de kullanılabilir
                    })
                    # Çizimler output_display_frame üzerine
                    color = (0, 100, 255) if is_approaching else (0, 165, 255) # Yaklaşana farklı renk
                    cv2.rectangle(output_display_frame, (gx, gy), (gx + w_r, gy + h_r), color, 2)
                    cv2.putText(output_display_frame, f"T {ratio:.2f}", (gx, gy - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Sarı Engel Tespiti
        mask_s = cv2.inRange(roi_hsv_for_detection, self.sari_lower, self.sari_upper)
        mask_s = cv2.erode(mask_s, kernel_morph, iterations=1)
        mask_s = cv2.dilate(mask_s, kernel_morph, iterations=2)
        contours_s, _ = cv2.findContours(mask_s, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours_s:
            area = cv2.contourArea(cnt)
            ratio = area / roi_area_total if roi_area_total > 0 else 0

            if self.min_contour_area_ratio < ratio < self.max_contour_area_ratio:
                x_r, y_r, w_r, h_r = cv2.boundingRect(cnt)
                gx, gy = x_r + xs, y_r + ys
                center_x_roi = x_r + w_r // 2

                is_on_left = self._is_obstacle_on_left_of_center(center_x_roi, roi_width_px)
                is_approaching_sari = ratio > self.approach_threshold_ratio_sari

                if is_on_left: # Sadece sol şeritteki sarıları dikkate al
                    sari_list.append({
                        'rect': (gx, gy, w_r, h_r),
                        'area_ratio': ratio,
                        'is_on_left_lane': is_on_left,
                        'is_approaching': is_approaching_sari
                    })
                    color = (0, 200, 200) if is_approaching_sari else (0, 255, 255)
                    cv2.rectangle(output_display_frame, (gx, gy), (gx + w_r, gy + h_r), color, 2)
                    cv2.putText(output_display_frame, f"S {ratio:.2f}", (gx, gy - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    
        return turuncu_list, sari_list, output_display_frame

if __name__ == "__main__":
    from picamera2 import Picamera2
    import time
    CAM_WIDTH, CAM_HEIGHT = 320, 240 # Test için düşük çözünürlük
    picam2_test = Picamera2()
    config_test = picam2_test.create_video_configuration(main={"size": (CAM_WIDTH, CAM_HEIGHT), "format": "RGB888"})
    picam2_test.configure(config_test)
    picam2_test.start()
    time.sleep(1.5) # Kamera ısınsın

    detector = ObstacleDetector()
    cv2.namedWindow("Obstacle Test", cv2.WINDOW_NORMAL) # Yeniden boyutlandırılabilir pencere
    print("Engel Tespit Testi. 'q' ile çıkın.")
    print(f"Turuncu Yaklaşma Eşiği: {detector.approach_threshold_ratio_turuncu}")
    print(f"ROI Y Başlangıç: {detector.roi_y_start_ratio}, X Marj: {detector.roi_x_margin_ratio}")


    try:
        while True:
            frame = picam2_test.capture_array("main")
            if frame is None: continue

            # find_obstacles, üzerine çizilmiş bir kopya döndürür
            turuncu_found, sari_found, display_img = detector.find_obstacles(frame)
            
            if turuncu_found:
                for obs in turuncu_found:
                    if obs['is_in_my_lane'] and obs['is_approaching']:
                        print(f"Yakın Turuncu Engel (merkezde): Alan={obs['area_ratio']:.3f}")
            if sari_found:
                 for obs in sari_found:
                    if obs['is_on_left_lane'] and obs['is_approaching']:
                        print(f"Yakın Sarı Engel (solda): Alan={obs['area_ratio']:.3f}")


            cv2.imshow("Obstacle Test", display_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"Test sırasında hata: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if picam2_test.started:
            picam2_test.stop()
        cv2.destroyAllWindows()
        print("Test bitti.")