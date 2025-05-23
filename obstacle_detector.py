# obstacle_detector.py

import cv2
import numpy as np

class ObstacleDetector:
    def __init__(self):
        print("Engel Tespit Modülü Başlatılıyor...")
        self.turuncu_lower = np.array([5, 90, 90])
        self.turuncu_upper = np.array([25, 255, 255])
        self.sari_lower = np.array([18, 90, 90])
        self.sari_upper = np.array([35, 255, 255])

        self.min_contour_area_ratio = 0.003 
        self.max_contour_area_ratio = 0.85 # Çok büyük bir alanı kaplayabilir (kamerayı doldurma durumu)
        
        # !!! YAKLAŞMA EŞİĞİ (ÖNEMLİ - İSTEĞİNİZE GÖRE AYARLANACAK) !!!
        # "Kameranın büyük bir bölümünü kapladığında" sollama için bu değeri artırın.
        # Örnek: ROI'nin %30'unu, %40'ını veya %50'sini kaplarsa. Test ederek bulun!
        self.approach_threshold_ratio_turuncu = 0.30 # Başlangıç için %30 deneyelim
        # Alternatif: Engelin bounding box yüksekliği de kontrol edilebilir.
        # self.approach_min_height_ratio_turuncu = 0.40 # Frame yüksekliğinin %40'ı gibi

        self.approach_threshold_ratio_sari = 0.06    

        self.roi_y_start_ratio = 0.40 
        self.roi_y_end_ratio = 0.95 # Neredeyse en alta kadar bak (engel yakınsa alt kısmı kaplar)
        self.roi_x_margin_ratio = 0.10 # Kenarlardan biraz daha geniş bakabiliriz
        print(f"ObstacleDetector: Turuncu yaklaşma eşiği = {self.approach_threshold_ratio_turuncu}")

    def _get_roi_coords(self, frame_shape):
        height, width = frame_shape[:2]
        y_start = int(height * self.roi_y_start_ratio)
        y_end = int(height * self.roi_y_end_ratio)
        x_start = int(width * self.roi_x_margin_ratio)
        x_end = int(width * (1 - self.roi_x_margin_ratio))
        return y_start, y_end, x_start, x_end

    def _is_obstacle_in_center_region(self, obs_center_x_in_roi, roi_width_px):
        # Engelin x koordinatı ROI'nin genişliğinin %25-%75 aralığındaysa merkezde kabul edelim.
        return roi_width_px * 0.25 < obs_center_x_in_roi < roi_width_px * 0.75

    def _is_obstacle_on_left_of_center(self, obs_center_x_in_roi, roi_width_px):
        return obs_center_x_in_roi < roi_width_px * 0.60 # Sol yarıdan biraz daha toleranslı

    def find_obstacles(self, input_frame_rgb, lane_info=None):
        output_display_frame = input_frame_rgb.copy()
        frame_height_original = input_frame_rgb.shape[0] # Orijinal frame yüksekliği
        hsv_original = cv2.cvtColor(input_frame_rgb, cv2.COLOR_RGB2HSV)

        ys, ye, xs, xe = self._get_roi_coords(input_frame_rgb.shape)
        roi_hsv_for_detection = hsv_original[ys:ye, xs:xe]
        
        if roi_hsv_for_detection.size == 0:
            return [], [], output_display_frame 
            
        roi_area_total = (ye - ys) * (xe - xs)
        roi_width_px = (xe - xs)

        cv2.rectangle(output_display_frame, (xs, ys), (xe, ye), (200, 200, 0), 1)

        turuncu_list, sari_list = [], []

        # Turuncu Engel Tespiti
        mask_t = cv2.inRange(roi_hsv_for_detection, self.turuncu_lower, self.turuncu_upper)
        kernel_morph = np.ones((3,3), np.uint8)
        mask_t = cv2.erode(mask_t, kernel_morph, iterations=1)
        mask_t = cv2.dilate(mask_t, kernel_morph, iterations=2)
        contours_t, _ = cv2.findContours(mask_t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours_t:
            area = cv2.contourArea(cnt)
            ratio_roi = area / roi_area_total if roi_area_total > 0 else 0

            if self.min_contour_area_ratio < ratio_roi < self.max_contour_area_ratio:
                x_r, y_r, w_r, h_r = cv2.boundingRect(cnt)
                gx, gy = x_r + xs, y_r + ys
                center_x_roi = x_r + w_r // 2
                
                # Alan oranına göre yaklaşma
                is_approaching_by_area = ratio_roi > self.approach_threshold_ratio_turuncu
                
                # İsteğe bağlı: Yükseklik oranına göre yaklaşma
                # obstacle_height_ratio_frame = h_r / frame_height_original
                # is_approaching_by_height = obstacle_height_ratio_frame > self.approach_min_height_ratio_turuncu
                # is_approaching = is_approaching_by_area and is_approaching_by_height # İki koşul da sağlanmalı
                is_approaching = is_approaching_by_area # Şimdilik sadece alan

                is_in_my_lane = self._is_obstacle_in_center_region(center_x_roi, roi_width_px)

                if is_in_my_lane:
                    turuncu_list.append({
                        'rect': (gx, gy, w_r, h_r),
                        'area_ratio': ratio_roi,
                        'is_approaching': is_approaching,
                        'is_in_my_lane': is_in_my_lane
                    })
                    color = (0, 0, 255) if is_approaching else (0, 165, 255) # Yaklaşana kırmızı
                    cv2.rectangle(output_display_frame, (gx, gy), (gx + w_r, gy + h_r), color, 2)
                    cv2.putText(output_display_frame, f"T {ratio_roi:.2f}{' APP' if is_approaching else ''}", (gx, gy - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Sarı Engel Tespiti (aynı kalabilir)
        # ... (önceki tam koddaki gibi) ...
        mask_s = cv2.inRange(roi_hsv_for_detection, self.sari_lower, self.sari_upper)
        mask_s = cv2.erode(mask_s, kernel_morph, iterations=1); mask_s = cv2.dilate(mask_s, kernel_morph, iterations=2)
        contours_s, _ = cv2.findContours(mask_s, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_s:
            area = cv2.contourArea(cnt); ratio = area/roi_area_total if roi_area_total > 0 else 0
            if self.min_contour_area_ratio < ratio < self.max_contour_area_ratio:
                x_r,y_r,w_r,h_r = cv2.boundingRect(cnt); gx,gy=x_r+xs,y_r+ys; cx_roi=x_r+w_r//2
                is_on_l=self._is_obstacle_on_left_of_center(cx_roi,roi_width_px)
                is_appr_s=ratio > self.approach_threshold_ratio_sari
                if is_on_l:
                    sari_list.append({'rect':(gx,gy,w_r,h_r),'area_ratio':ratio,'is_on_left_lane':is_on_l,'is_approaching':is_appr_s})
                    clr=(0,200,200)if is_appr_s else(0,255,255)
                    cv2.rectangle(output_display_frame,(gx,gy),(gx+w_r,gy+h_r),clr,2)
                    cv2.putText(output_display_frame,f"S{ratio:.2f}",(gx,gy-5),cv2.FONT_HERSHEY_SIMPLEX,0.4,clr,1)

        return turuncu_list, sari_list, output_display_frame

if __name__ == "__main__":
    # ... (Test bloğu aynı kalabilir, yeni approach_threshold_ratio_turuncu değerini test edin)
    from picamera2 import Picamera2; import time
    CAM_W,CAM_H=320,240; picam2=Picamera2(); conf=picam2.create_preview_configuration(main={"size":(CAM_W,CAM_H),"format":"RGB888"})
    picam2.configure(conf);picam2.start();time.sleep(1.5); detector=ObstacleDetector()
    cv2.namedWindow("Obs Test",cv2.WINDOW_NORMAL); print(f"T.Yaklaşma Eşik:{detector.approach_threshold_ratio_turuncu}")
    try:
        while True:
            frm=picam2.capture_array("main");
            if frm is None:continue
            t_obs,s_obs,disp=detector.find_obstacles(frm)
            # ... (test bloğundaki printler aynı)
            cv2.imshow("Obs Test",disp)
            if cv2.waitKey(1)&0xFF==ord('q'):break
    finally:picam2.stop();cv2.destroyAllWindows()