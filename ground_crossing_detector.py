# ground_crossing_detector.py

import cv2
import numpy as np

class GroundCrossingDetector:
    def __init__(self, image_width, image_height):
        self.image_width = image_width
        self.image_height = image_height

        # ROI: Geçitlerin daha ileride ve geniş bir alanı kaplayacak şekilde (AYARLANMALI)
        self.roi_y_start_ratio = 0.35  # Görüntünün %35'inden başla
        self.roi_y_end_ratio = 0.70    # Görüntünün %70'ine kadar
        self.roi_x_margin_ratio = 0.05 # Kenarlardan %5 boşluk (pist genişliğini kapsasın)

        # Beyaz renk için HSV aralıkları (AYARLANMALI)
        self.beyaz_lower = np.array([0, 0, 135]) # V min biraz daha düşük
        self.beyaz_upper = np.array([180, 95, 255])# S max biraz daha yüksek

        # Hough Line Transform parametreleri (AYARLANMALI)
        self.hough_rho = 1
        self.hough_theta = np.pi / 180
        self.hough_threshold = 18 # Daha hassas olması için düşürüldü
        self.hough_min_line_length = int(self.image_width * 0.18) # Daha kısa çizgiler
        self.hough_max_line_gap = int(self.image_width * 0.18)   # Kopukluklara daha toleranslı

        # Kontur analizi parametreleri (AYARLANMALI)
        self.contour_min_area_ratio = 0.002 # Daha küçük alanlar
        self.contour_aspect_ratio_threshold = 3.0 # En/boy oranı eşiği biraz daha düşük
        self.contour_width_roi_ratio_threshold = 0.30 # ROI'nin %30'unu kaplaması

        # Morfoloji için kerneller (Konturda)
        self.kernel_open_contour = np.ones((2,6), np.uint8) # Yatay açma
        self.kernel_close_contour = np.ones((4,18), np.uint8) # Güçlü yatay kapama

        print("Zemin Geçidi Tespit Modülü Başlatıldı (güncellenmiş ROI ve parametreler).")

    def _get_roi_coords(self):
        h, w = self.image_height, self.image_width
        return int(h*self.roi_y_start_ratio), int(h*self.roi_y_end_ratio), \
               int(w*self.roi_x_margin_ratio), int(w*(1-self.roi_x_margin_ratio))

    def _draw_roi_on_frame(self, frame_to_draw_on, roi_coords_tuple):
        if frame_to_draw_on is not None:
            ys, ye, xs, xe = roi_coords_tuple # Tuple'dan değerleri al
            cv2.rectangle(frame_to_draw_on, (xs,ys), (xe,ye), (50,150,150),1)

    def _draw_detected_feature_on_frame(self, frame_to_draw_on, feature_coords_in_roi, roi_offsets, color=(0,255,0), type="rect"):
        xs_roi, ys_roi = roi_offsets
        if feature_coords_in_roi and frame_to_draw_on is not None:
            if type == "line" and len(feature_coords_in_roi) == 4: # Hough line (x1,y1,x2,y2)
                x1,y1,x2,y2 = feature_coords_in_roi
                cv2.line(frame_to_draw_on, (x1+xs_roi, y1+ys_roi), (x2+xs_roi, y2+ys_roi), color, 3)
            elif type == "rect" and len(feature_coords_in_roi) == 4: # Contour rect (x,y,w,h)
                x,y,w,h = feature_coords_in_roi
                cv2.rectangle(frame_to_draw_on, (x+xs_roi, y+ys_roi), (x+xs_roi+w, y+ys_roi+h), color,2)

    def detect_horizontal_line_hough(self, frame_rgb_for_detection, frame_to_draw_on_if_detected):
        ys, ye, xs, xe = self._get_roi_coords()
        roi_offsets = (xs,ys)
        self._draw_roi_on_frame(frame_to_draw_on_if_detected, (ys, ye, xs, xe)) # ROI çiz

        # ROI'yi kes ve işle
        roi_to_process = frame_rgb_for_detection[ys:ye, xs:xe]
        if roi_to_process.size == 0: return False
        
        roi_gray = cv2.cvtColor(roi_to_process, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(roi_gray, 45, 135, apertureSize=3) # Canny eşikleri ayarlanabilir
        
        lines = cv2.HoughLinesP(edges, self.hough_rho,self.hough_theta,self.hough_threshold,
                                minLineLength=self.hough_min_line_length, maxLineGap=self.hough_max_line_gap)
        if lines is not None:
            for line in lines:
                x1,y1,x2,y2 = line[0]
                angle = np.rad2deg(np.arctan2(abs(y2-y1), abs(x2-x1)))
                if angle < 30: # Yataya yakınlık toleransını artır (0-30 derece)
                    # Çizginin ROI genişliğine göre oranını kontrol et
                    if abs(x2-x1) > (xe-xs) * self.contour_width_roi_ratio_threshold : # Kontur parametresini kullanabiliriz
                        self._draw_detected_feature_on_frame(frame_to_draw_on_if_detected, (x1,y1,x2,y2), roi_offsets, color=(255,0,255), type="line")
                        return True
        return False

    def detect_horizontal_line_contour(self, frame_rgb_for_detection, frame_to_draw_on_if_detected):
        ys, ye, xs, xe = self._get_roi_coords()
        roi_offsets = (xs,ys)
        self._draw_roi_on_frame(frame_to_draw_on_if_detected, (ys, ye, xs, xe)) # ROI çiz

        roi_to_process = frame_rgb_for_detection[ys:ye, xs:xe]
        if roi_to_process.size == 0: return False
        
        hsv_roi = cv2.cvtColor(roi_to_process, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv_roi, self.beyaz_lower, self.beyaz_upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open_contour)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close_contour)
        
        # if frame_to_draw_on_if_detected is not None: # Debug için maskeyi de gösterebiliriz
        #    cv2.imshow("ZG Mask", mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            min_abs_area = (ye-ys)*(xe-xs)*self.contour_min_area_ratio
            for cnt in sorted(contours,key=cv2.contourArea,reverse=True):
                if cv2.contourArea(cnt) < min_abs_area: continue # Çok küçükse atla
                
                x_c,y_c,w_c,h_c = cv2.boundingRect(cnt)
                aspect_ratio = w_c / (h_c + 1e-6) # Sıfıra bölme hatası için
                width_roi_ratio = w_c / (roi_to_process.shape[1] + 1e-6) # ROI genişliğine oran

                if aspect_ratio > self.contour_aspect_ratio_threshold and \
                   width_roi_ratio > self.contour_width_roi_ratio_threshold:
                    self._draw_detected_feature_on_frame(frame_to_draw_on_if_detected, (x_c,y_c,w_c,h_c), roi_offsets, color=(0,200,50), type="rect")
                    return True
        return False

    def detect_crossing(self, frame_rgb_for_detection, frame_to_draw_on_if_detected, method="contour"):
        if method == "hough":
            return self.detect_horizontal_line_hough(frame_rgb_for_detection, frame_to_draw_on_if_detected)
        # Varsayılan olarak kontur kullan
        return self.detect_horizontal_line_contour(frame_rgb_for_detection, frame_to_draw_on_if_detected)

if __name__ == "__main__":
    from picamera2 import Picamera2
    import time
    CAM_WIDTH, CAM_HEIGHT = 320, 240
    picam2_test = Picamera2()
    config_test = picam2_test.create_video_configuration(main={"size": (CAM_WIDTH, CAM_HEIGHT), "format": "RGB888"})
    picam2_test.configure(config_test)
    picam2_test.start()
    time.sleep(1.5)

    detector = GroundCrossingDetector(image_width=CAM_WIDTH, image_height=CAM_HEIGHT)
    cv2.namedWindow("Ground Crossing Test", cv2.WINDOW_NORMAL)
    use_method = "contour" # Başlangıç metodu
    print(f"Zemin Geçidi Testi. 'q'-çık, 'm'-metod değiştir. Başlangıç Metodu: {use_method}")

    try:
        while True:
            frame = picam2_test.capture_array("main")
            if frame is None: continue

            display_for_test = frame.copy() # Çizim için her zaman kopya al
            detected = detector.detect_crossing(frame, display_for_test, method=use_method) # frame'i tespit için, display'i çizim için ver

            if detected:
                cv2.putText(display_for_test, "GECIT TESPIT!", (30, CAM_HEIGHT - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            
            cv2.putText(display_for_test, f"Metod: {use_method}", (10,20), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255),1)
            cv2.imshow("Ground Crossing Test", display_for_test)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('m'):
                use_method = "hough" if use_method == "contour" else "contour"
                print(f"Tespit metodu değiştirildi: {use_method}")
    except Exception as e:
        print(f"Test sırasında hata: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if picam2_test.started:
            picam2_test.stop()
        cv2.destroyAllWindows()
        print("Test bitti.")