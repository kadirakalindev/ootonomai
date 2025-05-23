# ground_crossing_detector.py

import cv2
import numpy as np

class GroundCrossingDetector:
    def __init__(self, image_width, image_height):
        self.image_width = image_width
        self.image_height = image_height

        # Tespit parametreleri (ayarlanabilir)
        self.roi_y_start_ratio = 0.40  # Görüntünün %40'ından başla (ufuk çizgisine yakın)
        self.roi_y_end_ratio = 0.75    # Görüntünün %75'ine kadar (araca daha yakın)
        self.roi_x_margin_ratio = 0.10 # Kenarlardan %10 boşluk

        # Beyaz renk için HSV aralıkları
        self.beyaz_lower = np.array([0, 0, 160]) # V değeri ışığa göre ayarlanmalı
        self.beyaz_upper = np.array([180, 70, 255])# S değeri de önemli

        # Hough Line Transform parametreleri (ayarlanabilir)
        self.hough_rho = 1                  # Piksel cinsinden rho doğruluğu
        self.hough_theta = np.pi / 180    # Radyan cinsinden theta doğruluğu
        self.hough_threshold = 30         # Bir çizgiyi algılamak için gereken minimum kesişim sayısı
        self.hough_min_line_length = int(self.image_width * 0.3) # Çizginin minimum uzunluğu (ROI genişliğinin %30'u)
        self.hough_max_line_gap = int(self.image_width * 0.05)   # Aynı çizgi üzerindeki kopukluklara izin verilen maksimum boşluk

        # Kontur analizi parametreleri (Hough'a alternatif veya ek olarak)
        self.contour_min_area_ratio = 0.005 # ROI alanının %0.5'i
        self.contour_aspect_ratio_threshold = 5.0 # Genişlik / Yükseklik
        self.contour_width_roi_ratio_threshold = 0.5 # Kontur genişliğinin ROI genişliğine oranı

        print("Zemin Geçidi Tespit Modülü Başlatıldı.")

    def _get_roi_coords(self):
        y_start = int(self.image_height * self.roi_y_start_ratio)
        y_end = int(self.image_height * self.roi_y_end_ratio)
        x_start = int(self.image_width * self.roi_x_margin_ratio)
        x_end = int(self.image_width * (1 - self.roi_x_margin_ratio))
        return y_start, y_end, x_start, x_end

    def detect_horizontal_line_hough(self, frame_rgb, debug_frame=None):
        """Hough Line Transform kullanarak yatay çizgileri tespit eder."""
        y_start, y_end, x_start, x_end = self._get_roi_coords()
        roi_gray = cv2.cvtColor(frame_rgb[y_start:y_end, x_start:x_end], cv2.COLOR_RGB2GRAY)
        
        if roi_gray.size == 0: return False

        edges = cv2.Canny(roi_gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, self.hough_rho, self.hough_theta,
                                self.hough_threshold, minLineLength=self.hough_min_line_length,
                                maxLineGap=self.hough_max_line_gap)
        
        found_horizontal_line = False
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Açıyı hesapla (çok dikey olmayanları al)
                angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi
                if abs(angle) < 15 or abs(angle - 180) < 15 or abs(angle + 180) < 15: # Yataya yakın çizgiler
                    # Çizginin ROI içinde ne kadar yayıldığını kontrol et
                    line_width = abs(x2-x1)
                    if line_width > (x_end - x_start) * 0.6: # ROI genişliğinin %60'ından fazlaysa
                        found_horizontal_line = True
                        if debug_frame is not None:
                            # Orijinal frame koordinatlarına çevir
                            cv2.line(debug_frame, (x1 + x_start, y1 + y_start), (x2 + x_start, y2 + y_start), (255, 0, 255), 2) # Magenta
                        # print(f"Hough ile yatay çizgi bulundu: Açı={angle:.1f}, Genişlik={line_width}")
                        break # Bir tane bulmak yeterli
        
        if debug_frame is not None:
            cv2.rectangle(debug_frame, (x_start, y_start), (x_end, y_end), (0, 255, 255), 1) # ROI
            # cv2.imshow("Hough Edges", edges) # Kenarları görmek için

        return found_horizontal_line

    def detect_horizontal_line_contour(self, frame_rgb, debug_frame=None):
        """Kontur analizi ile yatay beyaz çizgileri tespit eder."""
        y_start, y_end, x_start, x_end = self._get_roi_coords()
        roi_rgb = frame_rgb[y_start:y_end, x_start:x_end]

        if roi_rgb.size == 0: return False

        hsv_roi = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
        mask_beyaz = cv2.inRange(hsv_roi, self.beyaz_lower, self.beyaz_upper)
        
        # Morfolojik operasyonlar
        kernel_open = np.ones((3,5),np.uint8) # Yatay gürültüyü birleştirmek için
        kernel_close = np.ones((5,15),np.uint8) # Yatay çizgileri birleştirmek için
        mask_beyaz = cv2.morphologyEx(mask_beyaz, cv2.MORPH_OPEN, kernel_open)
        mask_beyaz = cv2.morphologyEx(mask_beyaz, cv2.MORPH_CLOSE, kernel_close)
        
        contours, _ = cv2.findContours(mask_beyaz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        found_horizontal_line = False
        if contours:
            # ROI alanına göre filtrele
            roi_area_total = roi_rgb.shape[0] * roi_rgb.shape[1]
            min_abs_area = roi_area_total * self.contour_min_area_ratio

            for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
                area = cv2.contourArea(cnt)
                if area < min_abs_area: continue # Çok küçükse atla

                x_c, y_c, w_c, h_c = cv2.boundingRect(cnt)
                aspect_ratio = w_c / (h_c + 1e-6)
                width_roi_ratio = w_c / (roi_rgb.shape[1] + 1e-6)

                # print(f"Kontur: Alan={area:.0f}, AR={aspect_ratio:.1f}, WRR={width_roi_ratio:.2f}")

                if aspect_ratio > self.contour_aspect_ratio_threshold and \
                   width_roi_ratio > self.contour_width_roi_ratio_threshold:
                    found_horizontal_line = True
                    if debug_frame is not None:
                        # Orijinal frame koordinatlarına çevir
                        cv2.rectangle(debug_frame, (x_c + x_start, y_c + y_start), \
                                      (x_c + x_start + w_c, y_c + y_start + h_c), (0, 255, 0), 2) # Yeşil
                    break # Bir tane bulmak yeterli
        
        if debug_frame is not None:
            cv2.rectangle(debug_frame, (x_start, y_start), (x_end, y_end), (0, 255, 255), 1) # ROI
            # cv2.imshow("Beyaz Maske (Geçit)", mask_beyaz)

        return found_horizontal_line

    def detect_crossing(self, frame_rgb, debug_frame=None, method="hough"):
        """Belirtilen metodu kullanarak zemin geçidini tespit eder."""
        if method == "hough":
            return self.detect_horizontal_line_hough(frame_rgb, debug_frame)
        elif method == "contour":
            return self.detect_horizontal_line_contour(frame_rgb, debug_frame)
        else:
            # Varsayılan olarak kontur kullanalım veya ikisini birleştirelim
            return self.detect_horizontal_line_contour(frame_rgb, debug_frame)


if __name__ == "__main__":
    from picamera2 import Picamera2
    import time

    CAM_WIDTH, CAM_HEIGHT = 640, 480
    picam2_test = Picamera2()
    config_test = picam2_test.create_video_configuration(main={"size": (CAM_WIDTH, CAM_HEIGHT), "format": "RGB888"})
    picam2_test.configure(config_test)
    picam2_test.start()
    time.sleep(1)

    crossing_detector = GroundCrossingDetector(image_width=CAM_WIDTH, image_height=CAM_HEIGHT)
    print("Zemin Geçidi Tespit Testi Başlatılıyor. Çıkmak için Ctrl+C.")
    print("Kameraya yaya geçidi veya benzeri yatay bir çizgi gösterin.")

    use_method = "contour" # "hough" veya "contour"

    try:
        while True:
            frame = picam2_test.capture_array("main")
            if frame is None: continue

            debug_display_frame = frame.copy()
            is_crossing_detected = crossing_detector.detect_crossing(frame, debug_display_frame, method=use_method)

            if is_crossing_detected:
                cv2.putText(debug_display_frame, "ZEMIN GECIDI TESPIT EDILDI!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                print("Zemin Geçidi Tespit Edildi!")

            cv2.imshow(f"Zemin Geçidi Tespiti ({use_method})", debug_display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'): # Metot değiştir
                use_method = "hough" if use_method == "contour" else "contour"
                print(f"Tespit metodu: {use_method}")

    except KeyboardInterrupt:
        print("Program sonlandırılıyor.")
    finally:
        picam2_test.stop()
        cv2.destroyAllWindows()
        print("Test sonlandırıldı.")