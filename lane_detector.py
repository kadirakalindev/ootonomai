# lane_detector.py

import cv2
import numpy as np
from picamera2 import Picamera2
from libcamera import controls # Kamera kontrolleri için
import time

class LaneDetector:
    def __init__(self, image_width=640, image_height=480, camera_fps=30):
        print("Şerit Tespit Modülü Başlatılıyor...")
        self.image_width = image_width
        self.image_height = image_height
        self.camera_fps = camera_fps

        self.picam2 = Picamera2()
        # Kamera çözünürlüğü ve FPS'i main.py'den geliyor.
        # İsteğe bağlı: Otomatik pozlama yerine manuel pozlama deneyebilirsiniz
        # (parlama sorunları için)
        # control_settings = {"FrameRate": self.camera_fps}
        control_settings = {"FrameRate": self.camera_fps, "AnalogueGain": 1.5, "ExposureTime": 12000}
        # Eğer parlama çoksa ExposureTime'ı düşürün (örn: 8000-10000), AnalogueGain'i artırın.
        # self.picam2.set_controls({"AeEnable": False}) # Otomatik pozlamayı kapatmak için

        config = self.picam2.create_video_configuration(
            main={"size": (self.image_width, self.image_height), "format": "RGB888"},
            controls=control_settings
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(1.5) # Kameranın stabil hale gelmesi için yeterli süre
        print(f"Kamera {self.image_width}x{self.image_height} @ {self.camera_fps}FPS ile başlatıldı.")

        # --- Perspektif Dönüşümü Parametreleri (AYARLANMALI!) ---
        # ROI'nin alt kenarı (görüntünün en altından ne kadar yukarıda)
        roi_bottom_offset = 5 
        # ROI'nin üst y koordinatı (görüntü yüksekliğinin yüzdesi olarak)
        # Virajlarda daha iyi bir "ileriye bakış" için bu değerle oynayın.
        # Daha düşük değer (örn: 0.50) daha yakına, yüksek değer (örn: 0.65) daha uzağa bakar.
        roi_top_y_ratio = 0.58 
        roi_top_y = int(self.image_height * roi_top_y_ratio)
        
        # ROI'nin üst kenarının x koordinatları için orta noktadan sapma
        # Daha dar bir üst kenar, perspektifte daha uzağa odaklanmayı simüle eder.
        top_x_margin_ratio = 0.10 # Görüntü genişliğinin %10'u
        top_x_margin = int(self.image_width * top_x_margin_ratio)

        # ROI'nin alt kenarının x koordinatları için kenarlardan içe girinti
        bottom_x_margin = 10 # Piksel

        # Kaynak noktalar (orijinal görüntüde) - (sol-üst, sağ-üst, sağ-alt, sol-alt)
        self.src_points = np.float32([
            (self.image_width // 2 - top_x_margin, roi_top_y),
            (self.image_width // 2 + top_x_margin, roi_top_y),
            (self.image_width - bottom_x_margin, self.image_height - roi_bottom_offset),
            (bottom_x_margin, self.image_height - roi_bottom_offset)
        ])
        self.roi_poly_to_draw = np.array([self.src_points], dtype=np.int32) # Çizim için

        # Hedef noktalar (kuşbakışı görüntüde)
        # Kuşbakışı görüntünün genişliği, orijinalin bir oranı olarak
        dst_width_ratio = 0.90 
        self.warped_img_size = (int(self.image_width * dst_width_ratio), self.image_height) # Yükseklik aynı

        self.dst_points = np.float32([
            [0, 0],
            [self.warped_img_size[0] - 1, 0],
            [self.warped_img_size[0] - 1, self.warped_img_size[1] - 1],
            [0, self.warped_img_size[1] - 1]
        ])

        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.Minv = cv2.getPerspectiveTransform(self.dst_points, self.src_points)

        # --- Metre Başına Piksel (KALİBRASYON GEREKTİRİR!) ---
        # İki şerit merkezi arası yaklaşık 0.8m (her şerit 0.4m).
        # Kuşbakışı görüntüde bu 0.8m, kuşbakışı görüntünün genişliğinin (warped_img_size[0])
        # yaklaşık ne kadarına denk geliyor? (örn: %75'i)
        lane_coverage_in_warped_view = 0.75 
        self.xm_per_pix = (0.40 * 2) / (self.warped_img_size[0] * lane_coverage_in_warped_view)
        # Dikey kalibrasyon: Kesikli şeritler 20cm (0.2m). Kuşbakışında bu kaç piksel?
        # Bu değeri kuşbakışı görüntüden ölçerek bulun.
        pixels_for_20cm_vertical = 30 # Örnek değer
        self.ym_per_pix = 0.20 / pixels_for_20cm_vertical

        # Şerit takibi için durum değişkenleri
        self.left_fit = None
        self.right_fit = None
        self.left_fit_cr = None # Metrik uzayda katsayılar
        self.right_fit_cr = None
        self.ploty = None # Y değerleri (polinom çizimi için)

        # Değerleri yumuşatmak için (Exponential Moving Average)
        self.stable_offset_m = 0.0
        self.stable_curvature = float('inf') # Başlangıçta sonsuz (düz)
        self.smoothing_alpha = 0.35 # Daha düşük değer daha fazla yumuşatma (0 < alpha <= 1)

    def capture_frame(self):
        return self.picam2.capture_array("main") # RGB formatında

    def _preprocess_image(self, img_rgb):
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        # Kontrastı artırmak için CLAHE (isteğe bağlı, performansı etkileyebilir)
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6,6))
        # gray = clahe.apply(gray)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0) # Kernel boyutu (3,3) veya (5,5)
        # Adaptif eşikleme: blockSize ve C değerleri ışığa ve kontrasta göre ayarlanmalı.
        # THRESH_BINARY_INV: Siyah zemin, beyaz şeritler için.
        thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 15, 4) # (blockSize, C)
        return thresholded

    def _perspective_warp(self, img_binary):
        return cv2.warpPerspective(img_binary, self.M, self.warped_img_size, flags=cv2.INTER_LINEAR)

    def _find_lane_pixels_sliding_window(self, warped_binary_img):
        # Teşhis için renkli bir kopya oluştur
        out_img_diag = np.dstack((warped_binary_img, warped_binary_img, warped_binary_img))

        histogram = np.sum(warped_binary_img[warped_binary_img.shape[0]//2:, :], axis=0)
        midpoint = np.int32(histogram.shape[0]//2)
        
        # Başlangıç noktalarını bul (önceki fitler varsa onları kullanmayı dene)
        y_base_for_prev_fit = warped_binary_img.shape[0] - 1
        if self.left_fit is not None and self.ploty is not None:
            try: leftx_base = int(np.polyval(self.left_fit, y_base_for_prev_fit))
            except (TypeError, ValueError): leftx_base = np.argmax(histogram[:midpoint])
        else: leftx_base = np.argmax(histogram[:midpoint])

        if self.right_fit is not None and self.ploty is not None:
            try: rightx_base = int(np.polyval(self.right_fit, y_base_for_prev_fit))
            except (TypeError, ValueError): rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        else: rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        nwindows = 10 # Pencere sayısı (virajlar için artırılabilir)
        margin = int(self.warped_img_size[0] * 0.12) # Pencere genişliği
        minpix = 35 # Bir pencerede gereken minimum piksel

        window_height = np.int32(warped_binary_img.shape[0]//nwindows)
        nonzero = warped_binary_img.nonzero()
        nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])

        leftx_current, rightx_current = leftx_base, rightx_base
        left_lane_inds, right_lane_inds = [], []

        for window in range(nwindows):
            win_y_low = warped_binary_img.shape[0]-(window+1)*window_height
            win_y_high = warped_binary_img.shape[0]-window*window_height
            
            win_xleft_low, win_xleft_high = leftx_current-margin, leftx_current+margin
            win_xright_low, win_xright_high = rightx_current-margin, rightx_current+margin

            # Teşhis için pencereleri çiz
            cv2.rectangle(out_img_diag,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0),1)
            cv2.rectangle(out_img_diag,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0),1)

            good_left = ((nonzeroy>=win_y_low)&(nonzeroy<win_y_high)&(nonzerox>=win_xleft_low)&(nonzerox<win_xleft_high)).nonzero()[0]
            good_right = ((nonzeroy>=win_y_low)&(nonzeroy<win_y_high)&(nonzerox>=win_xright_low)&(nonzerox<win_xright_high)).nonzero()[0]
            
            left_lane_inds.append(good_left)
            right_lane_inds.append(good_right)
            
            if len(good_left)>minpix: leftx_current=np.int32(np.mean(nonzerox[good_left]))
            if len(good_right)>minpix: rightx_current=np.int32(np.mean(nonzerox[good_right]))
        
        try: 
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError: # Eğer hiç piksel bulunamazsa boş kalır
            pass

        return nonzerox[left_lane_inds], nonzeroy[left_lane_inds], \
               nonzerox[right_lane_inds], nonzeroy[right_lane_inds], out_img_diag

    def _fit_polynomial(self, leftx, lefty, rightx, righty, warped_shape):
        # self.ploty'yi burada bir kez ayarla, tüm metodlar bunu kullanır
        if self.ploty is None or len(self.ploty) != warped_shape[0]:
            self.ploty = np.linspace(0, warped_shape[0]-1, warped_shape[0])
        
        temp_left_fit, temp_right_fit = None, None
        temp_left_fit_cr, temp_right_fit_cr = None, None
        min_points_for_fit = 30 # Daha güvenilir fit için minimum nokta

        try:
            if len(leftx) > min_points_for_fit and len(lefty) > min_points_for_fit:
                temp_left_fit = np.polyfit(lefty, leftx, 2) # Piksel uzayında
                if self.ym_per_pix > 1e-7 and self.xm_per_pix > 1e-7: # Geçerli metrik oranlar
                    temp_left_fit_cr = np.polyfit(lefty*self.ym_per_pix, leftx*self.xm_per_pix, 2)
            
            if len(rightx) > min_points_for_fit and len(righty) > min_points_for_fit:
                temp_right_fit = np.polyfit(righty, rightx, 2)
                if self.ym_per_pix > 1e-7 and self.xm_per_pix > 1e-7:
                    temp_right_fit_cr = np.polyfit(righty*self.ym_per_pix, rightx*self.xm_per_pix, 2)
        except (np.RankWarning, TypeError, np.linalg.LinAlgError) as e:
            # print(f"Uyarı/Hata (Poli Fit): {e}") # Hata ayıklama için
            pass # Hata durumunda temp değişkenler None kalacak

        # Sadece geçerli yeni fitler varsa self değerlerini güncelle
        if temp_left_fit is not None: self.left_fit = temp_left_fit
        if temp_right_fit is not None: self.right_fit = temp_right_fit
        if temp_left_fit_cr is not None: self.left_fit_cr = temp_left_fit_cr
        if temp_right_fit_cr is not None: self.right_fit_cr = temp_right_fit_cr
        
        # Bu fonksiyon artık self.fitleri güncelliyor, dönüş değeri gerekmeyebilir
        # Ama tutarlılık için döndürelim
        return self.left_fit, self.right_fit, self.left_fit_cr, self.right_fit_cr

    def _calculate_curvature_offset(self, warped_img_shape):
        if self.ploty is None or self.ym_per_pix < 1e-7 or self.xm_per_pix < 1e-7:
            # Gerekli veriler yoksa, önceki stabil değerleri döndür
            return self.stable_curvature, self.stable_offset_m

        # Değerlendirme için y ekseninin en altını (araca en yakın) veya ortasını kullanabiliriz.
        # En alt nokta genellikle daha hassas offset verir, orta nokta eğrilik için daha stabil olabilir.
        y_eval_for_curve_m = np.mean(self.ploty) * self.ym_per_pix # Eğrilik için orta nokta
        y_eval_for_offset_px = warped_img_shape[0] - 1 # Offset için en alt nokta (piksel)

        current_curvature_left, current_curvature_right = float('inf'), float('inf')

        if self.left_fit_cr is not None:
            A, B = self.left_fit_cr[0], self.left_fit_cr[1]
            if abs(A) > 1e-7: # Sıfıra çok yakınsa düz çizgi (sonsuz eğrilik)
                current_curvature_left = ((1 + (2*A*y_eval_for_curve_m + B)**2)**1.5) / np.absolute(2*A)
        
        if self.right_fit_cr is not None:
            A, B = self.right_fit_cr[0], self.right_fit_cr[1]
            if abs(A) > 1e-7:
                current_curvature_right = ((1 + (2*A*y_eval_for_curve_m + B)**2)**1.5) / np.absolute(2*A)

        # Eğrilik değerlerini birleştirme ve yumuşatma
        valid_lc = current_curvature_left != float('inf') and current_curvature_left > 0
        valid_rc = current_curvature_right != float('inf') and current_curvature_right > 0
        
        calculated_cur = float('inf')
        if valid_lc and valid_rc: calculated_cur = (current_curvature_left + current_curvature_right) / 2
        elif valid_lc: calculated_cur = current_curvature_left
        elif valid_rc: calculated_cur = current_curvature_right
        
        # Sonsuz olmayan bir değerse yumuşat, değilse önceki stabil değeri koru
        if calculated_cur != float('inf'):
            self.stable_curvature = self.smoothing_alpha * calculated_cur + \
                                    (1 - self.smoothing_alpha) * self.stable_curvature
        # else: self.stable_curvature (değişmez)

        # Offset hesaplama
        current_offset_m = 0.0
        left_x_bottom, right_x_bottom = None, None

        if self.left_fit is not None:
            left_x_bottom = np.polyval(self.left_fit, y_eval_for_offset_px)
        if self.right_fit is not None:
            right_x_bottom = np.polyval(self.right_fit, y_eval_for_offset_px)
        
        vehicle_center_px = warped_img_shape[1] / 2.0
        
        if left_x_bottom is not None and right_x_bottom is not None:
            lane_center_px = (left_x_bottom + right_x_bottom) / 2.0
            current_offset_m = (vehicle_center_px - lane_center_px) * self.xm_per_pix
        elif left_x_bottom is not None: # Sadece sol şerit varsa
            # Aracın sol şeridin sağında olduğunu varsay (yarım şerit genişliği kadar)
            assumed_lane_center_px = left_x_bottom + (0.40 / self.xm_per_pix) / 2.0
            current_offset_m = (vehicle_center_px - assumed_lane_center_px) * self.xm_per_pix
        elif right_x_bottom is not None: # Sadece sağ şerit varsa
            assumed_lane_center_px = right_x_bottom - (0.40 / self.xm_per_pix) / 2.0
            current_offset_m = (vehicle_center_px - assumed_lane_center_px) * self.xm_per_pix
        # else: current_offset_m = self.stable_offset_m (hiç şerit yoksa önceki offset)
            
        self.stable_offset_m = self.smoothing_alpha * current_offset_m + \
                               (1 - self.smoothing_alpha) * self.stable_offset_m
        
        return self.stable_curvature, self.stable_offset_m

    def _draw_lanes_on_frame(self, frame_to_draw_on, warped_img_ref_shape):
        # Bu fonksiyon frame_to_draw_on'u yerinde günceller.
        cv2.polylines(frame_to_draw_on, [self.roi_poly_to_draw], isClosed=True, color=(0,100,100), thickness=1)

        if self.ploty is None or (self.left_fit is None and self.right_fit is None):
            return # Çizilecek şerit yoksa (ROI zaten çizildi)

        warp_zero = np.zeros(warped_img_ref_shape[:2], dtype=np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        line_thickness = 15 # Çizgi kalınlığı

        pts_left_render, pts_right_render = None, None
        if self.left_fit is not None:
            left_fitx = np.polyval(self.left_fit, self.ploty)
            pts_left_render = np.array([np.transpose(np.vstack([left_fitx, self.ploty]))])
            cv2.polylines(color_warp, np.int32([pts_left_render]), isClosed=False, color=(255,0,100), thickness=line_thickness)
        
        if self.right_fit is not None:
            right_fitx = np.polyval(self.right_fit, self.ploty)
            pts_right_render = np.array([np.transpose(np.vstack([right_fitx, self.ploty]))])
            cv2.polylines(color_warp, np.int32([pts_right_render]), isClosed=False, color=(100,0,255), thickness=line_thickness)
        
        if pts_left_render is not None and pts_right_render is not None:
            # İki şerit arasını doldur
            pts_fill = np.hstack((pts_left_render, pts_right_render[:, ::-1, :]))
            cv2.fillPoly(color_warp, np.int32([pts_fill]), (0,255,100)) # Yeşil dolgu

        # Kuşbakışı çizimi orijinal görüntüye geri yansıt
        unwarped_lines = cv2.warpPerspective(color_warp, self.Minv, 
                                             (frame_to_draw_on.shape[1], frame_to_draw_on.shape[0]))
        # Yerinde güncelleme (dst=frame_to_draw_on)
        cv2.addWeighted(unwarped_lines, 0.4, frame_to_draw_on, 1.0, 0, dst=frame_to_draw_on)


    def detect_lanes(self, input_frame_rgb):
        # Bu fonksiyon, üzerine çizim yapılmış bir GÖRÜNTÜ KOPYASI döndürmelidir.
        output_display_frame = input_frame_rgb.copy() # Her zaman kopya ile başla

        preprocessed_binary = self._preprocess_image(input_frame_rgb) # Tespit için orijinali kullan
        warped_binary = self._perspective_warp(preprocessed_binary)
        
        # lx, ly, rx, ry: bulunan şerit pikselleri
        # out_img_diag: kayar pencerelerin çizildiği kuşbakışı teşhis görüntüsü
        lx, ly, rx, ry, out_img_diag = self._find_lane_pixels_sliding_window(warped_binary)
        
        # _fit_polynomial, self.left_fit, self.right_fit vb. günceller
        self._fit_polynomial(lx, ly, rx, ry, warped_binary.shape)
        
        # _calculate_curvature_offset, self.stable_curvature ve self.stable_offset_m'yi günceller ve döndürür
        final_curvature, final_offset_m = self._calculate_curvature_offset(warped_binary.shape)
        
        # _draw_lanes_on_frame, output_display_frame'i (kopya) yerinde günceller
        self._draw_lanes_on_frame(output_display_frame, warped_binary.shape)

        # Teşhis görüntüsünü renkliye çevir (eğer griyse)
        diag_to_return = cv2.cvtColor(out_img_diag, cv2.COLOR_GRAY2BGR) \
                         if len(out_img_diag.shape)==2 or out_img_diag.shape[2]==1 \
                         else out_img_diag.copy() # Zaten renkliyse kopya al
            
        return output_display_frame, final_offset_m, final_curvature, diag_to_return

    def cleanup(self):
        print("Şerit Tespit Modülü Kapatılıyor...")
        if hasattr(self, 'picam2') and self.picam2.started:
            self.picam2.stop()
        print("Kamera durduruldu.")

if __name__ == "__main__":
    detector = None
    try:
        # Test için düşük çözünürlük ve FPS kullanabilirsiniz
        TEST_IMG_WIDTH, TEST_IMG_HEIGHT, TEST_FPS = 320, 240, 15
        detector = LaneDetector(image_width=TEST_IMG_WIDTH, image_height=TEST_IMG_HEIGHT, camera_fps=TEST_FPS)
        print("LaneDetector Test Modu. 'q' ile çıkın.")
        
        cv2.namedWindow("Lane Detection - Output", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Lane Detection - Diagnostic", cv2.WINDOW_NORMAL)

        while True:
            frame_original = detector.capture_frame()
            if frame_original is None: 
                print("Frame alınamadı!")
                time.sleep(0.1)
                continue

            # detect_lanes fonksiyonu, üzerine çizim yapılmış bir kopya döndürür
            display_output_frame, offset, curvature, diagnostic_image = detector.detect_lanes(frame_original)
            
            # Bilgileri ekrana yazdır
            cv2.putText(display_output_frame, f"Offset: {offset:.2f}m", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
            cv2.putText(display_output_frame, f"Offset: {offset:.2f}m", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
            cv2.putText(display_output_frame, f"Curv: {curvature:.0f}m", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
            cv2.putText(display_output_frame, f"Curv: {curvature:.0f}m", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

            cv2.imshow("Lane Detection - Output", display_output_frame)
            cv2.imshow("Lane Detection - Diagnostic", diagnostic_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Test kullanıcı tarafından durduruldu.")
    except Exception as e:
        print(f"Test sırasında bir hata oluştu: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if detector:
            detector.cleanup()
        cv2.destroyAllWindows()
        print("LaneDetector Test Modu Sonlandırıldı.")