# lane_detector.py

import cv2
import numpy as np
from picamera2 import Picamera2
from libcamera import controls
import time

class LaneDetector:
    def __init__(self, image_width=320, image_height=240, camera_fps=20): # Düşük çözünürlükle başla
        print("Şerit Tespit Modülü Başlatılıyor (Kapsamlı Güncelleme)...")
        self.image_width = image_width
        self.image_height = image_height
        self.camera_fps = camera_fps

        self.picam2 = Picamera2()
        control_settings = {"FrameRate": self.camera_fps, "AnalogueGain": 1.2, "ExposureTime": 10000}
        config = self.picam2.create_video_configuration(
            main={"size": (self.image_width, self.image_height), "format": "RGB888"},
            lores={"size": (self.image_width // 2, self.image_height // 2), "format": "YUV420"}, # Hızlı önizleme için
            controls=control_settings
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(2.0) # Kameranın iyice ayarlanması için daha uzun bekleme
        print(f"Kamera {self.image_width}x{self.image_height} @ {self.camera_fps}FPS ile başlatıldı.")

        # --- Perspektif Dönüşümü (ÇOK ÖNEMLİ - AYARLANMALI!) ---
        # Bu noktalar, şeritlerin yola paralel göründüğü bir yamuk oluşturmalı.
        # Üst noktalar ufka daha yakın, alt noktalar kameraya daha yakın olmalı.
        # (sol-üst, sağ-üst, sağ-alt, sol-alt)
        # Örnek değerler (320x240 için):
        # Daha uzağa bakmak için üst Y'yi azalt, üst X marjını azalt.
        # Daha geniş bir alt taban için alt X marjını azalt.
        roi_top_y = int(self.image_height * 0.50) # Ufuk çizgisine daha yakın (%50-%60 arası dene)
        roi_bottom_y_offset = 5                   # Görüntünün en altından boşluk
        
        # Üst kenarın darlığı (ufukta şeritler ne kadar daralıyor)
        top_x_width_ratio = 0.15 # Orta noktanın %15 sağı/solu (daha dar = daha uzağa bakış)
        # Alt kenarın genişliği (kameraya yakın şeritler ne kadar geniş)
        bottom_x_width_ratio = 0.95 # Görüntü genişliğinin %95'i

        src_pts_list = [
            (self.image_width//2 - int(self.image_width*top_x_width_ratio/2), roi_top_y), # Sol Üst
            (self.image_width//2 + int(self.image_width*top_x_width_ratio/2), roi_top_y), # Sağ Üst
            (self.image_width//2 + int(self.image_width*bottom_x_width_ratio/2), self.image_height - roi_bottom_y_offset), # Sağ Alt
            (self.image_width//2 - int(self.image_width*bottom_x_width_ratio/2), self.image_height - roi_bottom_y_offset)  # Sol Alt
        ]
        self.src_points = np.float32(src_pts_list)
        self.roi_poly_to_draw = np.array([self.src_points], dtype=np.int32)

        # Hedef kuşbakışı görüntü boyutları
        self.warped_img_width = int(self.image_width * 0.9) # Kuşbakışı genişliği
        self.warped_img_height = self.image_height # Yükseklik aynı kalsın
        self.warped_img_size = (self.warped_img_width, self.warped_img_height)

        self.dst_points = np.float32([
            [0, 0], [self.warped_img_width - 1, 0],
            [self.warped_img_width - 1, self.warped_img_height - 1], [0, self.warped_img_height - 1]
        ])

        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.Minv = cv2.getPerspectiveTransform(self.dst_points, self.src_points)

        # --- Metre Başına Piksel (KALİBRASYON GEREKTİRİR!) ---
        # Gerçekte iki şerit merkezi arası (örn: 0.8m) kuşbakışında kaç piksel?
        # Kuşbakışı görüntünün genişliğinin %X'i kadar olduğunu varsayalım.
        # Örn: Şeritler kuşbakışı görüntünün %70'ini kaplıyorsa.
        self.xm_per_pix = (0.40 * 2) / (self.warped_img_width * 0.70)
        # Gerçekte dikey 0.5m (örn) kuşbakışında kaç piksel?
        self.ym_per_pix = 0.50 / (self.warped_img_height * 0.5) # Yüksekliğin yarısı 0.5m ise

        self.left_fit, self.right_fit = None, None
        self.left_fit_cr, self.right_fit_cr = None, None
        self.ploty = np.linspace(0, self.warped_img_height - 1, self.warped_img_height)

        self.stable_offset_m = 0.0; self.stable_curvature = float('inf')
        self.smoothing_alpha = 0.4 # Yumuşatma faktörü (0-1)

        self.last_valid_left_fit = None # Virajlarda kaybolma durumunda kullanılabilir
        self.last_valid_right_fit = None


    def capture_frame(self):
        return self.picam2.capture_array("main")

    def _image_preprocessing(self, img_rgb):
        # 1. Renk Alanı Değişimi (İsteğe Bağlı: HLS veya Lab)
        # hls = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HLS)
        # l_channel = hls[:,:,1] # Lightness kanalı
        # s_channel = hls[:,:,2] # Saturation kanalı

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # 2. Gaussian Blur (Gürültü Azaltma)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0) # Kernel (3,3) veya (5,5)

        # 3. Eşikleme (Beyaz şeritleri ayırma)
        # Farklı eşikleme yöntemleri denenebilir:
        # _, binary_thresh = cv2.threshold(blurred, 160, 255, cv2.THRESH_BINARY) # Basit eşikleme
        binary_adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, 19, 5) # blockSize, C
        
        # Renk maskeleme (özellikle sarı/beyaz şeritler için HLS/HSV'de daha iyi olabilir)
        # Örnek beyaz maskesi (HLS)
        # lower_white = np.array([0, 200, 0], dtype=np.uint8) # L kanalı için yüksek eşik
        # upper_white = np.array([255, 255, 255], dtype=np.uint8)
        # white_mask = cv2.inRange(hls, lower_white, upper_white)
        # combined_binary = cv2.bitwise_or(binary_adaptive, white_mask) # Veya ikisini birleştir

        # Morfolojik operasyonlar (İsteğe Bağlı: küçük gürültüleri temizle, çizgileri birleştir)
        # kernel = np.ones((3,3), np.uint8)
        # binary_adaptive = cv2.morphologyEx(binary_adaptive, cv2.MORPH_OPEN, kernel, iterations=1)
        # binary_adaptive = cv2.morphologyEx(binary_adaptive, cv2.MORPH_CLOSE, kernel, iterations=1)

        return binary_adaptive # Veya combined_binary

    def _perspective_transform(self, img_binary):
        return cv2.warpPerspective(img_binary, self.M, self.warped_img_size, flags=cv2.INTER_NEAREST) # INTER_LINEAR veya INTER_NEAREST

    def _find_lane_pixels_sliding_window(self, warped_binary_img):
        out_img_diag = np.dstack((warped_binary_img, warped_binary_img, warped_binary_img))
        histogram = np.sum(warped_binary_img[warped_binary_img.shape[0]*3//4:, :], axis=0) # Alt çeyrekte ara
        
        midpoint = np.int32(histogram.shape[0]//2)
        leftx_base, rightx_base = np.argmax(histogram[:midpoint]), np.argmax(histogram[midpoint:]) + midpoint

        nwindows = 10 # Daha fazla pencere, virajlarda daha iyi detay
        margin = int(self.warped_img_width * 0.10) # Pencere genişliği (%10)
        minpix = 40 # Minimum piksel

        window_height = np.int32(warped_binary_img.shape[0]//nwindows)
        nonzero = warped_binary_img.nonzero(); nonzeroy,nonzerox = np.array(nonzero[0]),np.array(nonzero[1])
        leftx_c,rightx_c=leftx_base,rightx_base
        left_inds,right_inds=[],[]

        for w_idx in range(nwindows):
            y_low = warped_binary_img.shape[0]-(w_idx+1)*window_height
            y_high = warped_binary_img.shape[0]-w_idx*window_height
            xl_low,xl_high=leftx_c-margin,leftx_c+margin; xr_low,xr_high=rightx_c-margin,rightx_c+margin
            cv2.rectangle(out_img_diag,(xl_low,y_low),(xl_high,y_high),(0,255,0),1)
            cv2.rectangle(out_img_diag,(xr_low,y_low),(xr_high,y_high),(0,255,0),1)
            good_l=((nonzeroy>=y_low)&(nonzeroy<y_high)&(nonzerox>=xl_low)&(nonzerox<xl_high)).nonzero()[0]
            good_r=((nonzeroy>=y_low)&(nonzeroy<y_high)&(nonzerox>=xr_low)&(nonzerox<xr_high)).nonzero()[0]
            left_inds.append(good_l);right_inds.append(good_r)
            if len(good_l)>minpix:leftx_c=np.int32(np.mean(nonzerox[good_l]))
            if len(good_r)>minpix:rightx_c=np.int32(np.mean(nonzerox[good_r]))
        
        try: left_inds,right_inds=np.concatenate(left_inds),np.concatenate(right_inds)
        except ValueError: pass # Boşsa birleştirme hatası vermesin
        
        return nonzerox[left_inds],nonzeroy[left_inds],nonzerox[right_inds],nonzeroy[right_inds],out_img_diag

    def _fit_polynomial_from_pixels(self, nonzerox, nonzeroy, warped_img_shape):
        """Verilen piksellere polinom uydurur."""
        if len(nonzerox) < 30: # Yeterli piksel yoksa None döndür
            return None, None 
        try:
            fit_px = np.polyfit(nonzeroy, nonzerox, 2)
            if self.xm_per_pix > 1e-7 and self.ym_per_pix > 1e-7:
                fit_cr = np.polyfit(nonzeroy * self.ym_per_pix, nonzerox * self.xm_per_pix, 2)
            else:
                fit_cr = None
            return fit_px, fit_cr
        except (np.RankWarning, TypeError, np.linalg.LinAlgError):
            return None, None # Hata durumunda None

    def _update_lane_fits(self, leftx, lefty, rightx, righty, warped_img_shape):
        # Sol şerit için fit
        new_left_fit, new_left_fit_cr = self._fit_polynomial_from_pixels(leftx, lefty, warped_img_shape)
        if new_left_fit is not None: self.left_fit = new_left_fit
        if new_left_fit_cr is not None: self.left_fit_cr = new_left_fit_cr
        
        # Sağ şerit için fit
        new_right_fit, new_right_fit_cr = self._fit_polynomial_from_pixels(rightx, righty, warped_img_shape)
        if new_right_fit is not None: self.right_fit = new_right_fit
        if new_right_fit_cr is not None: self.right_fit_cr = new_right_fit_cr

        # Eğer bir şerit kaybolursa, diğerine göre tahmin etmeye çalışma (şimdilik)
        # Daha karmaşık senaryolarda bu eklenebilir.

    def _calculate_curvature_and_offset(self, warped_img_shape):
        if self.ploty is None or self.ym_per_pix < 1e-7 or self.xm_per_pix < 1e-7:
            return self.stable_curvature, self.stable_offset_m

        y_eval_m = np.mean(self.ploty) * self.ym_per_pix # Eğrilik için orta nokta
        y_offset_px = warped_img_shape[0] - 1 # Offset için en alt nokta

        cur_l, cur_r = float('inf'), float('inf')
        if self.left_fit_cr is not None:
            A,B=self.left_fit_cr[0],self.left_fit_cr[1]; 
            if abs(A)>1e-7: cur_l=((1+(2*A*y_eval_m+B)**2)**1.5)/abs(2*A)
        if self.right_fit_cr is not None:
            A,B=self.right_fit_cr[0],self.right_fit_cr[1]; 
            if abs(A)>1e-7: cur_r=((1+(2*A*y_eval_m+B)**2)**1.5)/abs(2*A)

        cur_calc = float('inf')
        vl,vr=(cur_l!=float('inf') and cur_l>0),(cur_r!=float('inf') and cur_r>0)
        if vl and vr: cur_calc = (cur_l+cur_r)/2
        elif vl: cur_calc = cur_l
        elif vr: cur_calc = cur_r
        
        if cur_calc != float('inf'):
            self.stable_curvature = self.smoothing_alpha*cur_calc + (1-self.smoothing_alpha)*self.stable_curvature

        offset_m_calc = 0.0
        lx_b, rx_b = None, None
        if self.left_fit: lx_b = np.polyval(self.left_fit, y_offset_px)
        if self.right_fit: rx_b = np.polyval(self.right_fit, y_offset_px)
        
        veh_c_px = warped_img_shape[1]/2.0
        if lx_b is not None and rx_b is not None:
            lane_c_px = (lx_b+rx_b)/2.0
            offset_m_calc = (veh_c_px-lane_c_px)*self.xm_per_pix
        elif lx_b is not None: # Sadece sol
            offset_m_calc = (veh_c_px - (lx_b + (0.40/self.xm_per_pix)/2.0))*self.xm_per_pix
        elif rx_b is not None: # Sadece sağ
            offset_m_calc = (veh_c_px - (rx_b - (0.40/self.xm_per_pix)/2.0))*self.xm_per_pix
        # else: offset_m_calc = self.stable_offset_m # Hiç şerit yoksa önceki offset

        self.stable_offset_m = self.smoothing_alpha*offset_m_calc + (1-self.smoothing_alpha)*self.stable_offset_m
        return self.stable_curvature, self.stable_offset_m

    def _draw_lanes_and_roi(self, output_display_frame, warped_img_ref_shape):
        cv2.polylines(output_display_frame, [self.roi_poly_to_draw], True, (0,180,180), 1) # ROI çizimi

        if self.ploty is None or (self.left_fit is None and self.right_fit is None):
            return # Çizilecek şerit yok

        color_warp = np.zeros((warped_img_ref_shape[0], warped_img_ref_shape[1], 3), dtype=np.uint8)
        line_thick = 12

        pts_l_render, pts_r_render = None, None
        if self.left_fit is not None:
            lfx = np.polyval(self.left_fit, self.ploty)
            pts_l_render = np.array([np.transpose(np.vstack([lfx, self.ploty]))])
            cv2.polylines(color_warp, np.int32(pts_l_render), False, (255,50,150), line_thick)
        if self.right_fit is not None:
            rfx = np.polyval(self.right_fit, self.ploty)
            pts_r_render = np.array([np.transpose(np.vstack([rfx, self.ploty]))])
            cv2.polylines(color_warp, np.int32(pts_r_render), False, (150,50,255), line_thick)
        if pts_l_render is not None and pts_r_render is not None:
            pts_fill = np.hstack((pts_l_render, pts_r_render[:,::-1,:]))
            cv2.fillPoly(color_warp, np.int32([pts_fill]), (50,255,150)) # Alan dolgusu

        unwarped_overlay = cv2.warpPerspective(color_warp, self.Minv, 
                                               (output_display_frame.shape[1], output_display_frame.shape[0]))
        cv2.addWeighted(unwarped_overlay, 0.35, output_display_frame, 1.0, 0, dst=output_display_frame)

    def detect_lanes(self, input_frame_rgb):
        output_frame_with_drawings = input_frame_rgb.copy()
        binary_processed = self._image_preprocessing(input_frame_rgb)
        binary_warped = self._perspective_transform(binary_processed)
        
        lx_px, ly_px, rx_px, ry_px, diag_warped_img = self._find_lane_pixels_sliding_window(binary_warped)
        
        self._update_lane_fits(lx_px, ly_px, rx_px, ry_px, binary_warped.shape)
        
        final_curvature, final_offset = self._calculate_curvature_and_offset(binary_warped.shape)
        
        self._draw_lanes_and_roi(output_frame_with_drawings, binary_warped.shape)

        diag_to_return = cv2.cvtColor(diag_warped_img, cv2.COLOR_GRAY2BGR) \
                         if len(diag_warped_img.shape)==2 else diag_warped_img.copy()
            
        return output_frame_with_drawings, final_offset, final_curvature, diag_to_return

    def cleanup(self):
        print("LaneDetector: Kamera kapatılıyor...");
        if hasattr(self, 'picam2') and self.picam2.started: self.picam2.stop()
        print("LaneDetector: Kamera durdu.")

if __name__ == "__main__":
    detector = None
    try:
        W, H, FPS = 320, 240, 15 # Test için düşük ayarlar
        detector = LaneDetector(image_width=W, image_height=H, camera_fps=FPS)
        print("LaneDetector Test. 'q' ile çık.")
        cv2.namedWindow("Lane Test - Output", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Lane Test - Diagnostic", cv2.WINDOW_NORMAL)
        while True:
            frame = detector.capture_frame()
            if frame is None: continue
            display_out, offset, curve, diag_img = detector.detect_lanes(frame)
            cv2.putText(display_out, f"O:{offset:.2f} C:{curve:.0f}",(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
            cv2.putText(display_out, f"O:{offset:.2f} C:{curve:.0f}",(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)
            cv2.imshow("Lane Test - Output", display_out)
            cv2.imshow("Lane Test - Diagnostic", diag_img)
            if cv2.waitKey(1)&0xFF==ord('q'): break
    except Exception as e: print(f"Hata: {e}"); import traceback; traceback.print_exc()
    finally:
        if detector: detector.cleanup()
        cv2.destroyAllWindows()