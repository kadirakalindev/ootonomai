# lane_detector.py

import cv2
import numpy as np
from picamera2 import Picamera2
from libcamera import controls
import time

class LaneDetector:
    def __init__(self, image_width=320, image_height=240, camera_fps=20):
        print("Şerit Tespit Modülü Başlatılıyor (Revize Edilmiş)...")
        self.image_width = image_width
        self.image_height = image_height
        self.camera_fps = camera_fps

        self.picam2 = Picamera2()
        # Kamera ayarları (parlama ve ışık koşulları için önemli)
        # Manuel ayar denemesi:
        cam_controls = {
            "FrameRate": float(self.camera_fps), # float olmalı
            "AeEnable": False, # Otomatik pozlamayı kapat
            "AwbEnable": False, # Otomatik beyaz dengesini kapat
            "AnalogueGain": 2.0, # Ortam ışığına göre 1.0 ile 8.0 arası dene
            "ExposureTime": 10000, # Mikrosaniye (10ms). Daha düşük = daha karanlık ama daha az hareket bulanıklığı
            "ColourGains": (1.5, 1.5) # (kırmızı_kazancı, mavi_kazancı) - beyaz dengesi için
        }
        # Otomatik ayarlar için:
        # cam_controls = {"FrameRate": float(self.camera_fps), "AeEnable": True, "AwbEnable": True}

        config = self.picam2.create_video_configuration(
            main={"size": (self.image_width, self.image_height), "format": "RGB888"},
            lores={"size": (self.image_width // 2, self.image_height // 2), "format": "YUV420"},
            controls=cam_controls
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(2.0)
        print(f"Kamera {self.image_width}x{self.image_height} @ {self.camera_fps}FPS ile başlatıldı.")
        print(f"Kullanılan kamera kontrolleri: {self.picam2.camera_controls}")


        # --- Perspektif Dönüşümü (ÇOK ÖNEMLİ - AYARLANMALI!) ---
        # Bu noktalar, şeritlerin yola paralel göründüğü bir yamuk oluşturmalı.
        # (sol-üst, sağ-üst, sağ-alt, sol-alt)
        # Örnek değerler (320x240 için):
        self.src_points = np.float32([ # BUNLARI KENDİ KAMERA AÇINIZA GÖRE AYARLAYIN!
            (self.image_width * 0.25, self.image_height * 0.60),  # Sol Üst (daha aşağıda ve içerde)
            (self.image_width * 0.75, self.image_height * 0.60),  # Sağ Üst
            (self.image_width * 0.98, self.image_height * 0.95),  # Sağ Alt (görüntü kenarına yakın)
            (self.image_width * 0.02, self.image_height * 0.95)   # Sol Alt
        ])
        self.roi_poly_to_draw = np.array([self.src_points], dtype=np.int32)

        self.warped_img_width = self.image_width # Kuşbakışı genişliği orijinalle aynı tut
        self.warped_img_height = self.image_height
        self.warped_img_size = (self.warped_img_width, self.warped_img_height)

        self.dst_points = np.float32([
            [self.warped_img_width * 0.15, 0], # Kuşbakışında da bir miktar perspektif bırakabiliriz
            [self.warped_img_width * 0.85, 0],
            [self.warped_img_width * 0.95, self.warped_img_height - 1],
            [self.warped_img_width * 0.05, self.warped_img_height - 1]
        ])

        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.Minv = cv2.getPerspectiveTransform(self.dst_points, self.src_points)

        # --- Metre Başına Piksel (KALİBRASYON GEREKTİRİR!) ---
        # Gerçekte iki şerit merkezi arası (örn: 0.8m) kuşbakışında kaç piksel?
        # Kuşbakışı görüntünün genişliğinin %X'i kadar olduğunu varsayalım.
        # Örneğin, dst_points'deki x farkı: (0.85-0.15)*warped_img_width = 0.7*warped_img_width
        self.xm_per_pix = (0.40 * 2) / (self.warped_img_width * 0.70) 
        # Gerçekte dikey 1m (örn) kuşbakışında kaç piksel? (ROI'nin dikey uzunluğuna göre)
        self.ym_per_pix = 1.0 / (self.warped_img_height * 0.6) # Yüksekliğin %60'ı 1m ise

        # Şerit takibi için durum değişkenleri
        self.left_fit_coeffs, self.right_fit_coeffs = None, None # Piksel uzayında katsayılar
        self.left_fit_coeffs_cr, self.right_fit_coeffs_cr = None, None # Metrik uzayda katsayılar
        self.ploty = np.linspace(0, self.warped_img_height - 1, self.warped_img_height)

        self.stable_offset_m = 0.0
        self.stable_curvature = 1e6 # Başlangıçta çok büyük (neredeyse düz)
        self.smoothing_alpha = 0.30 # Yumuşatma faktörü (0-1)

        self.frames_since_last_good_fit_left = 0
        self.frames_since_last_good_fit_right = 0
        self.max_frames_to_trust_old_fit = 5 # Kaç frame boyunca eski fit'e güvenelim

    def capture_frame(self):
        return self.picam2.capture_array("main")

    def _color_thresholding(self, img_rgb):
        # Beyaz şeritler için HLS renk uzayını kullanmak genellikle daha iyidir
        hls = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HLS)
        # Lightness (L) kanalı: Parlak beyazları yakalar
        l_channel = hls[:,:,1]
        # Saturation (S) kanalı: Renkli olmayanları (beyaz/gri) yakalar
        s_channel = hls[:,:,2]

        # Beyaz için L kanalı eşiği (AYARLANMALI)
        l_thresh_min, l_thresh_max = 180, 255 # Parlak beyazlar
        l_binary = np.zeros_like(l_channel)
        l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 255
        
        # Beyaz için S kanalı eşiği (düşük doygunluk) (AYARLANMALI)
        s_thresh_min, s_thresh_max = 0, 70 # Çok renkli olmayanlar
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 255

        # İki maskeyi birleştir (L AND S)
        # Veya sadece L kanalı ya da S kanalı bazen daha iyi sonuç verebilir.
        # combined_binary = cv2.bitwise_and(l_binary, s_binary)
        # Şimdilik sadece L kanalını kullanalım, daha basit.
        # Ya da S kanalı düşük doygunlukları daha iyi yakalayabilir.
        # return s_binary # VEYA l_binary VEYA combined_binary

        # Alternatif: Gri tonlama ve adaptif eşikleme (daha genel ama daha az renk bilgisi)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (3,3), 0) # Daha az blur
        adaptive_binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, 21, 4) # blockSize, C
        return adaptive_binary


    def _perspective_transform(self, img_binary):
        return cv2.warpPerspective(img_binary, self.M, self.warped_img_size, flags=cv2.INTER_LINEAR)

    def _find_lane_pixels(self, warped_binary_img, prev_left_fit, prev_right_fit):
        out_img_diag = np.dstack((warped_binary_img, warped_binary_img, warped_binary_img))
        nonzero = warped_binary_img.nonzero()
        nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])
        
        margin = int(self.warped_img_width * 0.08) # Arama marjını biraz daralt

        left_lane_inds, right_lane_inds = [], []

        # Eğer önceki frame'de şerit bulunduysa, o fitin etrafında arama yap (daha hızlı ve stabil)
        if prev_left_fit is not None and prev_right_fit is not None:
            left_line_window1 = np.polyval(prev_left_fit, nonzeroy) - margin
            left_line_window2 = np.polyval(prev_left_fit, nonzeroy) + margin
            right_line_window1 = np.polyval(prev_right_fit, nonzeroy) - margin
            right_line_window2 = np.polyval(prev_right_fit, nonzeroy) + margin

            left_lane_inds = ((nonzerox > left_line_window1) & (nonzerox < left_line_window2)).nonzero()[0]
            right_lane_inds = ((nonzerox > right_line_window1) & (nonzerox < right_line_window2)).nonzero()[0]
        
        # Eğer önceki fit yoksa veya etrafında arama başarısız olursa (çok az piksel), sliding window yap
        if len(left_lane_inds) < 100 or len(right_lane_inds) < 100: # Yeterli piksel bulunamazsa
            # print("Önceki fit ile arama başarısız veya ilk frame, sliding window kullanılıyor.")
            histogram = np.sum(warped_binary_img[warped_binary_img.shape[0]*2//3:, :], axis=0) # Alt 1/3
            midpoint = np.int32(histogram.shape[0]//2)
            leftx_base, rightx_base = np.argmax(histogram[:midpoint]), np.argmax(histogram[midpoint:]) + midpoint
            
            nwindows = 10
            minpix_sw = 45 # Sliding window için min piksel
            window_height_sw = np.int32(warped_binary_img.shape[0]//nwindows)
            
            current_lx, current_rx = leftx_base, rightx_base
            temp_left_inds, temp_right_inds = [], []

            for window in range(nwindows):
                y_low = warped_binary_img.shape[0]-(window+1)*window_height_sw
                y_high = warped_binary_img.shape[0]-window*window_height_sw
                xl_low,xl_high=current_lx-margin,current_lx+margin
                xr_low,xr_high=current_rx-margin,current_rx+margin
                cv2.rectangle(out_img_diag,(xl_low,y_low),(xl_high,y_high),(0,255,0),1)
                cv2.rectangle(out_img_diag,(xr_low,y_low),(xr_high,y_high),(0,255,0),1)
                good_l=((nonzeroy>=y_low)&(nonzeroy<y_high)&(nonzerox>=xl_low)&(nonzerox<xl_high)).nonzero()[0]
                good_r=((nonzeroy>=y_low)&(nonzeroy<y_high)&(nonzerox>=xr_low)&(nonzerox<xr_high)).nonzero()[0]
                temp_left_inds.append(good_l); temp_right_inds.append(good_r)
                if len(good_l)>minpix_sw: current_lx=np.int32(np.mean(nonzerox[good_l]))
                if len(good_r)>minpix_sw: current_rx=np.int32(np.mean(nonzerox[good_r]))
            try:
                left_lane_inds = np.concatenate(temp_left_inds)
                right_lane_inds = np.concatenate(temp_right_inds)
            except ValueError: pass
        
        # Teşhis için bulunan pikselleri boya
        out_img_diag[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255,0,0] # Sol Mavi
        out_img_diag[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0,0,255] # Sağ Kırmızı

        return nonzerox[left_lane_inds],nonzeroy[left_lane_inds], \
               nonzerox[right_lane_inds],nonzeroy[right_lane_inds], out_img_diag


    def _fit_and_validate_polynomial(self, nonzerox_lane, nonzeroy_lane, prev_fit_cr):
        if len(nonzerox_lane) < 50: # Güvenilir fit için minimum piksel
            self.frames_since_last_good_fit_left +=1 # Veya sağ için ayrı sayaç
            if prev_fit_cr is not None and self.frames_since_last_good_fit_left < self.max_frames_to_trust_old_fit:
                # print("Az piksel, önceki CR fit kullanılıyor.")
                return None, prev_fit_cr # Piksel fit yok, metrik eski
            # print("Az piksel, fit yok.")
            return None, None

        try:
            fit_px = np.polyfit(nonzeroy_lane, nonzerox_lane, 2)
            fit_cr = None
            if self.xm_per_pix > 1e-7 and self.ym_per_pix > 1e-7:
                fit_cr = np.polyfit(nonzeroy_lane*self.ym_per_pix, nonzerox_lane*self.xm_per_pix, 2)
            
            # Basit geçerlilik kontrolü (eğrilik çok absürt değilse)
            if fit_cr is not None and abs(fit_cr[0]) > 0.005: # A katsayısı çok büyükse (çok keskin anlamsız viraj)
                # print(f"Geçersiz CR fit (A katsayısı çok büyük): {fit_cr[0]}")
                self.frames_since_last_good_fit_left +=1 # Veya sağ
                if prev_fit_cr is not None and self.frames_since_last_good_fit_left < self.max_frames_to_trust_old_fit:
                    return None, prev_fit_cr # Piksel fit yok, metrik eski
                return None,None # Hem piksel hem metrik geçersiz

            self.frames_since_last_good_fit_left = 0 # Başarılı fit, sayacı sıfırla
            return fit_px, fit_cr
        except (np.RankWarning, TypeError, np.linalg.LinAlgError):
            self.frames_since_last_good_fit_left +=1 # Veya sağ
            # print("Polyfit hatası.")
            if prev_fit_cr is not None and self.frames_since_last_good_fit_left < self.max_frames_to_trust_old_fit:
                return None, prev_fit_cr
            return None, None


    def _update_lane_fits_with_validation(self, leftx, lefty, rightx, righty):
        new_left_px, new_left_cr = self._fit_and_validate_polynomial(leftx, lefty, self.left_fit_coeffs_cr)
        new_right_px, new_right_cr = self._fit_and_validate_polynomial(rightx, righty, self.right_fit_coeffs_cr)

        if new_left_px is not None: self.left_fit_coeffs = new_left_px
        if new_left_cr is not None: self.left_fit_coeffs_cr = new_left_cr
        
        if new_right_px is not None: self.right_fit_coeffs = new_right_px
        if new_right_cr is not None: self.right_fit_coeffs_cr = new_right_cr
        
        # Eğer bir şerit kaybolursa, diğerine göre tahmin etme (bu karmaşıklığı artırır)
        # Şimdilik, eğer bir fit None ise, o şerit kayıp kabul edilir.

    def _calculate_curvature_and_offset(self, warped_img_shape):
        # ploty'nin her zaman tanımlı olduğundan emin ol
        if self.ploty is None or len(self.ploty) != warped_img_shape[0]:
             self.ploty = np.linspace(0, warped_img_shape[0]-1, warped_img_shape[0])

        if self.ym_per_pix < 1e-7 or self.xm_per_pix < 1e-7:
            return self.stable_curvature, self.stable_offset_m

        y_eval_m = np.mean(self.ploty) * self.ym_per_pix # Eğrilik için orta nokta
        y_offset_px = warped_img_shape[0] - 5 # Offset için en alta yakın bir nokta

        cur_l, cur_r = 1e9, 1e9 # Başlangıçta çok büyük (neredeyse düz)
        if self.left_fit_coeffs_cr is not None:
            A,B=self.left_fit_coeffs_cr[0],self.left_fit_coeffs_cr[1]; 
            if abs(A)>1e-8: cur_l=abs(((1+(2*A*y_eval_m+B)**2)**1.5)/(2*A)) # Mutlak değer
        if self.right_fit_coeffs_cr is not None:
            A,B=self.right_fit_coeffs_cr[0],self.right_fit_coeffs_cr[1]; 
            if abs(A)>1e-8: cur_r=abs(((1+(2*A*y_eval_m+B)**2)**1.5)/(2*A))
        
        cur_calc = 1e9
        if cur_l < 1e8 and cur_r < 1e8 : cur_calc = (cur_l+cur_r)/2.0 # 1e8'den küçükse geçerli kabul et
        elif cur_l < 1e8: cur_calc = cur_l
        elif cur_r < 1e8: cur_calc = cur_r
        
        if cur_calc < 1e8: # Sadece geçerli bir eğrilik hesaplandıysa yumuşat
            self.stable_curvature = self.smoothing_alpha*cur_calc + (1-self.smoothing_alpha)*self.stable_curvature
        # else: self.stable_curvature (değişmez, önceki değerini korur)

        offset_m_calc = self.stable_offset_m # Başlangıçta önceki stabil değeri al
        lx_b, rx_b = None, None
        if self.left_fit_coeffs: lx_b = np.polyval(self.left_fit_coeffs, y_offset_px)
        if self.right_fit_coeffs: rx_b = np.polyval(self.right_fit_coeffs, y_offset_px)
        
        veh_c_px = warped_img_shape[1]/2.0
        lane_width_estimate_px = 0.80 / self.xm_per_pix # İki şerit arası ~80cm

        if lx_b is not None and rx_b is not None:
            # İki şerit de varsa, ortasını bul
            actual_lane_width_px = abs(rx_b - lx_b)
            # Eğer şerit genişliği çok anormal değilse kullan
            if 0.5 * lane_width_estimate_px < actual_lane_width_px < 1.5 * lane_width_estimate_px :
                 lane_c_px = (lx_b+rx_b)/2.0
                 offset_m_calc = (veh_c_px-lane_c_px)*self.xm_per_pix
            # else: print("Anormal şerit genişliği, offset hesaplanmıyor.") # Önceki offset kullanılır
        elif lx_b is not None: # Sadece sol şerit
            # Sol şeridin sağında (yarım toplam şerit genişliği kadar) olduğumuzu varsay
            assumed_lane_center_px = lx_b + lane_width_estimate_px / 2.0
            offset_m_calc = (veh_c_px - assumed_lane_center_px) * self.xm_per_pix
        elif rx_b is not None: # Sadece sağ şerit
            assumed_lane_center_px = rx_b - lane_width_estimate_px / 2.0
            offset_m_calc = (veh_c_px - assumed_lane_center_px) * self.xm_per_pix
            
        self.stable_offset_m = self.smoothing_alpha*offset_m_calc + (1-self.smoothing_alpha)*self.stable_offset_m
        return self.stable_curvature, self.stable_offset_m

    def _draw_lane_area_and_roi(self, output_display_frame, warped_img_ref_shape):
        # ROI her zaman çizilsin
        cv2.polylines(output_display_frame, [self.roi_poly_to_draw], True, (0,180,180), 1)

        if self.ploty is None: return

        # Sadece geçerli fitler varsa çizim yap
        can_draw_left = self.left_fit_coeffs is not None
        can_draw_right = self.right_fit_coeffs is not None

        if not can_draw_left and not can_draw_right: return # Çizilecek şerit yok

        color_warp = np.zeros((warped_img_ref_shape[0], warped_img_ref_shape[1], 3), dtype=np.uint8)
        line_thick = 10

        pts_l_render, pts_r_render = None, None
        if can_draw_left:
            lfx = np.polyval(self.left_fit_coeffs, self.ploty)
            pts_l_render = np.array([np.transpose(np.vstack([lfx, self.ploty]))])
            cv2.polylines(color_warp, np.int32(pts_l_render), False, (255,80,180), line_thick) # Canlı pembe/mor
        if can_draw_right:
            rfx = np.polyval(self.right_fit_coeffs, self.ploty)
            pts_r_render = np.array([np.transpose(np.vstack([rfx, self.ploty]))])
            cv2.polylines(color_warp, np.int32(pts_r_render), False, (180,80,255), line_thick) # Diğer canlı renk
        
        if pts_l_render is not None and pts_r_render is not None:
            pts_fill = np.hstack((pts_l_render, pts_r_render[:,::-1,:]))
            cv2.fillPoly(color_warp, np.int32([pts_fill]), (80,255,180)) # Açık yeşil alan

        unwarped_overlay = cv2.warpPerspective(color_warp, self.Minv, 
                                               (output_display_frame.shape[1], output_display_frame.shape[0]))
        cv2.addWeighted(unwarped_overlay, 0.3, output_display_frame, 1.0, 0, dst=output_display_frame)


    def detect_lanes(self, input_frame_rgb):
        output_frame_with_drawings = input_frame_rgb.copy() # Her zaman kopya ile başla

        binary_processed = self._image_preprocessing(input_frame_rgb)
        binary_warped = self._perspective_transform(binary_processed)
        
        # Önceki frame'in fitlerini kullanarak piksel ara veya sliding window yap
        lx_px, ly_px, rx_px, ry_px, diag_warped_img = self._find_lane_pixels(
            binary_warped, self.left_fit_coeffs, self.right_fit_coeffs
        )
        
        self._update_lane_fits_with_validation(lx_px, ly_px, rx_px, ry_px)
        
        final_curvature, final_offset = self._calculate_curvature_and_offset(binary_warped.shape)
        
        self._draw_lane_area_and_roi(output_frame_with_drawings, binary_warped.shape)

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
        W, H, FPS = 320, 240, 15
        detector = LaneDetector(image_width=W, image_height=H, camera_fps=FPS)
        print("LaneDetector Test (Revize). 'q' ile çık.")
        cv2.namedWindow("Lane Test - Output", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Lane Test - Binary Warped", cv2.WINDOW_NORMAL) # Kuşbakışı binary
        cv2.namedWindow("Lane Test - Diagnostic", cv2.WINDOW_NORMAL) # Kayar pencere/piksel

        while True:
            frame = detector.capture_frame()
            if frame is None: time.sleep(0.01); continue

            # Test için binary ve warped görüntüleri de alalım
            binary_img_for_test = detector._image_preprocessing(frame)
            warped_binary_for_test = detector._perspective_transform(binary_img_for_test)
            
            display_out, offset, curve, diag_img = detector.detect_lanes(frame) # Bu zaten içerde yapıyor
            
            cv2.putText(display_out, f"O:{offset:.3f} C:{curve:.1f}",(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0),2)
            cv2.putText(display_out, f"O:{offset:.3f} C:{curve:.1f}",(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,0),1)
            cv2.imshow("Lane Test - Output", display_out)
            cv2.imshow("Lane Test - Binary Warped", warped_binary_for_test) # Ham kuşbakışı binary
            cv2.imshow("Lane Test - Diagnostic", diag_img) # Kayar pencere/piksel işaretli

            if cv2.waitKey(1)&0xFF==ord('q'): break
    except Exception as e: print(f"Hata: {e}"); import traceback; traceback.print_exc()
    finally:
        if detector: detector.cleanup()
        cv2.destroyAllWindows()