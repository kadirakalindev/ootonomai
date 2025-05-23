# lane_detector.py

import cv2
import numpy as np
from picamera2 import Picamera2
from libcamera import controls 
import time

class LaneDetector:
    def __init__(self, image_width=320, image_height=240, camera_fps=20): # Düşük çözünürlük/FPS ile başla
        print("Şerit Tespit Modülü Başlatılıyor (v_ stabilized)...")
        self.image_width = image_width
        self.image_height = image_height
        self.camera_fps = camera_fps

        self.picam2 = Picamera2()
        # Kamera ayarları (parlama ve ışık koşulları için önemli)
        cam_controls = {
            "FrameRate": float(self.camera_fps), 
            "AeEnable": False, 
            "AwbEnable": False, 
            "AnalogueGain": 1.8, # Ortam ışığına göre ayarla (1.0-8.0)
            "ExposureTime": 12000, # Mikrosaniye (12ms). Daha düşük = daha az hareket bulanıklığı
            "ColourGains": (1.6, 1.4) # (kırmızı_kazancı, mavi_kazancı) - beyaz dengesi için dene
        }
        config = self.picam2.create_video_configuration(
            main={"size": (self.image_width, self.image_height), "format": "RGB888"},
            lores={"size": (self.image_width // 2, self.image_height // 2), "format": "YUV420"}, # Hızlı önizleme
            controls=cam_controls
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(2.0) 
        print(f"Kamera {self.image_width}x{self.image_height} @ {self.camera_fps}FPS başlatıldı.")
        # print(f"Kamera kontrolleri: {self.picam2.camera_controls}") # Debug için

        # --- Perspektif Dönüşümü (ÇOK ÖNEMLİ - AYARLANMALI!) ---
        # (sol-üst, sağ-üst, sağ-alt, sol-alt)
        self.src_points = np.float32([ # BUNLARI KENDİ KAMERA AÇINIZA GÖRE AYARLAYIN!
            (self.image_width * 0.20, self.image_height * 0.62),  # Sol Üst 
            (self.image_width * 0.80, self.image_height * 0.62),  # Sağ Üst
            (self.image_width * 0.99, self.image_height * 0.98),  # Sağ Alt
            (self.image_width * 0.01, self.image_height * 0.98)   # Sol Alt
        ])
        self.roi_poly_to_draw = np.array([self.src_points], dtype=np.int32)

        # Hedef kuşbakışı görüntü boyutları
        self.warped_img_width = int(self.image_width * 0.95) 
        self.warped_img_height = self.image_height 
        self.warped_img_size = (self.warped_img_width, self.warped_img_height)

        self.dst_points = np.float32([
            [self.warped_img_width * 0.10, 0], 
            [self.warped_img_width * 0.90, 0],
            [self.warped_img_width * 0.95, self.warped_img_height - 1],
            [self.warped_img_width * 0.05, self.warped_img_height - 1]
        ])

        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.Minv = cv2.getPerspectiveTransform(self.dst_points, self.src_points)

        # --- Metre Başına Piksel (KALİBRASYON GEREKTİRİR!) ---
        self.xm_per_pix = (0.40 * 2) / (self.warped_img_width * 0.75) # İki şerit arası 0.8m, kuşbakışı genişliğin %75'i ise
        self.ym_per_pix = 0.30 / (self.warped_img_height * 0.30) # Dikey 0.3m, kuşbakışı yüksekliğin %30'u ise

        # Şerit takibi için durum değişkenleri
        self.left_fit_coeffs, self.right_fit_coeffs = None, None 
        self.left_fit_coeffs_cr, self.right_fit_coeffs_cr = None, None 
        self.ploty = np.linspace(0, self.warped_img_height - 1, self.warped_img_height)

        self.stable_offset_m = 0.0
        self.stable_curvature = 1e7 # Başlangıçta çok büyük (neredeyse düz)
        self.smoothing_alpha = 0.30 

        self.frames_no_left_fit = 0 # Sol şerit için ardışık başarısız fit sayısı
        self.frames_no_right_fit = 0 # Sağ şerit için
        self.max_frames_use_old_fit = 7 # Kaç frame boyunca eski fite güvenelim

    def capture_frame(self):
        return self.picam2.capture_array("main")

    def _preprocess_image(self, img_rgb): # TUTARLI METOD ADI
        # HLS renk uzayında L (Lightness) ve S (Saturation) kanallarını kullanmak
        # beyaz ve sarı çizgileri ayırmada genellikle daha etkilidir.
        hls = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HLS)
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]

        # Beyaz çizgiler için: Yüksek L, Düşük S
        # (Bu eşikler sizin pistinize ve ışık koşullarınıza göre AYARLANMALIDIR)
        l_thresh_white_min, l_thresh_white_max = 170, 255
        s_thresh_white_min, s_thresh_white_max = 0, 80
        
        l_binary_white = np.zeros_like(l_channel)
        l_binary_white[(l_channel >= l_thresh_white_min) & (l_channel <= l_thresh_white_max)] = 255
        
        s_binary_white = np.zeros_like(s_channel)
        s_binary_white[(s_channel >= s_thresh_white_min) & (s_channel <= s_thresh_white_max)] = 255
        
        white_mask = cv2.bitwise_and(l_binary_white, s_binary_white)

        # İsteğe bağlı: Sarı çizgiler için ayrı bir maske oluşturup birleştirebilirsiniz.
        # Şimdilik sadece beyaz şeritlere odaklanalım.
        # Eğer siyah zemin, beyaz şerit ise, gri tonlama + adaptif eşikleme de iyi çalışabilir.
        # gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        # blurred = cv2.GaussianBlur(gray, (3,3), 0)
        # binary_output = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                                       cv2.THRESH_BINARY_INV, 21, 5) # blockSize, C
        
        # Morfolojik operasyonlar (gürültüyü azaltmak için)
        kernel = np.ones((3,3), np.uint8)
        processed_binary = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        processed_binary = cv2.morphologyEx(processed_binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return processed_binary


    def _perspective_transform(self, img_binary): # TUTARLI METOD ADI
        return cv2.warpPerspective(img_binary, self.M, self.warped_img_size, flags=cv2.INTER_LINEAR)

    def _find_lane_pixels(self, warped_binary_img, prev_left_fit_coeffs, prev_right_fit_coeffs): # TUTARLI METOD ADI
        out_img_diag = np.dstack((warped_binary_img, warped_binary_img, warped_binary_img)) * 70 # Daha görünür olması için biraz karart
        nonzero = warped_binary_img.nonzero()
        nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])
        
        margin = int(self.warped_img_width * 0.10) # Arama marjı (%10)
        min_pixels_for_refit = 150 # Eğer önceki fitten bu kadar piksel bulunursa, yeni fit yapma

        left_lane_inds, right_lane_inds = [], []

        # 1. Önceki frame'in fitleri etrafında arama
        found_enough_with_prev_left = False
        if prev_left_fit_coeffs is not None and isinstance(prev_left_fit_coeffs, np.ndarray):
            left_line_center = np.polyval(prev_left_fit_coeffs, nonzeroy)
            current_left_inds = ((nonzerox > left_line_center - margin) & (nonzerox < left_line_center + margin)).nonzero()[0]
            if len(current_left_inds) > min_pixels_for_refit :
                left_lane_inds = current_left_inds
                found_enough_with_prev_left = True
        
        found_enough_with_prev_right = False
        if prev_right_fit_coeffs is not None and isinstance(prev_right_fit_coeffs, np.ndarray):
            right_line_center = np.polyval(prev_right_fit_coeffs, nonzeroy)
            current_right_inds = ((nonzerox > right_line_center - margin) & (nonzerox < right_line_center + margin)).nonzero()[0]
            if len(current_right_inds) > min_pixels_for_refit:
                right_lane_inds = current_right_inds
                found_enough_with_prev_right = True

        # 2. Eğer önceki fitlerle yeterli piksel bulunamazsa, sliding window
        if not found_enough_with_prev_left or not found_enough_with_prev_right:
            # print("Sliding window kullanılıyor...")
            histogram = np.sum(warped_binary_img[warped_binary_img.shape[0]*2//3:, :], axis=0)
            midpoint = np.int32(histogram.shape[0]//2)
            
            leftx_base = np.argmax(histogram[:midpoint]) if np.any(histogram[:midpoint]) else midpoint // 2
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint if np.any(histogram[midpoint:]) else midpoint + midpoint // 2
            
            nwindows = 10
            minpix_sw = 40
            window_height_sw = np.int32(warped_binary_img.shape[0]//nwindows)
            
            current_lx, current_rx = leftx_base, rightx_base
            temp_left_inds, temp_right_inds = [], []

            for window in range(nwindows):
                y_low = warped_binary_img.shape[0]-(window+1)*window_height_sw
                y_high = warped_binary_img.shape[0]-window*window_height_sw
                xl_low,xl_high=current_lx-margin,current_lx+margin
                xr_low,xr_high=current_rx-margin,current_rx+margin
                if not found_enough_with_prev_left: # Sadece ihtiyaç varsa çiz
                    cv2.rectangle(out_img_diag,(xl_low,y_low),(xl_high,y_high),(50,200,50),1)
                if not found_enough_with_prev_right:
                    cv2.rectangle(out_img_diag,(xr_low,y_low),(xr_high,y_high),(50,200,50),1)

                good_l=((nonzeroy>=y_low)&(nonzeroy<y_high)&(nonzerox>=xl_low)&(nonzerox<xl_high)).nonzero()[0]
                good_r=((nonzeroy>=y_low)&(nonzeroy<y_high)&(nonzerox>=xr_low)&(nonzerox<xr_high)).nonzero()[0]
                
                if not found_enough_with_prev_left: temp_left_inds.append(good_l)
                if not found_enough_with_prev_right: temp_right_inds.append(good_r)
                
                if len(good_l)>minpix_sw and not found_enough_with_prev_left: current_lx=np.int32(np.mean(nonzerox[good_l]))
                if len(good_r)>minpix_sw and not found_enough_with_prev_right: current_rx=np.int32(np.mean(nonzerox[good_r]))
            
            try:
                if not found_enough_with_prev_left: left_lane_inds = np.concatenate(temp_left_inds)
                if not found_enough_with_prev_right: right_lane_inds = np.concatenate(temp_right_inds)
            except ValueError: pass
        
        # Teşhis için bulunan pikselleri boya
        if len(left_lane_inds) > 0: out_img_diag[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255,100,100] # Sol Açık Mavi
        if len(right_lane_inds) > 0: out_img_diag[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [100,100,255] # Sağ Açık Kırmızı

        return nonzerox[left_lane_inds],nonzeroy[left_lane_inds], \
               nonzerox[right_lane_inds],nonzeroy[right_lane_inds], out_img_diag


    def _fit_and_validate_polynomial(self, nonzerox_lane, nonzeroy_lane, is_left_lane): # METOD ADI KONTROL EDİLDİ
        fit_px, fit_cr = None, None
        frames_no_fit_counter = self.frames_no_left_fit if is_left_lane else self.frames_no_right_fit
        
        if len(nonzerox_lane) < 50: # Yeterli piksel yoksa
            frames_no_fit_counter += 1
            # print(f"{'Sol' if is_left_lane else 'Sağ'} şerit için az piksel ({len(nonzerox_lane)}). Eski fit kullanılacak ({frames_no_fit_counter}/{self.max_frames_use_old_fit}).")
            if frames_no_fit_counter < self.max_frames_use_old_fit:
                # Önceki geçerli CR katsayılarını kullan (eğer varsa)
                prev_cr = self.left_fit_coeffs_cr if is_left_lane else self.right_fit_coeffs_cr
                # Önceki geçerli PX katsayılarını kullan (eğer varsa)
                prev_px = self.left_fit_coeffs if is_left_lane else self.right_fit_coeffs
                if prev_cr is not None: fit_cr = prev_cr
                if prev_px is not None: fit_px = prev_px
            # else: print(f"{'Sol' if is_left_lane else 'Sağ'} şerit için eski fit güveni aşıldı.")
        else: # Yeterli piksel var, yeni fit yap
            try:
                fit_px = np.polyfit(nonzeroy_lane, nonzerox_lane, 2)
                if self.xm_per_pix > 1e-7 and self.ym_per_pix > 1e-7:
                    fit_cr = np.polyfit(nonzeroy_lane*self.ym_per_pix, nonzerox_lane*self.xm_per_pix, 2)
                
                # Basit geçerlilik kontrolü (eğrilik çok absürt değilse)
                if fit_cr is not None and isinstance(fit_cr, np.ndarray) and abs(fit_cr[0]) > 0.008: # A katsayısı (eğrilik)
                    # print(f"{'Sol' if is_left_lane else 'Sağ'} şerit için geçersiz CR fit (A çok büyük): {fit_cr[0]:.4f}")
                    fit_px, fit_cr = None, None # Geçersiz kıl
                    frames_no_fit_counter += 1
                    if frames_no_fit_counter < self.max_frames_use_old_fit: # Eskiye dön
                         prev_cr = self.left_fit_coeffs_cr if is_left_lane else self.right_fit_coeffs_cr
                         prev_px = self.left_fit_coeffs if is_left_lane else self.right_fit_coeffs
                         if prev_cr is not None: fit_cr = prev_cr
                         if prev_px is not None: fit_px = prev_px
                else: # Fit geçerli
                    frames_no_fit_counter = 0 
            except (np.RankWarning, TypeError, np.linalg.LinAlgError):
                # print(f"{'Sol' if is_left_lane else 'Sağ'} şerit için polyfit hatası.")
                frames_no_fit_counter += 1
                if frames_no_fit_counter < self.max_frames_use_old_fit:
                    prev_cr = self.left_fit_coeffs_cr if is_left_lane else self.right_fit_coeffs_cr
                    prev_px = self.left_fit_coeffs if is_left_lane else self.right_fit_coeffs
                    if prev_cr is not None: fit_cr = prev_cr
                    if prev_px is not None: fit_px = prev_px
        
        if is_left_lane: self.frames_no_left_fit = frames_no_fit_counter
        else: self.frames_no_right_fit = frames_no_fit_counter
        
        return fit_px, fit_cr

    def _update_lane_fits_with_validation(self, leftx, lefty, rightx, righty): # METOD ADI KONTROL EDİLDİ
        new_left_px, new_left_cr = self._fit_and_validate_polynomial(leftx, lefty, True)
        new_right_px, new_right_cr = self._fit_and_validate_polynomial(rightx, righty, False)

        if new_left_px is not None: self.left_fit_coeffs = new_left_px
        if new_left_cr is not None: self.left_fit_coeffs_cr = new_left_cr
        
        if new_right_px is not None: self.right_fit_coeffs = new_right_px
        if new_right_cr is not None: self.right_fit_coeffs_cr = new_right_cr

        # ploty'nin her zaman tanımlı olduğundan emin ol
        if self.ploty is None or len(self.ploty) != self.warped_img_height:
            self.ploty = np.linspace(0, self.warped_img_height - 1, self.warped_img_height)


    def _calculate_curvature_and_offset(self, warped_img_shape): # METOD ADI KONTROL EDİLDİ
        if self.ploty is None or self.ym_per_pix < 1e-7 or self.xm_per_pix < 1e-7:
            return self.stable_curvature, self.stable_offset_m
        
        y_eval_m = np.mean(self.ploty) * self.ym_per_pix # Eğrilik için orta nokta
        y_offset_px = warped_img_shape[0] - 10 # Offset için en alta yakın (5-10 piksel yukarı)

        cur_l, cur_r = 1e9, 1e9 # Başlangıçta çok büyük (düz)

        if self.left_fit_coeffs_cr is not None and isinstance(self.left_fit_coeffs_cr,np.ndarray) and len(self.left_fit_coeffs_cr)==3:
            A,B=self.left_fit_coeffs_cr[0],self.left_fit_coeffs_cr[1]
            if abs(A)>1e-8: cur_l=abs(((1+(2*A*y_eval_m+B)**2)**1.5)/(2*A))
        if self.right_fit_coeffs_cr is not None and isinstance(self.right_fit_coeffs_cr,np.ndarray) and len(self.right_fit_coeffs_cr)==3:
            A,B=self.right_fit_coeffs_cr[0],self.right_fit_coeffs_cr[1]
            if abs(A)>1e-8: cur_r=abs(((1+(2*A*y_eval_m+B)**2)**1.5)/(2*A))
        
        cur_calc=1e9; vl,vr=(cur_l<1e8),(cur_r<1e8) # 1e8'den küçükse geçerli
        if vl and vr: cur_calc=(cur_l+cur_r)/2.0
        elif vl: cur_calc=cur_l
        elif vr: cur_calc=cur_r
        
        if cur_calc<1e8: # Sadece geçerli bir eğrilik hesaplandıysa yumuşat
            self.stable_curvature=self.smoothing_alpha*cur_calc+(1-self.smoothing_alpha)*self.stable_curvature
        
        offset_m_calc = self.stable_offset_m # Başlangıçta önceki stabil değeri al
        lx_b, rxb = None, None
        if self.left_fit_coeffs is not None and isinstance(self.left_fit_coeffs,np.ndarray)and len(self.left_fit_coeffs)==3:
            lx_b=np.polyval(self.left_fit_coeffs,y_offset_px)
        if self.right_fit_coeffs is not None and isinstance(self.right_fit_coeffs,np.ndarray)and len(self.right_fit_coeffs)==3:
            rxb=np.polyval(self.right_fit_coeffs,y_offset_px)
        
        veh_c_px=warped_img_shape[1]/2.0
        # Gerçek dünya şerit genişliği (iki şerit merkezi arası)
        world_lane_width_m = 0.8 
        world_lane_width_px = world_lane_width_m / self.xm_per_pix

        if lx_b is not None and rxb is not None:
            detected_lane_width_px = abs(rxb - lx_b)
            # Eğer tespit edilen şerit genişliği çok anormal değilse kullan
            if 0.6 * world_lane_width_px < detected_lane_width_px < 1.4 * world_lane_width_px:
                lane_c_px = (lx_b+rxb)/2.0
                offset_m_calc = (veh_c_px-lane_c_px)*self.xm_per_pix
            # else: print(f"Anormal şerit genişliği: {detected_lane_width_px:.0f}px, bekleniyor: {world_lane_width_px:.0f}px")
        elif lx_b is not None: # Sadece sol şerit bulundu
            # Sol şeridin sağında (yarım toplam şerit genişliği kadar) olduğumuzu varsay
            assumed_lane_center_px = lx_b + world_lane_width_px / 2.0
            offset_m_calc = (veh_c_px - assumed_lane_center_px) * self.xm_per_pix
        elif rxb is not None: # Sadece sağ şerit bulundu
            assumed_lane_center_px = rxb - world_lane_width_px / 2.0
            offset_m_calc = (veh_c_px - assumed_lane_center_px) * self.xm_per_pix
            
        self.stable_offset_m = self.smoothing_alpha*offset_m_calc + (1-self.smoothing_alpha)*self.stable_offset_m
        return self.stable_curvature, self.stable_offset_m

    def _draw_lane_area_and_roi(self, output_display_frame, warped_img_ref_shape): # METOD ADI KONTROL EDİLDİ
        cv2.polylines(output_display_frame,[self.roi_poly_to_draw],True,(0,180,180),1) # ROI
        if self.ploty is None: return

        can_draw_l=self.left_fit_coeffs is not None and isinstance(self.left_fit_coeffs,np.ndarray)and len(self.left_fit_coeffs)==3
        can_draw_r=self.right_fit_coeffs is not None and isinstance(self.right_fit_coeffs,np.ndarray)and len(self.right_fit_coeffs)==3
        if not can_draw_l and not can_draw_r: return

        color_w=np.zeros((warped_img_ref_shape[0],warped_img_ref_shape[1],3),dtype=np.uint8)
        thick=10 # Çizgi kalınlığı
        ptsl,ptsr=None,None
        
        if can_draw_l:
            lfx=np.polyval(self.left_fit_coeffs,self.ploty)
            ptsl=np.array([np.transpose(np.vstack([lfx,self.ploty]))])
            cv2.polylines(color_w,np.int32(ptsl),False,(255,80,180),thick)
        if can_draw_r:
            rfx=np.polyval(self.right_fit_coeffs,self.ploty)
            ptsr=np.array([np.transpose(np.vstack([rfx,self.ploty]))])
            cv2.polylines(color_w,np.int32(ptsr),False,(180,80,255),thick)
        if ptsl is not None and ptsr is not None:
            cv2.fillPoly(color_w,np.int32([np.hstack((ptsl,ptsr[:,::-1,:]))]),(80,255,180))
        
        unwarp=cv2.warpPerspective(color_w,self.Minv,(output_display_frame.shape[1],output_display_frame.shape[0]))
        cv2.addWeighted(unwarp,0.3,output_display_frame,1.0,0,dst=output_display_frame)


    def detect_lanes(self, input_frame_rgb):
        output_frame_with_drawings = input_frame_rgb.copy()

        binary_processed = self._preprocess_image(input_frame_rgb)
        binary_warped = self._perspective_transform(binary_processed)
        
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
        print("LaneDetector Test (Revize Edilmiş). 'q' ile çık.")
        
        cv2.namedWindow("Lane Test - Output", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Lane Test - Binary Preprocessed", cv2.WINDOW_NORMAL) # Ön işlenmiş
        cv2.namedWindow("Lane Test - Binary Warped", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Lane Test - Diagnostic Sliding", cv2.WINDOW_NORMAL)

        while True:
            frame = detector.capture_frame()
            if frame is None: time.sleep(0.01); continue

            binary_img_for_test = detector._preprocess_image(frame)
            warped_binary_for_test = detector._perspective_transform(binary_img_for_test)
            
            display_out, offset, curve, diag_img_sliding = detector.detect_lanes(frame)
            
            cv2.putText(display_out,f"O:{offset:.3f} C:{curve:.1f}",(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0),2)
            cv2.putText(display_out,f"O:{offset:.3f} C:{curve:.1f}",(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,0),1)
            
            cv2.imshow("Lane Test - Output",display_out)
            cv2.imshow("Lane Test - Binary Preprocessed", binary_img_for_test) # Eşiklenmiş görüntü
            cv2.imshow("Lane Test - Binary Warped",warped_binary_for_test)
            cv2.imshow("Lane Test - Diagnostic Sliding",diag_img_sliding) # Kayar pencere veya piksel işaretli

            if cv2.waitKey(1)&0xFF==ord('q'):break
    except Exception as e:print(f"Hata:{e}");import traceback;traceback.print_exc()
    finally:
        if detector:detector.cleanup()
        cv2.destroyAllWindows()