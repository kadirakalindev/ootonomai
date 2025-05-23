# lane_detector.py

import cv2
import numpy as np
from picamera2 import Picamera2
from libcamera import controls
import time

class LaneDetector:
    def __init__(self, image_width=640, image_height=480, camera_fps=30):
        print("Şerit Tespit Modülü Başlatılıyor...")
        self.image_width = image_width
        self.image_height = image_height
        self.camera_fps = camera_fps

        self.picam2 = Picamera2()
        # Düşük çözünürlük ve FPS ile başla (donma sorunları için)
        # self.image_width = 320
        # self.image_height = 240
        # self.camera_fps = 15

        config = self.picam2.create_video_configuration(
            main={"size": (self.image_width, self.image_height), "format": "RGB888"},
            controls={"FrameRate": self.camera_fps}
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(1.5) # Kameranın stabil hale gelmesi için biraz daha bekle
        print(f"Kamera {self.image_width}x{self.image_height} @ {self.camera_fps}FPS ile başlatıldı.")

        # Perspektif dönüşümü için kaynak ve hedef noktaları (AYARLANMALI)
        roi_bottom_offset = 10
        roi_top_y = int(self.image_height * 0.60)
        top_x_margin = int(self.image_width * 0.22) # Biraz daha daraltılabilir

        self.src_points = np.float32([
            (self.image_width // 2 - top_x_margin, roi_top_y),
            (self.image_width // 2 + top_x_margin, roi_top_y),
            (self.image_width - 20, self.image_height - roi_bottom_offset), # Kenarlardan biraz içerde
            (20, self.image_height - roi_bottom_offset)
        ])

        dst_width_ratio = 0.9 # Kuşbakışı genişliği biraz artırabiliriz
        self.warped_img_size = (int(self.image_width * dst_width_ratio), self.image_height) # Yükseklik aynı kalsın

        self.dst_points = np.float32([
            [0, 0],
            [self.warped_img_size[0] - 1, 0],
            [self.warped_img_size[0] - 1, self.warped_img_size[1] - 1],
            [0, self.warped_img_size[1] - 1]
        ])

        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.Minv = cv2.getPerspectiveTransform(self.dst_points, self.src_points)

        # Metre başına piksel (KALİBRASYON GEREKTİRİR!)
        # Örnek: 80cm (iki şerit) kuşbakışında warped_img_size[0]'ın %80'i ise
        self.xm_per_pix = (0.40 * 2) / (self.warped_img_size[0] * 0.80)
        self.ym_per_pix = 0.20 / 40 # Örnek: 20cm / 40 piksel (kuşbakışında ölçülmeli)

        self.left_fit = None
        self.right_fit = None
        self.left_fit_cr = None
        self.right_fit_cr = None
        self.ploty = None
        self.roi_poly_to_draw = np.array([self.src_points], dtype=np.int32) # Çizim için

    def capture_frame(self):
        return self.picam2.capture_array("main")

    def _preprocess_image(self, img_rgb):
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 13, 3) # blockSize ve C ayarlanabilir
        return thresholded

    def _perspective_warp(self, img):
        return cv2.warpPerspective(img, self.M, self.warped_img_size, flags=cv2.INTER_LINEAR)

    def _find_lane_pixels_sliding_window(self, warped_img):
        histogram = np.sum(warped_img[warped_img.shape[0]//2:, :], axis=0)
        out_img_diag = np.dstack((warped_img, warped_img, warped_img)) # Teşhis için kopya
        
        midpoint = np.int32(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Eğer önceki fitler varsa ve güvenilirse, onların etrafında arama yapılabilir (daha stabil)
        # Şimdilik her seferinde histogramdan başlatalım
        
        nwindows = 9
        margin = int(self.warped_img_size[0] * 0.15) # Pencere genişliği (warped genişliğe göre)
        minpix = 50

        window_height = np.int32(warped_img.shape[0]//nwindows)
        nonzero = warped_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base
        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            win_y_low = warped_img.shape[0] - (window+1)*window_height
            win_y_high = warped_img.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            cv2.rectangle(out_img_diag,(win_xleft_low,win_y_low), (win_xleft_high,win_y_high),(0,255,0), 1)
            cv2.rectangle(out_img_diag,(win_xright_low,win_y_low), (win_xright_high,win_y_high),(0,255,0), 1)

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > minpix: leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix: rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))
        
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError: pass

        leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
        rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]
        return leftx, lefty, rightx, righty, out_img_diag

    def _fit_polynomial(self, leftx, lefty, rightx, righty, warped_shape):
        ploty = np.linspace(0, warped_shape[0]-1, warped_shape[0])
        left_f, right_f, left_f_cr, right_f_cr = None, None, None, None
        try:
            if len(leftx) > 15: 
                left_f = np.polyfit(lefty, leftx, 2)
                left_f_cr = np.polyfit(lefty*self.ym_per_pix, leftx*self.xm_per_pix, 2)
            if len(rightx) > 15:
                right_f = np.polyfit(righty, rightx, 2)
                right_f_cr = np.polyfit(righty*self.ym_per_pix, rightx*self.xm_per_pix, 2)
        except (np.RankWarning, TypeError, np.linalg.LinAlgError): pass # Hata olursa None kalır

        self.left_fit, self.right_fit = left_f if left_f is not None else self.left_fit, right_f if right_f is not None else self.right_fit
        self.left_fit_cr, self.right_fit_cr = left_f_cr if left_f_cr is not None else self.left_fit_cr, right_f_cr if right_f_cr is not None else self.right_fit_cr
        self.ploty = ploty
        return self.left_fit, self.right_fit, self.left_fit_cr, self.right_fit_cr, self.ploty

    def _calculate_curvature_offset(self, warped_shape): # Artık fitleri self'den alacak
        if self.ploty is None: return float('inf'), 0 # Eğer ploty yoksa hesaplama yapma

        y_eval_m = np.max(self.ploty) * self.ym_per_pix
        left_cur, right_cur = float('inf'), float('inf')

        if self.left_fit_cr is not None:
            A, B = self.left_fit_cr[0], self.left_fit_cr[1]
            if A != 0: left_cur = ((1+(2*A*y_eval_m+B)**2)**1.5)/np.absolute(2*A)
        if self.right_fit_cr is not None:
            A, B = self.right_fit_cr[0], self.right_fit_cr[1]
            if A != 0: right_cur = ((1+(2*A*y_eval_m+B)**2)**1.5)/np.absolute(2*A)

        cur = float('inf')
        if left_cur!=float('inf') and right_cur!=float('inf'): cur=(left_cur+right_cur)/2
        elif left_cur!=float('inf'): cur=left_cur
        elif right_cur!=float('inf'): cur=right_cur
        
        offset_m = 0
        y_b_px = warped_shape[0]-1
        lx_b, rx_b = warped_shape[1]*0.25, warped_shape[1]*0.75 # Varsayılan
        if self.left_fit: lx_b = self.left_fit[0]*y_b_px**2+self.left_fit[1]*y_b_px+self.left_fit[2]
        if self.right_fit: rx_b = self.right_fit[0]*y_b_px**2+self.right_fit[1]*y_b_px+self.right_fit[2]
        
        lane_w_px = 0.40 / self.xm_per_pix # Tek şerit genişliği
        if self.left_fit and not self.right_fit: rx_b = lx_b + lane_w_px * 2 # İki şerit
        elif self.right_fit and not self.left_fit: lx_b = rx_b - lane_w_px * 2

        lane_c_px = (lx_b + rx_b) / 2
        veh_c_px = warped_shape[1] / 2
        offset_m = (veh_c_px - lane_c_px) * self.xm_per_pix
        return cur, offset_m

    def _draw_lanes_on_frame(self, output_display_frame, warped_img_ref_shape):
        # Bu fonksiyon output_display_frame'i doğrudan modifiye ETMEMELİ.
        # Bir kopya üzerinde çalışıp onu döndürmeli veya main.py kopya vermeli.
        # En temizi, main.py'nin verdiği frame üzerine çizim yapıp, o frame'i döndürmek.
        # Ya da sadece çizim için gerekli noktaları döndürüp main.py'de çizdirmek.
        # Şimdilik, verilen frame üzerine çizim yapıp onu döndürsün.
        
        if self.ploty is None or (self.left_fit is None and self.right_fit is None):
            # Eğer çizilecek bir şey yoksa, orijinal frame'i (ROI ile) döndür
            cv2.polylines(output_display_frame, [self.roi_poly_to_draw], isClosed=True, color=(0,100,100), thickness=1)
            return output_display_frame

        warp_zero = np.zeros(warped_img_ref_shape[:2], dtype=np.uint8) # warped_img boyutunda
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        if self.left_fit is not None:
            lfx = self.left_fit[0]*self.ploty**2 + self.left_fit[1]*self.ploty + self.left_fit[2]
            pts_l = np.array([np.transpose(np.vstack([lfx, self.ploty]))])
            cv2.polylines(color_warp, np.int_([pts_l]), False, (255,0,0), 10)
        if self.right_fit is not None:
            rfx = self.right_fit[0]*self.ploty**2 + self.right_fit[1]*self.ploty + self.right_fit[2]
            pts_r = np.array([np.transpose(np.vstack([rfx, self.ploty]))])
            cv2.polylines(color_warp, np.int_([pts_r]), False, (0,0,255), 10)
        if self.left_fit is not None and self.right_fit is not None:
            # lfx, rfx zaten yukarıda hesaplandı
            pts = np.hstack((pts_l, pts_r[:, ::-1, :]))
            cv2.fillPoly(color_warp, np.int_([pts]), (0,255,0))

        new_warp = cv2.warpPerspective(color_warp, self.Minv, (output_display_frame.shape[1], output_display_frame.shape[0]))
        result_img = cv2.addWeighted(output_display_frame, 1, new_warp, 0.4, 0)
        cv2.polylines(result_img, [self.roi_poly_to_draw], isClosed=True, color=(0,100,100), thickness=1) # ROI'yi de çiz
        return result_img


    def detect_lanes(self, input_frame_rgb):
        # Bu fonksiyon, üzerine çizim yapılmış bir GÖRÜNTÜ KOPYASI döndürmeli.
        output_display_frame = input_frame_rgb.copy() # Üzerine çizim yapılacak kopya

        preprocessed = self._preprocess_image(input_frame_rgb) # Tespit için orijinali kullan
        warped = self._perspective_warp(preprocessed)
        
        # out_img_diag, kuşbakışı teşhis görüntüsü olacak
        _, _, _, _, out_img_diag = self._find_lane_pixels_sliding_window(warped)
        
        # Polinom uydurma (self.left_fit vb. güncellenir)
        self._fit_polynomial(_, _, _, _, warped.shape) # leftx vb. burada kullanılmıyor, self'e yazılıyor

        curvature, offset_m = self._calculate_curvature_offset(warped.shape)

        # Şeritleri output_display_frame üzerine çiz ve güncellenmiş frame'i al
        final_display_with_lanes = self._draw_lanes_on_frame(output_display_frame, warped.shape)

        # Teşhis görüntüsünü renkliye çevir (eğer griyse)
        if len(out_img_diag.shape) == 2 or out_img_diag.shape[2] == 1:
            diag_to_return = cv2.cvtColor(out_img_diag, cv2.COLOR_GRAY2BGR)
        else:
            diag_to_return = out_img_diag.copy()
            
        return final_display_with_lanes, offset_m, curvature, diag_to_return

    def cleanup(self):
        print("Şerit Tespit Modülü Kapatılıyor...")
        if hasattr(self, 'picam2') and self.picam2.started:
            self.picam2.stop()
        print("Kamera durduruldu.")

if __name__ == "__main__":
    # ... (Test bloğu aynı kalabilir, sadece cv2.imshow çağrılarına dikkat edin)
    # Test bloğundaki imshow'lar okeydir, çünkü main.py çalışmıyordur.
    detector = None
    try:
        # Düşük çözünürlükle test et
        detector = LaneDetector(image_width=320, image_height=240, camera_fps=15) # Veya istediğiniz değerler
        print("LaneDetector Test. 'q' ile çık.")
        cv2.namedWindow("Lane Detection Test", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Diagnostic Warped", cv2.WINDOW_NORMAL)

        while True:
            frame = detector.capture_frame()
            if frame is None: continue

            # detect_lanes şimdi üzerine çizilmiş bir frame döndürüyor
            display_output, offset, curve, diag_img = detector.detect_lanes(frame)
            
            cv2.putText(display_output, f"Off:{offset:.2f} Crv:{curve:.0f}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
            cv2.putText(display_output, f"Off:{offset:.2f} Crv:{curve:.0f}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)


            cv2.imshow("Lane Detection Test", display_output)
            cv2.imshow("Diagnostic Warped", diag_img)

            if cv2.waitKey(1) & 0xFF == ord('q'): break
    except KeyboardInterrupt: print("Durduruldu.")
    except Exception as e: print(f"Hata: {e}"); import traceback; traceback.print_exc()
    finally:
        if detector: detector.cleanup()
        cv2.destroyAllWindows()