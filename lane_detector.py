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
        # Donma sorunları için düşük çözünürlük ve FPS ile deneyebilirsiniz:
        # self.image_width = 320
        # self.image_height = 240
        # self.camera_fps = 15

        config = self.picam2.create_video_configuration(
            main={"size": (self.image_width, self.image_height), "format": "RGB888"},
            controls={"FrameRate": self.camera_fps}
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(1.5) 
        print(f"Kamera {self.image_width}x{self.image_height} @ {self.camera_fps}FPS ile başlatıldı.")

        # Perspektif dönüşümü için kaynak ve hedef noktaları (AYARLANMALI)
        roi_bottom_offset = 10
        roi_top_y = int(self.image_height * 0.58) # Biraz daha aşağıdan alabiliriz
        top_x_margin = int(self.image_width * 0.15) # Daha dar bir üst kısım

        self.src_points = np.float32([
            (self.image_width // 2 - top_x_margin, roi_top_y),
            (self.image_width // 2 + top_x_margin, roi_top_y),
            (self.image_width - 10, self.image_height - roi_bottom_offset), 
            (10, self.image_height - roi_bottom_offset)
        ])

        dst_width_ratio = 0.8 
        self.warped_img_size = (int(self.image_width * dst_width_ratio), self.image_height)

        self.dst_points = np.float32([
            [0, 0],
            [self.warped_img_size[0] - 1, 0],
            [self.warped_img_size[0] - 1, self.warped_img_size[1] - 1],
            [0, self.warped_img_size[1] - 1]
        ])

        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.Minv = cv2.getPerspectiveTransform(self.dst_points, self.src_points)

        # Metre başına piksel (KALİBRASYON GEREKTİRİR!)
        self.xm_per_pix = (0.40 * 2) / (self.warped_img_size[0] * 0.75) # 2 şerit ~75% kaplıyorsa
        self.ym_per_pix = 0.20 / 30 # Örnek: 20cm / 30 piksel (kuşbakışında ölçülmeli)

        self.left_fit = None
        self.right_fit = None
        self.left_fit_cr = None
        self.right_fit_cr = None
        self.ploty = None # _fit_polynomial içinde ayarlanacak
        self.roi_poly_to_draw = np.array([self.src_points], dtype=np.int32)

    def capture_frame(self):
        return self.picam2.capture_array("main")

    def _preprocess_image(self, img_rgb):
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Siyah zemin beyaz şerit için THRESH_BINARY_INV
        # blockSize ve C değerleri ışığa göre ayarlanmalı.
        thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 15, 4) # 13,3 veya 15,4 deneyin
        return thresholded

    def _perspective_warp(self, img):
        return cv2.warpPerspective(img, self.M, self.warped_img_size, flags=cv2.INTER_LINEAR)

    def _find_lane_pixels_sliding_window(self, warped_img):
        histogram = np.sum(warped_img[warped_img.shape[0]//2:, :], axis=0)
        out_img_diag = np.dstack((warped_img, warped_img, warped_img))
        
        midpoint = np.int32(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        nwindows = 10 # Pencere sayısını artırabiliriz
        margin = int(self.warped_img_size[0] * 0.12) # Biraz daha daraltabiliriz
        minpix = 40

        window_height = np.int32(warped_img.shape[0]//nwindows)
        nonzero = warped_img.nonzero()
        nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])

        leftx_current, rightx_current = leftx_base, rightx_base
        left_lane_inds, right_lane_inds = [], []

        for window in range(nwindows):
            win_y_low = warped_img.shape[0]-(window+1)*window_height
            win_y_high = warped_img.shape[0]-window*window_height
            # ... (pencere çizimi ve indis bulma önceki gibi) ...
            cv2.rectangle(out_img_diag,(leftx_current-margin,win_y_low), (leftx_current+margin,win_y_high),(0,255,0),1)
            cv2.rectangle(out_img_diag,(rightx_current-margin,win_y_low), (rightx_current+margin,win_y_high),(0,255,0),1)

            good_left = ((nonzeroy>=win_y_low)&(nonzeroy<win_y_high)&(nonzerox>=leftx_current-margin)&(nonzerox<leftx_current+margin)).nonzero()[0]
            good_right = ((nonzeroy>=win_y_low)&(nonzeroy<win_y_high)&(nonzerox>=rightx_current-margin)&(nonzerox<rightx_current+margin)).nonzero()[0]
            left_lane_inds.append(good_left)
            right_lane_inds.append(good_right)
            if len(good_left)>minpix: leftx_current=np.int32(np.mean(nonzerox[good_left]))
            if len(good_right)>minpix: rightx_current=np.int32(np.mean(nonzerox[good_right]))
        
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError: pass # Boşsa birleştirme

        return nonzerox[left_lane_inds], nonzeroy[left_lane_inds], \
               nonzerox[right_lane_inds], nonzeroy[right_lane_inds], out_img_diag

    def _fit_polynomial(self, leftx, lefty, rightx, righty, warped_shape):
        # self.ploty'yi burada ayarla
        self.ploty = np.linspace(0, warped_shape[0]-1, warped_shape[0])
        
        left_f_temp, right_f_temp = None, None
        left_f_cr_temp, right_f_cr_temp = None, None

        min_points_for_fit = 20 # Fit için minimum nokta sayısı

        try:
            if len(leftx) > min_points_for_fit and len(lefty) > min_points_for_fit:
                left_f_temp = np.polyfit(lefty, leftx, 2)
                if self.ym_per_pix > 1e-6 and self.xm_per_pix > 1e-6:
                    left_f_cr_temp = np.polyfit(lefty*self.ym_per_pix, leftx*self.xm_per_pix, 2)
            if len(rightx) > min_points_for_fit and len(righty) > min_points_for_fit:
                right_f_temp = np.polyfit(righty, rightx, 2)
                if self.ym_per_pix > 1e-6 and self.xm_per_pix > 1e-6:
                    right_f_cr_temp = np.polyfit(righty*self.ym_per_pix, rightx*self.xm_per_pix, 2)
        except (np.RankWarning, TypeError, np.linalg.LinAlgError) as e:
            print(f"Polinom uydurma sırasında uyarı/hata: {e}")
            # Hata durumunda None kalacaklar

        # self.'deki fitleri güncelle (eğer yeni fit geçerliyse)
        if left_f_temp is not None: self.left_fit = left_f_temp
        if right_f_temp is not None: self.right_fit = right_f_temp
        if left_f_cr_temp is not None: self.left_fit_cr = left_f_cr_temp
        if right_f_cr_temp is not None: self.right_fit_cr = right_f_cr_temp
        
        # Fonksiyonun dönüş değerleri artık kullanılmıyor, self güncelleniyor.
        # Ama yine de tutarlılık için döndürebiliriz.
        return self.left_fit, self.right_fit, self.left_fit_cr, self.right_fit_cr

    def _calculate_curvature_offset(self, warped_shape):
        if self.ploty is None: return float('inf'), 0.0 # ploty yoksa çık

        y_eval_m = np.max(self.ploty) * self.ym_per_pix if self.ym_per_pix > 1e-6 else np.max(self.ploty)
        left_cur, right_cur = float('inf'), float('inf')

        if self.left_fit_cr is not None: # ÖNEMLİ: None kontrolü
            A, B = self.left_fit_cr[0], self.left_fit_cr[1]
            if abs(A) > 1e-7:  # Çok küçük A değerleri sonsuz eğriliğe yol açabilir
                left_cur = ((1 + (2*A*y_eval_m + B)**2)**1.5) / np.absolute(2*A)
        
        if self.right_fit_cr is not None: # ÖNEMLİ: None kontrolü
            A, B = self.right_fit_cr[0], self.right_fit_cr[1]
            if abs(A) > 1e-7:
                right_cur = ((1 + (2*A*y_eval_m + B)**2)**1.5) / np.absolute(2*A)

        cur = float('inf')
        valid_left_cur = left_cur != float('inf') and left_cur > 0
        valid_right_cur = right_cur != float('inf') and right_cur > 0

        if valid_left_cur and valid_right_cur: cur = (left_cur + right_cur) / 2
        elif valid_left_cur: cur = left_cur
        elif valid_right_cur: cur = right_cur
        
        offset_m = 0.0
        y_bottom_px = warped_shape[0] - 1
        lx_b, rx_b = None, None # Başlangıçta None

        if self.left_fit is not None: # ÖNEMLİ: None kontrolü
            lx_b = self.left_fit[0]*y_bottom_px**2 + self.left_fit[1]*y_bottom_px + self.left_fit[2]
        if self.right_fit is not None: # ÖNEMLİ: None kontrolü
            rx_b = self.right_fit[0]*y_bottom_px**2 + self.right_fit[1]*y_bottom_px + self.right_fit[2]
        
        lane_center_px = warped_shape[1] / 2 # Varsayılan (eğer şerit bulunamazsa offset 0 olur)
        if lx_b is not None and rx_b is not None:
            lane_center_px = (lx_b + rx_b) / 2
        elif lx_b is not None and self.xm_per_pix > 1e-6: # Sadece sol şerit
            lane_center_px = lx_b + (0.40 / self.xm_per_pix) # Tek şerit genişliğinin yarısı kadar sağda
        elif rx_b is not None and self.xm_per_pix > 1e-6: # Sadece sağ şerit
            lane_center_px = rx_b - (0.40 / self.xm_per_pix)
            
        vehicle_center_px = warped_shape[1] / 2
        offset_m = (vehicle_center_px - lane_center_px) * self.xm_per_pix if self.xm_per_pix > 1e-6 else 0.0
        return cur, offset_m

    def _draw_lanes_on_frame(self, output_display_frame, warped_img_ref_shape):
        # output_display_frame'i doğrudan modifiye et, main.py kopya veriyor.
        cv2.polylines(output_display_frame, [self.roi_poly_to_draw], isClosed=True, color=(0,100,100), thickness=1) # ROI her zaman çizilsin

        if self.ploty is None or (self.left_fit is None and self.right_fit is None):
            return # Çizilecek şerit yoksa çık (ROI zaten çizildi)

        warp_zero = np.zeros(warped_img_ref_shape[:2], dtype=np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        pts_l, pts_r = None, None # Başlangıçta None
        if self.left_fit is not None:
            lfx = self.left_fit[0]*self.ploty**2 + self.left_fit[1]*self.ploty + self.left_fit[2]
            pts_l = np.array([np.transpose(np.vstack([lfx, self.ploty]))])
            cv2.polylines(color_warp, np.int32([pts_l]), False, (255,0,0), 10)
        if self.right_fit is not None:
            rfx = self.right_fit[0]*self.ploty**2 + self.right_fit[1]*self.ploty + self.right_fit[2]
            pts_r = np.array([np.transpose(np.vstack([rfx, self.ploty]))])
            cv2.polylines(color_warp, np.int32([pts_r]), False, (0,0,255), 10)
        
        if pts_l is not None and pts_r is not None: # Her iki şerit de varsa alanı boya
            pts = np.hstack((pts_l, pts_r[:, ::-1, :])) # ::-1 sağ şeridi ters çevirir
            cv2.fillPoly(color_warp, np.int32([pts]), (0,255,0))

        new_warp = cv2.warpPerspective(color_warp, self.Minv, (output_display_frame.shape[1], output_display_frame.shape[0]))
        # cv2.addWeighted ile output_display_frame'i modifiye et
        cv2.addWeighted(new_warp, 0.4, output_display_frame, 1, 0, dst=output_display_frame)
        # Bu satır output_display_frame'i yerinde günceller.
        # Eğer döndürmek gerekirse: return cv2.addWeighted(output_display_frame, 1, new_warp, 0.4, 0)


    def detect_lanes(self, input_frame_rgb):
        output_display_frame = input_frame_rgb.copy() # Ana çizim için kopya

        preprocessed = self._preprocess_image(input_frame_rgb)
        warped = self._perspective_warp(preprocessed)
        
        lx, ly, rx, ry, out_img_diag = self._find_lane_pixels_sliding_window(warped)
        
        self._fit_polynomial(lx, ly, rx, ry, warped.shape) # Bu self.left_fit vb. ayarlar
        curvature, offset_m = self._calculate_curvature_offset(warped.shape) # Bu self'den okur

        # _draw_lanes_on_frame, output_display_frame'i yerinde günceller
        self._draw_lanes_on_frame(output_display_frame, warped.shape)

        diag_to_return = cv2.cvtColor(out_img_diag, cv2.COLOR_GRAY2BGR) \
                         if len(out_img_diag.shape)==2 or out_img_diag.shape[2]==1 \
                         else out_img_diag.copy()
            
        return output_display_frame, offset_m, curvature, diag_to_return

    def cleanup(self):
        print("Şerit Tespit Modülü Kapatılıyor...")
        if hasattr(self, 'picam2') and self.picam2.started:
            self.picam2.stop()
        print("Kamera durduruldu.")

if __name__ == "__main__":
    detector = None
    try:
        detector = LaneDetector(image_width=320, image_height=240, camera_fps=15)
        print("LaneDetector Test. 'q' ile çık.")
        cv2.namedWindow("Lane Test", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Warped Diag", cv2.WINDOW_NORMAL)
        while True:
            frame = detector.capture_frame()
            if frame is None: continue
            display_out, offset, curve, diag_img = detector.detect_lanes(frame)
            cv2.putText(display_out, f"O:{offset:.2f} C:{curve:.0f}",(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
            cv2.putText(display_out, f"O:{offset:.2f} C:{curve:.0f}",(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)
            cv2.imshow("Lane Test", display_out)
            cv2.imshow("Warped Diag", diag_img)
            if cv2.waitKey(1)&0xFF==ord('q'): break
    except Exception as e: print(f"Hata: {e}"); import traceback; traceback.print_exc()
    finally:
        if detector: detector.cleanup()
        cv2.destroyAllWindows()