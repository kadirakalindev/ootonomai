# ground_crossing_detector.py

import cv2
import numpy as np

class GroundCrossingDetector:
    def __init__(self, image_width, image_height):
        self.image_width = image_width
        self.image_height = image_height
        self.roi_y_start_ratio = 0.40
        self.roi_y_end_ratio = 0.75
        self.roi_x_margin_ratio = 0.10
        self.beyaz_lower = np.array([0, 0, 155]) # V biraz daha düşük olabilir
        self.beyaz_upper = np.array([180, 80, 255]) # S biraz daha yüksek
        self.hough_rho, self.hough_theta = 1, np.pi/180
        self.hough_threshold = 25 # Biraz daha düşük eşik
        self.hough_min_line_length = int(self.image_width * 0.25)
        self.hough_max_line_gap = int(self.image_width * 0.1) # Daha büyük boşluk
        self.contour_min_area_ratio = 0.004
        self.contour_aspect_ratio_threshold = 4.5
        self.contour_width_roi_ratio_threshold = 0.45 # Biraz daha düşük
        print("Zemin Geçidi Tespit Modülü Başlatıldı.")

    def _get_roi_coords(self):
        h, w = self.image_height, self.image_width
        return int(h*self.roi_y_start_ratio), int(h*self.roi_y_end_ratio), \
               int(w*self.roi_x_margin_ratio), int(w*(1-self.roi_x_margin_ratio))

    def _draw_detected_line_on_frame(self, frame_to_draw_on, line_coords_in_roi, roi_offsets):
        xs_roi, ys_roi = roi_offsets
        if line_coords_in_roi and frame_to_draw_on is not None:
            if len(line_coords_in_roi) == 4: # x1,y1,x2,y2
                x1,y1,x2,y2 = line_coords_in_roi
                cv2.line(frame_to_draw_on, (x1+xs_roi, y1+ys_roi), (x2+xs_roi, y2+ys_roi), (255,0,255), 2)
            elif len(line_coords_in_roi) == 2: # rect (x,y,w,h)
                x,y,w,h = line_coords_in_roi
                cv2.rectangle(frame_to_draw_on, (x+xs_roi, y+ys_roi), (x+xs_roi+w, y+ys_roi+h), (0,255,0),2)


    def detect_horizontal_line_hough(self, frame_rgb_for_detection, frame_to_draw_on_if_detected):
        ys, ye, xs, xe = self._get_roi_coords()
        roi_offsets = (xs,ys)
        roi_gray = cv2.cvtColor(frame_rgb_for_detection[ys:ye, xs:xe], cv2.COLOR_RGB2GRAY)
        if roi_gray.size==0: return False
        edges = cv2.Canny(roi_gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, self.hough_rho,self.hough_theta,self.hough_threshold,
                                minLineLength=self.hough_min_line_length, maxLineGap=self.hough_max_line_gap)
        if frame_to_draw_on_if_detected is not None: # ROI'yi her zaman çiz
            cv2.rectangle(frame_to_draw_on_if_detected, (xs,ys), (xe,ye), (0,150,150),1)

        if lines is not None:
            for line in lines:
                x1,y1,x2,y2 = line[0]
                angle = np.rad2deg(np.arctan2(y2-y1, x2-x1))
                if abs(angle) < 20 or abs(angle-180) < 20 or abs(angle+180) < 20:
                    if abs(x2-x1) > (xe-xs)*0.5:
                        self._draw_detected_line_on_frame(frame_to_draw_on_if_detected, (x1,y1,x2,y2), roi_offsets)
                        return True
        return False

    def detect_horizontal_line_contour(self, frame_rgb_for_detection, frame_to_draw_on_if_detected):
        ys, ye, xs, xe = self._get_roi_coords()
        roi_offsets = (xs,ys)
        roi_rgb = frame_rgb_for_detection[ys:ye, xs:xe]
        if roi_rgb.size==0: return False
        
        hsv_roi = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv_roi, self.beyaz_lower, self.beyaz_upper)
        k_op=np.ones((2,4),np.uint8); k_cl=np.ones((4,12),np.uint8) # Daha ince morfoloji
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_op)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_cl)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if frame_to_draw_on_if_detected is not None: # ROI'yi her zaman çiz
            cv2.rectangle(frame_to_draw_on_if_detected, (xs,ys), (xe,ye), (0,150,150),1)
            # cv2.imshow("Maske_Gecit", mask) # Debug için

        if contours:
            min_abs_area = (ye-ys)*(xe-xs)*self.contour_min_area_ratio
            for cnt in sorted(contours,key=cv2.contourArea,reverse=True):
                if cv2.contourArea(cnt) < min_abs_area: continue
                x_c,y_c,w_c,h_c = cv2.boundingRect(cnt)
                ar = w_c/(h_c+1e-6); wrr = w_c/(roi_rgb.shape[1]+1e-6)
                if ar > self.contour_aspect_ratio_threshold and wrr > self.contour_width_roi_ratio_threshold:
                    self._draw_detected_line_on_frame(frame_to_draw_on_if_detected, (x_c,y_c,w_c,h_c), roi_offsets)
                    return True
        return False

    def detect_crossing(self, frame_rgb_for_detection, frame_to_draw_on_if_detected, method="contour"):
        if method == "hough":
            return self.detect_horizontal_line_hough(frame_rgb_for_detection, frame_to_draw_on_if_detected)
        return self.detect_horizontal_line_contour(frame_rgb_for_detection, frame_to_draw_on_if_detected)

if __name__ == "__main__":
    # ... (Test bloğu imshow kullanabilir, sorun değil) ...
    from picamera2 import Picamera2
    CAM_WIDTH, CAM_HEIGHT = 320, 240
    picam2_test = Picamera2()
    # ... (config, start) ...
    # Test bloğu önceki gibi kalabilir, sadece detect_crossing'in frame_to_draw_on_if_detected
    # parametresine debug_display_frame'i yollayın.