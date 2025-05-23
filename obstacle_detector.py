# obstacle_detector.py

import cv2
import numpy as np

class ObstacleDetector:
    def __init__(self):
        print("Engel Tespit Modülü Başlatılıyor (Kamerayı Kaplama Kriteri)...")
        self.turuncu_lower = np.array([5, 80, 80])   # Daha geniş aralıklar
        self.turuncu_upper = np.array([28, 255, 255])
        self.sari_lower = np.array([18, 80, 80])
        self.sari_upper = np.array([38, 255, 255])

        self.min_contour_area_ratio = 0.002 # Çok küçükleri ele
        self.max_contour_area_ratio = 0.90  # Neredeyse tüm ROI'yi kaplayabilir
        
        # !!! YAKLAŞMA EŞİĞİ (KAMERAYI KAPLAMA) - YÜKSEK AYARLANMALI !!!
        self.approach_threshold_ratio_turuncu = 0.35 # ROI'nin %35'i (veya daha yüksek)
        self.approach_threshold_ratio_sari = 0.08    

        # ROI (Engellerin, özellikle yakın olanların, görüneceği alan)
        self.roi_y_start_ratio = 0.35  # Görüntünün ortalarından başla
        self.roi_y_end_ratio = 0.98    # Neredeyse en alta kadar
        self.roi_x_margin_ratio = 0.05 # Kenarlardan az boşluk (geniş bakış)
        print(f"ObstacleDetector: Turuncu Yaklaşma Eşiği = {self.approach_threshold_ratio_turuncu}")

    def _get_roi_coords(self, frame_shape): # (Aynı)
        h,w=frame_shape[:2]; return int(h*self.roi_y_start_ratio),int(h*self.roi_y_end_ratio), \
               int(w*self.roi_x_margin_ratio),int(w*(1-self.roi_x_margin_ratio))

    def _is_obstacle_in_center_region(self, obs_cx_roi, roi_w): # (Aynı)
        return roi_w*0.20 < obs_cx_roi < roi_w*0.80 # Daha geniş merkez bölge

    def _is_obstacle_on_left_of_center(self, obs_cx_roi, roi_w): # (Aynı)
        return obs_cx_roi < roi_w*0.50 # Tam sol yarı

    def find_obstacles(self, input_frame_rgb, lane_info=None): # (Çizim ve mantık önceki gibi)
        # (Önceki yanıttaki find_obstacles içeriği buraya gelecek.
        #  Sadece __init__ içindeki yeni parametreleri kullanacak.)
        out_disp=input_frame_rgb.copy();hsv_orig=cv2.cvtColor(input_frame_rgb,cv2.COLOR_RGB2HSV)
        ys,ye,xs,xe=self._get_roi_coords(input_frame_rgb.shape)
        roi_hsv=hsv_orig[ys:ye,xs:xe]
        if roi_hsv.size==0:return [],[],out_disp
        roi_area=(ye-ys)*(xe-xs);roi_w=(xe-xs)
        cv2.rectangle(out_disp,(xs,ys),(xe,ye),(180,180,0),1)
        t_list,s_list=[],[]
        #Turuncu
        mask_t=cv2.inRange(roi_hsv,self.turuncu_lower,self.turuncu_upper)
        k=np.ones((3,3),np.uint8);mask_t=cv2.erode(mask_t,k,iterations=1);mask_t=cv2.dilate(mask_t,k,iterations=2)
        cnts_t,_=cv2.findContours(mask_t,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts_t:
            area=cv2.contourArea(c);ratio=area/roi_area if roi_area>0 else 0
            if self.min_contour_area_ratio<ratio<self.max_contour_area_ratio:
                xr,yr,wr,hr=cv2.boundingRect(c);gx,gy=xr+xs,yr+ys;cx_r=xr+wr//2
                is_appr=ratio>self.approach_threshold_ratio_turuncu
                is_in_l=self._is_obstacle_in_center_region(cx_r,roi_w)
                if is_in_l:
                    t_list.append({'rect':(gx,gy,wr,hr),'area_ratio':ratio,'is_approaching':is_appr,'is_in_my_lane':is_in_l})
                    clr=(0,0,255)if is_appr else(0,165,255)
                    cv2.rectangle(out_disp,(gx,gy),(gx+wr,gy+hr),clr,2)
                    cv2.putText(out_disp,f"T{ratio:.2f}{' YAKIN'if is_appr else''}",(gx,gy-5),cv2.FONT_HERSHEY_SIMPLEX,0.35,clr,1)
        #Sarı (Benzer şekilde)
        mask_s=cv2.inRange(roi_hsv,self.sari_lower,self.sari_upper) # ... (erode, dilate, findContours)
        mask_s=cv2.erode(mask_s,k,iterations=1);mask_s=cv2.dilate(mask_s,k,iterations=2)
        cnts_s,_=cv2.findContours(mask_s,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts_s: # Sarı için döngü
            area=cv2.contourArea(c);ratio=area/roi_area if roi_area > 0 else 0
            if self.min_contour_area_ratio < ratio < self.max_contour_area_ratio:
                xr,yr,wr,hr=cv2.boundingRect(c);gx,gy=xr+xs,yr+ys;cx_r=xr+wr//2
                is_on_l = self._is_obstacle_on_left_of_center(cx_r, roi_w)
                is_appr_s = ratio > self.approach_threshold_ratio_sari
                if is_on_l:
                    s_list.append({'rect':(gx,gy,wr,hr),'area_ratio':ratio,'is_on_left_lane':is_on_l,'is_approaching':is_appr_s})
                    clr=(0,200,200)if is_appr_s else(0,255,255)
                    cv2.rectangle(out_disp,(gx,gy),(gx+wr,gy+hr),clr,2)
                    cv2.putText(out_disp,f"S{ratio:.2f}",(gx,gy-5),cv2.FONT_HERSHEY_SIMPLEX,0.35,clr,1)

        return t_list, s_list, out_disp

if __name__ == "__main__":
    # ... (Test bloğu aynı, yeni parametrelerle test edin)
    from picamera2 import Picamera2; import time
    # ... (picam2 init, ObstacleDetector init,namedWindow)
    # Test bloğunun geri kalanı bir önceki yanıttaki gibi.
    # Sadece detector.approach_threshold_ratio_turuncu değerini print ettirin.
    W,H=320,240;p2=Picamera2();cf=p2.create_preview_configuration(main={"size":(W,H),"format":"RGB888"})
    p2.configure(cf);p2.start();time.sleep(1.5);det=ObstacleDetector()
    cv2.namedWindow("OT",cv2.WINDOW_NORMAL);print(f"Turuncu Yaklaşma:{det.approach_threshold_ratio_turuncu}")
    try:
        while True:
            fr=p2.capture_array("main");
            if fr is None:continue
            t,s,dsp=det.find_obstacles(fr)
            # ...
            cv2.imshow("OT",dsp)
            if cv2.waitKey(1)&0xFF==ord('q'):break
    finally:p2.stop();cv2.destroyAllWindows()