# lane_detector.py

import cv2
import numpy as np
from picamera2 import Picamera2
from libcamera import controls
import time

class LaneDetector:
    def __init__(self, image_width=320, image_height=240, camera_fps=20):
        print("Şerit Tespit Modülü Başlatılıyor (Hata Düzeltmeli)...")
        self.image_width = image_width
        self.image_height = image_height
        self.camera_fps = camera_fps

        self.picam2 = Picamera2()
        control_settings = {"FrameRate": self.camera_fps, "AnalogueGain": 1.2, "ExposureTime": 10000}
        config = self.picam2.create_video_configuration(
            main={"size": (self.image_width, self.image_height), "format": "RGB888"},
            lores={"size": (self.image_width // 2, self.image_height // 2), "format": "YUV420"},
            controls=control_settings
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(2.0)
        print(f"Kamera {self.image_width}x{self.image_height} @ {self.camera_fps}FPS ile başlatıldı.")

        roi_top_y_ratio = 0.50
        roi_bottom_y_offset = 5
        top_x_width_ratio = 0.15
        bottom_x_width_ratio = 0.95
        roi_top_y = int(self.image_height * roi_top_y_ratio)

        src_pts_list = [
            (self.image_width//2 - int(self.image_width*top_x_width_ratio/2), roi_top_y),
            (self.image_width//2 + int(self.image_width*top_x_width_ratio/2), roi_top_y),
            (self.image_width//2 + int(self.image_width*bottom_x_width_ratio/2), self.image_height - roi_bottom_y_offset),
            (self.image_width//2 - int(self.image_width*bottom_x_width_ratio/2), self.image_height - roi_bottom_y_offset)
        ]
        self.src_points = np.float32(src_pts_list)
        self.roi_poly_to_draw = np.array([self.src_points], dtype=np.int32)

        dst_width_ratio = 0.9
        self.warped_img_width = int(self.image_width * dst_width_ratio)
        self.warped_img_height = self.image_height
        self.warped_img_size = (self.warped_img_width, self.warped_img_height)
        self.dst_points = np.float32([[0,0],[self.warped_img_width-1,0],[self.warped_img_width-1,self.warped_img_height-1],[0,self.warped_img_height-1]])
        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.Minv = cv2.getPerspectiveTransform(self.dst_points, self.src_points)

        self.xm_per_pix = (0.40 * 2) / (self.warped_img_width * 0.70)
        self.ym_per_pix = 0.20 / 30

        self.left_fit, self.right_fit = None, None
        self.left_fit_cr, self.right_fit_cr = None, None
        self.ploty = np.linspace(0, self.warped_img_height - 1, self.warped_img_height)
        self.stable_offset_m, self.stable_curvature = 0.0, float('inf')
        self.smoothing_alpha = 0.35
        self.last_valid_left_fit, self.last_valid_right_fit = None, None

    def capture_frame(self): return self.picam2.capture_array("main")

    def _image_preprocessing(self, img_rgb):
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5,5),0)
        return cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,19,5)

    def _perspective_transform(self, img_binary):
        return cv2.warpPerspective(img_binary,self.M,self.warped_img_size,flags=cv2.INTER_NEAREST)

    def _find_lane_pixels_sliding_window(self, warped_binary_img):
        out_img_diag = np.dstack((warped_binary_img,)*3)
        hist = np.sum(warped_binary_img[warped_binary_img.shape[0]*3//4:,:],axis=0)
        mid = np.int32(hist.shape[0]//2)
        lxb,rxb = np.argmax(hist[:mid]),np.argmax(hist[mid:])+mid
        if self.left_fit is not None:
            try: lxb=int(np.polyval(self.left_fit,warped_binary_img.shape[0]-1))
            except: pass
        if self.right_fit is not None:
            try: rxb=int(np.polyval(self.right_fit,warped_binary_img.shape[0]-1))
            except: pass

        nwin,margin,minpix=10,int(self.warped_img_width*0.1),40
        win_h=np.int32(warped_binary_img.shape[0]//nwin)
        nz=warped_binary_img.nonzero();nzy,nzx=np.array(nz[0]),np.array(nz[1])
        lxc,rxc=lxb,rxb; l_inds,r_inds=[],[]
        for w in range(nwin):
            ylo,yhi=warped_binary_img.shape[0]-(w+1)*win_h,warped_binary_img.shape[0]-w*win_h
            xl_lo,xl_hi=lxc-margin,lxc+margin;xr_lo,xr_hi=rxc-margin,rxc+margin
            cv2.rectangle(out_img_diag,(xl_lo,ylo),(xl_hi,yhi),(0,255,0),1)
            cv2.rectangle(out_img_diag,(xr_lo,ylo),(xr_hi,yhi),(0,255,0),1)
            good_l=((nzy>=ylo)&(nzy<yhi)&(nzx>=xl_lo)&(nzx<xl_hi)).nonzero()[0]
            good_r=((nzy>=ylo)&(nzy<yhi)&(nzx>=xr_lo)&(nzx<xr_hi)).nonzero()[0]
            l_inds.append(good_l);r_inds.append(good_r)
            if len(good_l)>minpix:lxc=np.int32(np.mean(nzx[good_l]))
            if len(good_r)>minpix:rxc=np.int32(np.mean(nzx[good_r]))
        try:l_inds,r_inds=np.concatenate(l_inds),np.concatenate(r_inds)
        except ValueError:pass
        return nzx[l_inds],nzy[l_inds],nzx[r_inds],nzy[r_inds],out_img_diag

    def _fit_polynomial_from_pixels(self, nzx, nzy, shape):
        if nzx is None or nzy is None or len(nzx)<30 or len(nzy)<30: return None,None
        try:
            fit_px=np.polyfit(nzy,nzx,2)
            fit_cr=None
            if self.xm_per_pix>1e-7 and self.ym_per_pix>1e-7:
                fit_cr=np.polyfit(nzy*self.ym_per_pix,nzx*self.xm_per_pix,2)
            return fit_px,fit_cr
        except: return None,None

    def _update_lane_fits(self, lx, ly, rx, ry, shape):
        nlfit_px,nlfit_cr=self._fit_polynomial_from_pixels(lx,ly,shape)
        if nlfit_px is not None: self.left_fit=nlfit_px; self.last_valid_left_fit=nlfit_px
        if nlfit_cr is not None: self.left_fit_cr=nlfit_cr
        
        nrfit_px,nrfit_cr=self._fit_polynomial_from_pixels(rx,ry,shape)
        if nrfit_px is not None: self.right_fit=nrfit_px; self.last_valid_right_fit=nrfit_px
        if nrfit_cr is not None: self.right_fit_cr=nrfit_cr

        if self.ploty is None or len(self.ploty)!=shape[0]:
            self.ploty=np.linspace(0,shape[0]-1,shape[0])

    def _calculate_curvature_and_offset(self, warped_shape):
        if self.ploty is None or self.ym_per_pix<1e-7 or self.xm_per_pix<1e-7:
            return self.stable_curvature,self.stable_offset_m
        
        y_eval_m=np.mean(self.ploty)*self.ym_per_pix
        y_off_px=warped_shape[0]-1
        cl,cr=float('inf'),float('inf')

        if self.left_fit_cr is not None and isinstance(self.left_fit_cr,np.ndarray) and len(self.left_fit_cr)==3:
            A,B=self.left_fit_cr[0],self.left_fit_cr[1]
            if abs(A)>1e-7: cl=((1+(2*A*y_eval_m+B)**2)**1.5)/abs(2*A)
        if self.right_fit_cr is not None and isinstance(self.right_fit_cr,np.ndarray) and len(self.right_fit_cr)==3:
            A,B=self.right_fit_cr[0],self.right_fit_cr[1]
            if abs(A)>1e-7: cr=((1+(2*A*y_eval_m+B)**2)**1.5)/abs(2*A)
        
        cur_c=float('inf'); vl,vr=(cl!=float('inf')and cl>0),(cr!=float('inf')and cr>0)
        if vl and vr:cur_c=(cl+cr)/2
        elif vl:cur_c=cl
        elif vr:cur_c=cr
        if cur_c!=float('inf'): self.stable_curvature=self.smoothing_alpha*cur_c+(1-self.smoothing_alpha)*self.stable_curvature

        off_m_c=0.0; lxb,rxb=None,None
        if self.left_fit is not None and isinstance(self.left_fit,np.ndarray) and len(self.left_fit)==3:
            lxb=np.polyval(self.left_fit,y_off_px)
        if self.right_fit is not None and isinstance(self.right_fit,np.ndarray) and len(self.right_fit)==3:
            rxb=np.polyval(self.right_fit,y_off_px)
        
        veh_c=warped_shape[1]/2.0
        if lxb is not None and rxb is not None:
            off_m_c=(veh_c-(lxb+rxb)/2.0)*self.xm_per_pix
        elif lxb is not None: off_m_c=(veh_c-(lxb+(0.40/self.xm_per_pix)/2.0))*self.xm_per_pix
        elif rxb is not None: off_m_c=(veh_c-(rxb-(0.40/self.xm_per_pix)/2.0))*self.xm_per_pix
        else: off_m_c=self.stable_offset_m
        self.stable_offset_m=self.smoothing_alpha*off_m_c+(1-self.smoothing_alpha)*self.stable_offset_m
        return self.stable_curvature,self.stable_offset_m

    def _draw_lanes_and_roi(self, out_disp, warped_shape):
        cv2.polylines(out_disp,[self.roi_poly_to_draw],True,(0,180,180),1)
        if self.ploty is None or (self.left_fit is None and self.right_fit is None): return
        
        color_w=np.zeros((warped_shape[0],warped_shape[1],3),dtype=np.uint8)
        thick=12; ptsl,ptsr=None,None
        if self.left_fit is not None and isinstance(self.left_fit,np.ndarray) and len(self.left_fit)==3:
            lfx=np.polyval(self.left_fit,self.ploty)
            ptsl=np.array([np.transpose(np.vstack([lfx,self.ploty]))])
            cv2.polylines(color_w,np.int32(ptsl),False,(255,50,150),thick)
        if self.right_fit is not None and isinstance(self.right_fit,np.ndarray) and len(self.right_fit)==3:
            rfx=np.polyval(self.right_fit,self.ploty)
            ptsr=np.array([np.transpose(np.vstack([rfx,self.ploty]))])
            cv2.polylines(color_w,np.int32(ptsr),False,(150,50,255),thick)
        if ptsl is not None and ptsr is not None:
            cv2.fillPoly(color_w,np.int32([np.hstack((ptsl,ptsr[:,::-1,:]))]),(50,255,150))
        unwarp=cv2.warpPerspective(color_w,self.Minv,(out_disp.shape[1],out_disp.shape[0]))
        cv2.addWeighted(unwarp,0.35,out_disp,1.0,0,dst=out_disp)

    def detect_lanes(self, in_rgb):
        out_disp=in_rgb.copy()
        bin_proc=self._image_preprocessing(in_rgb)
        bin_warp=self._perspective_transform(bin_proc)
        lx,ly,rx,ry,diag_w_img=self._find_lane_pixels_sliding_window(bin_warp)
        self._update_lane_fits(lx,ly,rx,ry,bin_warp.shape)
        cur,off=self._calculate_curvature_and_offset(bin_warp.shape)
        self._draw_lanes_and_roi(out_disp,bin_warp.shape)
        diag_ret=cv2.cvtColor(diag_w_img,cv2.COLOR_GRAY2BGR) if len(diag_w_img.shape)==2 else diag_w_img.copy()
        return out_disp,off,cur,diag_ret

    def cleanup(self):
        print("LD Kapatılıyor...");
        if hasattr(self,'picam2')and self.picam2.started:self.picam2.stop()
        print("Kamera durdu.")

if __name__=="__main__":
    # ... (Test bloğu önceki gibi, aynı kalabilir)
    det=None
    try:
        W,H,FPS=320,240,15;det=LaneDetector(W,H,FPS)
        print("LD Test. q-çık");cv2.namedWindow("LT",cv2.WINDOW_NORMAL);cv2.namedWindow("WD",cv2.WINDOW_NORMAL)
        while True:
            f=det.capture_frame()
            if f is None:continue
            d_o,o,c,diag=det.detect_lanes(f)
            cv2.putText(d_o,f"O:{o:.2f} C:{c:.0f}",(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
            cv2.putText(d_o,f"O:{o:.2f} C:{c:.0f}",(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)
            cv2.imshow("LT",d_o);cv2.imshow("WD",diag)
            if cv2.waitKey(1)&0xFF==ord('q'):break
    except Exception as e:print(f"H:{e}");import traceback;traceback.print_exc()
    finally:
        if det:det.cleanup()
        cv2.destroyAllWindows()