# main.py

import cv2
import time
import numpy as np

from motor_kontrol import MotorController
from lane_detector import LaneDetector
from obstacle_detector import ObstacleDetector
from ground_crossing_detector import GroundCrossingDetector

# --- Durum Tanımları ---
class State:
    INITIALIZING = "INITIALIZING"
    LANE_FOLLOWING = "LANE_FOLLOWING"
    OBSTACLE_APPROACHING_TURUNCU = "OBSTACLE_APPROACHING_TURUNCU"
    OBSTACLE_AVOID_PREPARE_TURUNCU = "OBSTACLE_AVOID_PREPARE_TURUNCU"
    OVERTAKING_TURUNCU_CHANGING_LANE = "OVERTAKING_TURUNCU_CHANGING_LANE"
    OVERTAKING_TURUNCU_IN_LEFT_LANE = "OVERTAKING_TURUNCU_IN_LEFT_LANE"
    RETURNING_TO_RIGHT_LANE_AFTER_TURUNCU = "RETURNING_TO_RIGHT_LANE_AFTER_TURUNCU"
    GROUND_CROSSING_APPROACHING = "GROUND_CROSSING_APPROACHING"
    GROUND_CROSSING_WAITING = "GROUND_CROSSING_WAITING"
    ERROR = "ERROR"
    MISSION_COMPLETE = "MISSION_COMPLETE"

# --- Genel Parametreler ---
# Düşük çözünürlük ve FPS ile başla (donma sorunları için)
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_FPS = 15 # Veya 10

BASE_SPEED = 0.30 # Hızı biraz düşür
MAX_SPEED_CAP = 0.40
MIN_SPEED_IN_CURVE = 0.18 # Daha düşük
CURVATURE_THRESHOLD_FOR_SLOWDOWN = 180

STEERING_GAIN_P_STRAIGHT = 0.85
STEERING_GAIN_P_CURVE = 1.25
CURVATURE_THRESHOLD_FOR_GAIN_ADJUST = 220

TURUNCU_ENGEL_GERI_CEKILME_SURESI = 0.5
TURUNCU_ENGEL_GERI_CEKILME_HIZI = 0.20
SOLLAMA_SOL_SERITTE_KALMA_SURESI = 2.5
OVERTAKE_CLEARANCE_CHECK_DURATION = 0.6
SERIT_DEGISTIRME_HIZI = BASE_SPEED * 0.80
SERIT_DEGISTIRME_DIREKSIYON_SOL = -0.65
SERIT_DEGISTIRME_DIREKSIYON_SAG = 0.65

TARGET_LANE_OFFSET_M = 0.0
TARGET_LANE_OFFSET_LEFT_LANE_M = -0.38

SARI_ENGEL_YAVASLAMA_HIZI = 0.18
SARI_ENGEL_KACINMA_DIREKSIYONU_ITME = 0.30
SARI_ENGEL_AKTIF_SURESI = 2.0

GROUND_CROSSING_WAIT_TIME_S = 5.0
GROUND_CROSSING_COOLDOWN_S = 12.0
GROUND_CROSSING_DETECTION_METHOD = "contour"

# --- Yardımcı Fonksiyonlar (Aynı) ---
def get_dynamic_speed(current_curvature, base_speed_ref=BASE_SPEED):
    # ... (önceki gibi)
    if 0 < current_curvature < CURVATURE_THRESHOLD_FOR_SLOWDOWN :
        if current_curvature < 80 : return MIN_SPEED_IN_CURVE # Daha keskin için daha düşük
        else: return (MIN_SPEED_IN_CURVE + base_speed_ref) / 1.8 # Daha yavaşlat
    return min(base_speed_ref, MAX_SPEED_CAP)

def get_dynamic_steering_gain(current_curvature):
    # ... (önceki gibi)
    if 0 < current_curvature < CURVATURE_THRESHOLD_FOR_GAIN_ADJUST:
        return STEERING_GAIN_P_CURVE
    return STEERING_GAIN_P_STRAIGHT

def calculate_steering(offset_m, current_curvature):
    # ... (önceki gibi)
    dynamic_gain = get_dynamic_steering_gain(current_curvature)
    steering_p = -offset_m * dynamic_gain
    return max(-1.0, min(1.0, steering_p))

def apply_motor_speeds(mc, base_speed_val, steering_val):
    # ... (önceki gibi, hassasiyet ayarlanabilir)
    l_s = base_speed_val - steering_val * (base_speed_val * 0.8) 
    r_s = base_speed_val + steering_val * (base_speed_val * 0.8)
    l_s = max(-MAX_SPEED_CAP, min(MAX_SPEED_CAP, l_s))
    r_s = max(-MAX_SPEED_CAP, min(MAX_SPEED_CAP, r_s))
    mc.set_speeds(l_s, r_s)

# --- Ana Program ---
def main():
    current_state = State.INITIALIZING
    mc, ld, od, gcd = None, None, None, None
    state_timer_start, overtake_maneuver_start_time = 0, 0
    last_ground_crossing_stop_time, sari_engel_aktif_ts = 0, 0
    running = True

    # Teşhis için (isteğe bağlı)
    diag_img_lanes_display = np.zeros((CAMERA_HEIGHT // 2, CAMERA_WIDTH // 2, 3), dtype=np.uint8)


    try:
        print("Ana program başlatılıyor...")
        mc = MotorController()
        ld = LaneDetector(image_width=CAMERA_WIDTH, image_height=CAMERA_HEIGHT, camera_fps=CAMERA_FPS)
        od = ObstacleDetector()
        gcd = GroundCrossingDetector(image_width=CAMERA_WIDTH, image_height=CAMERA_HEIGHT)
        
        current_state = State.LANE_FOLLOWING
        print("Tüm modüller yüklendi. Ana döngü başlıyor.")
        cv2.namedWindow("Otonom Araç Kontrol", cv2.WINDOW_AUTOSIZE)

        while running:
            frame_rgb_original = ld.capture_frame()
            if frame_rgb_original is None:
                time.sleep(0.01); continue

            # Her döngüde, üzerine çizim yapılacak ana display_frame'i orijinalden kopyala
            display_frame_processed = frame_rgb_original.copy()

            # 1. Şerit Tespiti
            lane_offset_m, lane_curvature = 0, float('inf') # Varsayılan
            try:
                # ld.detect_lanes, üzerine çizilmiş bir KOPYA ve teşhis görüntüsü döndürür
                display_frame_processed, lane_offset_m, lane_curvature, diag_img_lanes_current = \
                    ld.detect_lanes(frame_rgb_original) # Ham frame'i ver, kopya içinde alınacak
                if diag_img_lanes_current is not None and diag_img_lanes_current.size >0:
                    diag_img_lanes_display = cv2.resize(diag_img_lanes_current, (CAMERA_WIDTH // 3, CAMERA_HEIGHT // 3))
            except Exception as e:
                print(f"Şerit tespiti hatası: {e}")
                # display_frame_processed = frame_rgb_original.copy() # Ham kalsın

            # 2. Engel Tespiti
            turuncu_engeller, sari_engeller = [], []
            try:
                # od.find_obstacles, üzerine çizilmiş bir KOPYA döndürür
                # Girdi olarak bir önceki adımdan gelen (şeritler çizilmiş) frame'i ver
                turuncu_engeller, sari_engeller, display_frame_processed = \
                    od.find_obstacles(display_frame_processed, 
                                      lane_info={"offset_m": lane_offset_m, "image_width_px": CAMERA_WIDTH})
            except Exception as e:
                print(f"Engel tespiti hatası: {e}")

            effective_base_speed = get_dynamic_speed(lane_curvature, BASE_SPEED)

            # 3. Zemin Geçidi Tespiti
            is_ground_crossing_detected_now = False
            can_check_gc = current_state not in [
                State.GROUND_CROSSING_WAITING, State.OBSTACLE_AVOID_PREPARE_TURUNCU,
                State.OVERTAKING_TURUNCU_CHANGING_LANE, State.OVERTAKING_TURUNCU_IN_LEFT_LANE,
                State.RETURNING_TO_RIGHT_LANE_AFTER_TURUNCU
            ]
            if can_check_gc:
                # gcd.detect_crossing, boolean döndürür ve verilen frame üzerine çizer
                is_ground_crossing_detected_now = gcd.detect_crossing(
                    frame_rgb_original, # Tespit için ham frame
                    display_frame_processed, # Çizim için en son işlenmiş frame
                    method=GROUND_CROSSING_DETECTION_METHOD
                )
            
            time_since_last_gc_stop = time.time() - last_ground_crossing_stop_time
            if is_ground_crossing_detected_now and time_since_last_gc_stop > GROUND_CROSSING_COOLDOWN_S:
                if current_state != State.GROUND_CROSSING_APPROACHING:
                    print(f"Zemin geçidi ({time_since_last_gc_stop:.1f}s cooldown sonrası).")
                    current_state = State.GROUND_CROSSING_APPROACHING
                    mc.stop()
            
            # --- Ana Durum Makinesi ---
            # (Durum makinesi mantığı önceki tam kodda olduğu gibi kalacak,
            #  sadece display_frame_processed'i kullanacak ve motor komutlarını verecek)
            # Örnek LANE_FOLLOWING:
            if current_state == State.LANE_FOLLOWING:
                approaching_turuncu_in_my_lane = None
                for obs_t in turuncu_engeller:
                    if obs_t.get('is_in_my_lane', False) and obs_t.get('is_approaching', False):
                        approaching_turuncu_in_my_lane = obs_t; break
                if approaching_turuncu_in_my_lane:
                    current_state = State.OBSTACLE_APPROACHING_TURUNCU; mc.stop()
                else:
                    steering_val = calculate_steering(lane_offset_m, lane_curvature)
                    speed_final = effective_base_speed
                    # Sarı engel kontrolü... (önceki gibi)
                    apply_motor_speeds(mc, speed_final, steering_val)
            
            elif current_state == State.OBSTACLE_APPROACHING_TURUNCU: # ... (diğer durumlar)
                # ...
                pass # Diğer durumların mantığı önceki tam kodda olduğu gibi

            elif current_state == State.GROUND_CROSSING_APPROACHING:
                mc.stop(); print("ZG: Duruldu, bekleniyor...")
                current_state = State.GROUND_CROSSING_WAITING
                state_timer_start = time.time()
                last_ground_crossing_stop_time = time.time()

            elif current_state == State.GROUND_CROSSING_WAITING:
                mc.stop()
                if time.time() - state_timer_start >= GROUND_CROSSING_WAIT_TIME_S:
                    print(f"ZG: {GROUND_CROSSING_WAIT_TIME_S}s bekleme bitti."); current_state = State.LANE_FOLLOWING
            
            # ... (Diğer durumların tam mantığı önceki main.py'den alınacak)


            # --- Görselleştirme (Tek bir yerde) ---
            # Durum, offset vb. metinleri display_frame_processed üzerine çiz
            cv2.putText(display_frame_processed, f"D:{current_state}",(5,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
            cv2.putText(display_frame_processed, f"D:{current_state}",(5,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(50,205,50),1)
            cv2.putText(display_frame_processed, f"O:{lane_offset_m:.2f} C:{lane_curvature:.0f} S:{effective_base_speed:.2f}",(5,40),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0),2)
            cv2.putText(display_frame_processed, f"O:{lane_offset_m:.2f} C:{lane_curvature:.0f} S:{effective_base_speed:.2f}",(5,40),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,0),1)

            # Sarı engel uyarısı
            if sari_engel_aktif_ts > 0 and (time.time() - sari_engel_aktif_ts < SARI_ENGEL_AKTIF_SURESI):
                 cv2.putText(display_frame_processed, "SARI SOLDA!",(CAMERA_WIDTH-150,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
                 cv2.putText(display_frame_processed, "SARI SOLDA!",(CAMERA_WIDTH-150,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
            
            # Zemin geçidi bekleme sayacı
            if current_state == State.GROUND_CROSSING_WAITING:
                 rem_w = max(0, GROUND_CROSSING_WAIT_TIME_S-(time.time()-state_timer_start))
                 cv2.putText(display_frame_processed, f"ZG Bekleme: {rem_w:.1f}s",(CAMERA_WIDTH//2-80,CAMERA_HEIGHT-15),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)
                 cv2.putText(display_frame_processed, f"ZG Bekleme: {rem_w:.1f}s",(CAMERA_WIDTH//2-80,CAMERA_HEIGHT-15),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),1)

            # Teşhis görüntüsünü (lane detector) sağ üste ekle (daha küçük)
            try:
                if diag_img_lanes_display is not None and diag_img_lanes_display.size > 0:
                    h_d, w_d = diag_img_lanes_display.shape[:2]
                    display_frame_processed[5 : 5+h_d, CAMERA_WIDTH-w_d-5 : CAMERA_WIDTH-5] = diag_img_lanes_display
            except Exception: pass # Boyut uyumsuzluğu olursa çökmesin

            cv2.imshow("Otonom Araç Kontrol", display_frame_processed)
            key = cv2.waitKey(1) & 0xFF # Çok önemli! GUI olaylarını işler
            if key == ord('q'): running = False
            
            # ... (Manevra zaman aşımı kontrolü - önceki gibi)

    except KeyboardInterrupt: print("Ctrl+C ile durduruldu.")
    except Exception as e:
        print(f"Ana döngüde HATA: {e}"); import traceback; traceback.print_exc()
        if mc: mc.stop() # Hata durumunda motorları durdur
    finally:
        print("Program sonu. Kaynaklar temizleniyor...")
        if mc: mc.stop(); mc.cleanup()
        if ld: ld.cleanup()
        # gcd'nin özel cleanup'ı yok
        cv2.destroyAllWindows()
        print("Temizlendi. Çıkıldı.")

if __name__ == "__main__":
    main()