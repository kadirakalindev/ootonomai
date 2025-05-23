# main.py

import cv2
import time
import numpy as np

from motor_kontrol import MotorController
from lane_detector import LaneDetector
from obstacle_detector import ObstacleDetector
from ground_crossing_detector import GroundCrossingDetector

# --- Durum Tanımları --- (Aynı)
class State:
    INITIALIZING="INITIALIZING"; LANE_FOLLOWING="LANE_FOLLOWING"
    OBSTACLE_APPROACHING_TURUNCU="OBSTACLE_APPROACHING_TURUNCU"
    OBSTACLE_AVOID_PREPARE_TURUNCU="OBSTACLE_AVOID_PREPARE_TURUNCU"
    OVERTAKING_TURUNCU_CHANGING_LANE_TO_LEFT="OVERTAKING_TURUNCU_CHANGING_LANE_TO_LEFT"
    OVERTAKING_TURUNCU_PASSING_IN_LEFT_LANE="OVERTAKING_TURUNCU_PASSING_IN_LEFT_LANE"
    OVERTAKING_TURUNCU_CHANGING_LANE_TO_RIGHT="OVERTAKING_TURUNCU_CHANGING_LANE_TO_RIGHT"
    GROUND_CROSSING_APPROACHING="GROUND_CROSSING_APPROACHING"
    GROUND_CROSSING_WAITING="GROUND_CROSSING_WAITING"
    ERROR="ERROR"; MISSION_COMPLETE="MISSION_COMPLETE"

# --- Genel Parametreler --- (Bir önceki main.py'deki gibi, özellikle SOLLAMA_ parametreleri)
CAMERA_WIDTH = 320; CAMERA_HEIGHT = 240; CAMERA_FPS = 15
BASE_SPEED = 0.35; MAX_SPEED_CAP = 0.42; MIN_SPEED_IN_CURVE = 0.20
CURVATURE_THRESHOLD_FOR_SLOWDOWN = 180
STEERING_GAIN_P_STRAIGHT = 0.90; STEERING_GAIN_P_CURVE = 1.35
CURVATURE_THRESHOLD_FOR_GAIN_ADJUST = 220
TURUNCU_ENGEL_GERI_CEKILME_SURESI = 0.4; TURUNCU_ENGEL_GERI_CEKILME_HIZI = 0.28
SOLLAMA_SERIT_DEGISTIRME_SURESI_SOLA = 0.7; SOLLAMA_SERIT_DEGISTIRME_HIZI_SOLA = 0.45
SOLLAMA_SERIT_DEGISTIRME_DIREKSIYON_SOL = -0.85
SOLLAMA_SOL_SERITTE_ILERLEME_SURESI = 1.0; SOLLAMA_SOL_SERITTE_ILERLEME_HIZI = 0.50
SOLLAMA_SERIT_DEGISTIRME_SURESI_SAGA = 0.8; SOLLAMA_SERIT_DEGISTIRME_HIZI_SAGA = 0.40
SOLLAMA_SERIT_DEGISTIRME_DIREKSIYON_SAG = 0.80
TARGET_LANE_OFFSET_M = 0.0; TARGET_LANE_OFFSET_LEFT_LANE_M = -0.38
SARI_ENGEL_YAVASLAMA_HIZI = 0.18; SARI_ENGEL_KACINMA_DIREKSIYONU_ITME = 0.35
SARI_ENGEL_AKTIF_SURESI = 2.0
GROUND_CROSSING_WAIT_TIME_S = 5.0; GROUND_CROSSING_COOLDOWN_S = 12.0
GROUND_CROSSING_DETECTION_METHOD = "contour"

# YENİ: Sollama sonrası cooldown süresi
OVERTAKE_COOLDOWN_S = 7.0 # saniye (bu süre boyunca tekrar sollama yapılmaz)

# --- Yardımcı Fonksiyonlar (Aynı) ---
def get_dynamic_speed(c,b=BASE_SPEED): # ... (önceki gibi)
    if 0<c<CURVATURE_THRESHOLD_FOR_SLOWDOWN: return MIN_SPEED_IN_CURVE if c<80 else (MIN_SPEED_IN_CURVE+b)/1.9
    return min(b,MAX_SPEED_CAP)
def get_dynamic_steering_gain(c): # ... (önceki gibi)
    return STEERING_GAIN_P_CURVE if 0<c<CURVATURE_THRESHOLD_FOR_GAIN_ADJUST else STEERING_GAIN_P_STRAIGHT
def calculate_steering(o,c): # ... (önceki gibi)
    return max(-1.,min(1.,-o*get_dynamic_steering_gain(c)))
def apply_motor_speeds(mc,b,s): # ... (önceki gibi)
    ls,rs = b-s*(b*0.85), b+s*(b*0.85)
    mc.set_speeds(max(-MAX_SPEED_CAP,min(MAX_SPEED_CAP,ls)),max(-MAX_SPEED_CAP,min(MAX_SPEED_CAP,rs)))

# --- Ana Program ---
def main():
    current_state = State.INITIALIZING
    mc, ld, od, gcd = None, None, None, None
    state_timer_start, overtake_maneuver_start_time = 0, 0
    last_ground_crossing_stop_time, sari_engel_aktif_ts = 0, 0
    
    # YENİ: En son sollama bitiş zamanı
    last_overtake_finish_time = 0

    running = True
    diag_img_lanes_display = np.zeros((CAMERA_HEIGHT//3, CAMERA_WIDTH//3,3),dtype=np.uint8)

    try:
        print("Ana program başlatılıyor...")
        mc = MotorController(); ld = LaneDetector(CAMERA_WIDTH,CAMERA_HEIGHT,CAMERA_FPS)
        od = ObstacleDetector(); gcd = GroundCrossingDetector(CAMERA_WIDTH,CAMERA_HEIGHT)
        current_state = State.LANE_FOLLOWING
        print("Modüller yüklendi. Döngü başlıyor.")
        cv2.namedWindow("Otonom Araç", cv2.WINDOW_AUTOSIZE)

        while running:
            frame_rgb_original = ld.capture_frame()
            if frame_rgb_original is None: time.sleep(0.01); continue
            display_frame_processed = frame_rgb_original.copy()

            # 1. Şerit Tespiti
            lane_offset_m, lane_curvature = 0, float('inf')
            try:
                display_frame_processed, lane_offset_m, lane_curvature, diag_img_lanes_current = \
                    ld.detect_lanes(frame_rgb_original)
                if diag_img_lanes_current is not None and diag_img_lanes_current.size >0:
                    diag_img_lanes_display = cv2.resize(diag_img_lanes_current, (CAMERA_WIDTH//3, CAMERA_HEIGHT//3))
            except Exception as e: print(f"LANE ERR: {e}")

            # 2. Engel Tespiti
            turuncu_engeller, sari_engeller = [], []
            try:
                turuncu_engeller, sari_engeller, display_frame_processed = \
                    od.find_obstacles(display_frame_processed, 
                                      lane_info={"offset_m": lane_offset_m, "image_width_px": CAMERA_WIDTH})
            except Exception as e: print(f"OBSTACLE ERR: {e}")

            effective_base_speed = get_dynamic_speed(lane_curvature, BASE_SPEED)

            # 3. Zemin Geçidi Tespiti
            # ... (önceki tam koddaki gibi) ...
            is_gc_detected = False
            can_check_gc = current_state not in [ State.GROUND_CROSSING_WAITING, State.OBSTACLE_AVOID_PREPARE_TURUNCU,
                State.OVERTAKING_TURUNCU_CHANGING_LANE_TO_LEFT, State.OVERTAKING_TURUNCU_PASSING_IN_LEFT_LANE,
                State.OVERTAKING_TURUNCU_CHANGING_LANE_TO_RIGHT ]
            if can_check_gc:
                is_gc_detected = gcd.detect_crossing(frame_rgb_original,display_frame_processed,method=GROUND_CROSSING_DETECTION_METHOD)
            time_since_last_gc_stop = time.time()-last_ground_crossing_stop_time
            if is_gc_detected and time_since_last_gc_stop > GROUND_CROSSING_COOLDOWN_S:
                if current_state != State.GROUND_CROSSING_APPROACHING:
                    current_state=State.GROUND_CROSSING_APPROACHING; mc.stop()
            
            # --- Ana Durum Makinesi ---
            if current_state == State.LANE_FOLLOWING:
                # Sollama için cooldown kontrolü
                time_since_last_overtake = time.time() - last_overtake_finish_time
                can_attempt_overtake = time_since_last_overtake > OVERTAKE_COOLDOWN_S

                approaching_turuncu = None
                if can_attempt_overtake: # Sadece cooldown süresi dolduysa turuncu engeli kontrol et
                    for obs_t in turuncu_engeller:
                        if obs_t.get('is_in_my_lane') and obs_t.get('is_approaching'):
                            approaching_turuncu=obs_t; break
                
                if approaching_turuncu:
                    print(f"Turuncu engel yaklaşıyor. Cooldown sonrası sollama ({time_since_last_overtake:.1f}s).")
                    current_state=State.OBSTACLE_APPROACHING_TURUNCU; mc.stop()
                else:
                    # ... (LANE_FOLLOWING'deki normal sürüş ve sarı engel kontrolü - önceki gibi) ...
                    s_val=calculate_steering(lane_offset_m,lane_curvature);spd_final=effective_base_speed
                    is_sari_act_now=False
                    for obs_s in sari_engeller:
                        if obs_s.get('is_on_left_lane')and obs_s.get('is_approaching'):is_sari_act_now=True;sari_engel_aktif_ts=time.time();break
                    is_sari_cooldown=(sari_engel_aktif_ts>0 and time.time()-sari_engel_aktif_ts<SARI_ENGEL_AKTIF_SURESI)
                    if is_sari_act_now or is_sari_cooldown:
                        spd_final=min(spd_final,SARI_ENGEL_YAVASLAMA_HIZI)
                        if lane_offset_m<-0.05 or s_val<-0.1:s_val+=SARI_ENGEL_KACINMA_DIREKSIYONU_ITME;s_val=min(s_val,0.5)
                    apply_motor_speeds(mc,spd_final,s_val)


            elif current_state == State.OBSTACLE_APPROACHING_TURUNCU:
                print("Engel için duruldu. Geri çekilmeye hazırlanılıyor.")
                current_state = State.OBSTACLE_AVOID_PREPARE_TURUNCU
                state_timer_start = time.time()
                mc.backward(TURUNCU_ENGEL_GERI_CEKILME_HIZI)

            elif current_state == State.OBSTACLE_AVOID_PREPARE_TURUNCU:
                if time.time() - state_timer_start > TURUNCU_ENGEL_GERI_CEKILME_SURESI:
                    mc.stop(); print("Geri çekilme bitti. Sol şeride geçiliyor.")
                    current_state = State.OVERTAKING_TURUNCU_CHANGING_LANE_TO_LEFT
                    state_timer_start = time.time() 
                    overtake_maneuver_start_time = time.time() 
            
            elif current_state == State.OVERTAKING_TURUNCU_CHANGING_LANE_TO_LEFT:
                apply_motor_speeds(mc, SOLLAMA_SERIT_DEGISTIRME_HIZI_SOLA, SOLLAMA_SERIT_DEGISTIRME_DIREKSIYON_SOL)
                if time.time() - state_timer_start > SOLLAMA_SERIT_DEGISTIRME_SURESI_SOLA:
                    print("Sol şeride geçildi. Engeli geçmek için ilerleniyor.")
                    current_state = State.OVERTAKING_TURUNCU_PASSING_IN_LEFT_LANE
                    state_timer_start = time.time()
            
            elif current_state == State.OVERTAKING_TURUNCU_PASSING_IN_LEFT_LANE:
                apply_motor_speeds(mc, SOLLAMA_SOL_SERITTE_ILERLEME_HIZI, 0) # Düz git
                if time.time() - state_timer_start > SOLLAMA_SOL_SERITTE_ILERLEME_SURESI:
                    print("Engel geçildi (tahmini). Sağ şeride dönülüyor.")
                    current_state = State.OVERTAKING_TURUNCU_CHANGING_LANE_TO_RIGHT
                    state_timer_start = time.time()

            elif current_state == State.OVERTAKING_TURUNCU_CHANGING_LANE_TO_RIGHT:
                apply_motor_speeds(mc, SOLLAMA_SERIT_DEGISTIRME_HIZI_SAGA, SOLLAMA_SERIT_DEGISTIRME_DIREKSIYON_SAG)
                if time.time() - state_timer_start > SOLLAMA_SERIT_DEGISTIRME_SURESI_SAGA:
                    print("Sağ şeride dönüldü. Sollama tamamlandı.")
                    current_state = State.LANE_FOLLOWING
                    last_overtake_finish_time = time.time() # Sollama bitiş zamanını kaydet
            
            elif current_state == State.GROUND_CROSSING_APPROACHING: # (Önceki gibi)
                mc.stop();print("ZG: Duruldu, bekleniyor...")
                current_state=State.GROUND_CROSSING_WAITING
                state_timer_start=time.time();last_ground_crossing_stop_time=time.time()
            elif current_state == State.GROUND_CROSSING_WAITING: # (Önceki gibi)
                mc.stop()
                if time.time()-state_timer_start >= GROUND_CROSSING_WAIT_TIME_S:
                    print(f"ZG: {GROUND_CROSSING_WAIT_TIME_S}s bekleme bitti.");current_state=State.LANE_FOLLOWING
            elif current_state == State.ERROR: # (Önceki gibi)
                print("HATA durumunda.");mc.stop();running=False

            # --- Görselleştirme --- (Önceki gibi)
            # ... (Tüm cv2.putText ve teşhis görüntüsü ekleme kodları buraya)
            cv2.putText(display_frame_processed,f"D:{current_state}",(5,20),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0),2)
            cv2.putText(display_frame_processed,f"D:{current_state}",(5,20),cv2.FONT_HERSHEY_SIMPLEX,0.4,(50,205,50),1)
            cv2.putText(display_frame_processed,f"O:{lane_offset_m:.2f} C:{lane_curvature:.0f} S:{effective_base_speed:.2f}",(5,40),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0),2)
            cv2.putText(display_frame_processed,f"O:{lane_offset_m:.2f} C:{lane_curvature:.0f} S:{effective_base_speed:.2f}",(5,40),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,0),1)
            if sari_engel_aktif_ts>0 and(time.time()-sari_engel_aktif_ts<SARI_ENGEL_AKTIF_SURESI):cv2.putText(display_frame_processed,"SARI SOLDA!",(CAMERA_WIDTH-150,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
            if current_state==State.GROUND_CROSSING_WAITING:rem_w=max(0,GROUND_CROSSING_WAIT_TIME_S-(time.time()-state_timer_start));cv2.putText(display_frame_processed,f"ZG Bekleme:{rem_w:.1f}s",(CAMERA_WIDTH//2-80,CAMERA_HEIGHT-15),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),1)
            try:
                if diag_img_lanes_display is not None and diag_img_lanes_display.size>0:
                    h_d,w_d=diag_img_lanes_display.shape[:2];display_frame_processed[5:5+h_d,CAMERA_WIDTH-w_d-5:CAMERA_WIDTH-5]=diag_img_lanes_display
            except:pass
            cv2.imshow("Otonom Araç", display_frame_processed)
            key=cv2.waitKey(1)&0xFF
            if key==ord('q'):running=False
            
            # Manevra zaman aşımı (önceki gibi)
            MAX_MANEUVER_TIME_S = 20
            current_maneuver_time_ref = state_timer_start
            if current_state.startswith("OVERTAKING"): current_maneuver_time_ref = overtake_maneuver_start_time
            if current_state not in [State.LANE_FOLLOWING,State.INITIALIZING,State.ERROR] and \
               (time.time()-current_maneuver_time_ref)>MAX_MANEUVER_TIME_S:
                print(f"UYARI: Durum {current_state} çok uzun sürdü. LANE_FOLLOWING'e dönülüyor.")
                current_state=State.LANE_FOLLOWING;mc.stop()

    # ... (except ve finally blokları önceki gibi) ...
    except KeyboardInterrupt: print("Ctrl+C ile durduruldu.")
    except Exception as e: print(f"Ana HATA: {e}"); import traceback; traceback.print_exc();
    finally:
        print("Program sonu.");
        if mc: mc.stop(); mc.cleanup()
        if ld: ld.cleanup()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()