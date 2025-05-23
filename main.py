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
    OBSTACLE_APPROACHING_TURUNCU = "OBSTACLE_APPROACHING_TURUNCU" # Engeli fark etme ve durma
    OBSTACLE_AVOID_PREPARE_TURUNCU = "OBSTACLE_AVOID_PREPARE_TURUNCU" # Geri çekilme
    OVERTAKING_TURUNCU_CHANGING_LANE_TO_LEFT = "OVERTAKING_TURUNCU_CHANGING_LANE_TO_LEFT"
    OVERTAKING_TURUNCU_PASSING_IN_LEFT_LANE = "OVERTAKING_TURUNCU_PASSING_IN_LEFT_LANE"
    OVERTAKING_TURUNCU_CHANGING_LANE_TO_RIGHT = "OVERTAKING_TURUNCU_CHANGING_LANE_TO_RIGHT"
    GROUND_CROSSING_APPROACHING = "GROUND_CROSSING_APPROACHING"
    GROUND_CROSSING_WAITING = "GROUND_CROSSING_WAITING"
    ERROR = "ERROR"
    MISSION_COMPLETE = "MISSION_COMPLETE"

# --- Genel Parametreler ve Ayarlar ---
# Kamera (Donma sorunları için düşük tutuldu, ihtiyaca göre artırılabilir)
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_FPS = 15

# Temel Sürüş
BASE_SPEED = 0.33 # Biraz daha yavaş genel hız
MAX_SPEED_CAP = 0.40 # Maksimum hız sınırı
MIN_SPEED_IN_CURVE = 0.18 # Virajlarda daha yavaş
CURVATURE_THRESHOLD_FOR_SLOWDOWN = 170 # Bu eğrilik yarıçapının altı yavaşlama gerektirir

# Direksiyon
STEERING_GAIN_P_STRAIGHT = 0.95 # Düz yolda kazanç
STEERING_GAIN_P_CURVE = 1.40    # Virajlarda daha yüksek kazanç
CURVATURE_THRESHOLD_FOR_GAIN_ADJUST = 210 # Bu eğriliğin altı viraj kazancı

# Sollama Parametreleri (Hızlı ve Keskin Ayarlar - DİKKATLİCE TEST EDİLMELİ!)
TURUNCU_ENGEL_GERI_CEKILME_SURESI = 0.35      # saniye (kısa geri çekilme)
TURUNCU_ENGEL_GERI_CEKILME_HIZI = 0.30      # hızlı geri
SOLLAMA_SERIT_DEGISTIRME_SURESI_SOLA = 0.65   # saniye (hızlı sol şerit geçişi)
SOLLAMA_SERIT_DEGISTIRME_HIZI_SOLA = 0.48   # Sol şeride geçerken yüksek hız
SOLLAMA_SERIT_DEGISTIRME_DIREKSIYON_SOL = -0.90 # Keskin sola dönüş (max -1.0)
SOLLAMA_SOL_SERITTE_ILERLEME_SURESI = 0.8   # saniye (engeli hızla geçmek için kısa süre)
SOLLAMA_SOL_SERITTE_ILERLEME_HIZI = 0.55    # Sol şeritte ilerlerken çok yüksek hız
SOLLAMA_SERIT_DEGISTIRME_SURESI_SAGA = 0.75   # saniye (sağa dönüş)
SOLLAMA_SERIT_DEGISTIRME_HIZI_SAGA = 0.42   # Sağ şeride dönerken hız
SOLLAMA_SERIT_DEGISTIRME_DIREKSIYON_SAG = 0.85  # Keskin sağa dönüş (max 1.0)
OVERTAKE_COOLDOWN_S = 8.0 # saniye (sollama sonrası tekrar sollama yapmama süresi)

# Diğer
TARGET_LANE_OFFSET_M = 0.0
TARGET_LANE_OFFSET_LEFT_LANE_M = -0.38
SARI_ENGEL_YAVASLAMA_HIZI = 0.18
SARI_ENGEL_KACINMA_DIREKSIYONU_ITME = 0.35
SARI_ENGEL_AKTIF_SURESI = 2.5
GROUND_CROSSING_WAIT_TIME_S = 5.0
GROUND_CROSSING_COOLDOWN_S = 12.0
GROUND_CROSSING_DETECTION_METHOD = "contour" # "hough" veya "contour"

# --- Yardımcı Fonksiyonlar ---
def get_dynamic_speed(current_curvature, base_speed_ref=BASE_SPEED):
    if 0 < current_curvature < CURVATURE_THRESHOLD_FOR_SLOWDOWN:
        if current_curvature < 70: return MIN_SPEED_IN_CURVE # Çok keskin viraj
        else: return (MIN_SPEED_IN_CURVE + base_speed_ref) / 1.95 # Orta viraj
    return min(base_speed_ref, MAX_SPEED_CAP)

def get_dynamic_steering_gain(current_curvature):
    if 0 < current_curvature < CURVATURE_THRESHOLD_FOR_GAIN_ADJUST:
        return STEERING_GAIN_P_CURVE
    return STEERING_GAIN_P_STRAIGHT

def calculate_steering(offset_m, current_curvature):
    dynamic_gain = get_dynamic_steering_gain(current_curvature)
    steering_val = -offset_m * dynamic_gain
    return max(-1.0, min(1.0, steering_val)) # -1 ile 1 arasında sınırla

def apply_motor_speeds(mc, base_speed_val, steering_val):
    # Diferansiyel sürüş: direksiyon bir tekeri yavaşlatırken diğerini hızlandırır (veya tam tersi)
    # steering_val < 0 ise sola dön: sol yavaşlar, sağ hızlanır
    # steering_val > 0 ise sağa dön: sağ yavaşlar, sol hızlanır
    # steering_sensitivity faktörü, direksiyonun hıza ne kadar etki edeceğini belirler.
    steering_sensitivity = 0.90 # Bu değerle oynayarak dönüş keskinliğini ayarlayın
    
    left_speed = base_speed_val - (steering_val * base_speed_val * steering_sensitivity)
    right_speed = base_speed_val + (steering_val * base_speed_val * steering_sensitivity)

    # Hızları [-MAX_SPEED_CAP, MAX_SPEED_CAP] aralığına kliple
    left_speed = max(-MAX_SPEED_CAP, min(MAX_SPEED_CAP, left_speed))
    right_speed = max(-MAX_SPEED_CAP, min(MAX_SPEED_CAP, right_speed))
    
    mc.set_speeds(left_speed, right_speed)

# --- Ana Program ---
def main():
    current_state = State.INITIALIZING
    mc, ld, od, gcd = None, None, None, None
    state_timer_start = 0 # Genel durum zamanlayıcısı
    overtake_maneuver_start_time = 0 # Tüm sollama manevrasının başlangıç zamanı
    last_ground_crossing_stop_time = 0
    sari_engel_aktif_ts = 0
    last_overtake_finish_time = 0 # En son sollama bitiş zamanı

    running = True
    # Teşhis için (main.py'de oluşturulup güncellenecek)
    diag_img_lanes_display = np.zeros((CAMERA_HEIGHT//3, CAMERA_WIDTH//3, 3), dtype=np.uint8)

    try:
        print("Ana program başlatılıyor...")
        mc = MotorController()
        ld = LaneDetector(image_width=CAMERA_WIDTH, image_height=CAMERA_HEIGHT, camera_fps=CAMERA_FPS)
        od = ObstacleDetector() # approach_threshold_ratio_turuncu değeri yüksek olmalı
        gcd = GroundCrossingDetector(image_width=CAMERA_WIDTH, image_height=CAMERA_HEIGHT)
        
        current_state = State.LANE_FOLLOWING
        print(f"Modüller yüklendi. Başlangıç durumu: {current_state}")
        cv2.namedWindow("Otonom Araç Kontrol", cv2.WINDOW_AUTOSIZE)

        while running:
            frame_rgb_original = ld.capture_frame()
            if frame_rgb_original is None:
                time.sleep(0.01)
                continue

            # Her döngüde, üzerine çizim yapılacak ana display_frame'i orijinalden kopyala
            display_frame_processed = frame_rgb_original.copy()

            # 1. Şerit Tespiti
            lane_offset_m, lane_curvature = 0.0, float('inf') # Varsayılan
            try:
                # ld.detect_lanes, (çizilmiş_kopya, offset, curvature, teşhis_img) döndürür
                display_frame_processed, lane_offset_m, lane_curvature, diag_img_lanes_current = \
                    ld.detect_lanes(frame_rgb_original) # Ham frame'i ver
                if diag_img_lanes_current is not None and diag_img_lanes_current.size > 0:
                    # Teşhis görüntüsünü yeniden boyutlandırıp sakla
                    diag_img_lanes_display = cv2.resize(diag_img_lanes_current, (CAMERA_WIDTH//3, CAMERA_HEIGHT//3))
            except Exception as e:
                print(f"HATA (Şerit Tespiti): {e}")
                # Hata durumunda display_frame_processed ham kalır

            # 2. Engel Tespiti
            turuncu_engeller, sari_engeller = [], []
            try:
                # od.find_obstacles, (turuncu_liste, sari_liste, çizilmiş_kopya) döndürür
                # Girdi olarak bir önceki adımdan gelen (şeritler çizilmiş) frame'i ver
                turuncu_engeller, sari_engeller, display_frame_processed = \
                    od.find_obstacles(display_frame_processed, 
                                      lane_info={"offset_m": lane_offset_m, "image_width_px": CAMERA_WIDTH})
            except Exception as e:
                print(f"HATA (Engel Tespiti): {e}")

            effective_base_speed = get_dynamic_speed(lane_curvature, BASE_SPEED)

            # 3. Zemin Geçidi Tespiti
            is_gc_detected_this_frame = False
            # Sadece belirli durumlarda zemin geçidi kontrolü yap
            can_check_ground_crossing = current_state not in [
                State.GROUND_CROSSING_WAITING, State.OBSTACLE_AVOID_PREPARE_TURUNCU,
                State.OVERTAKING_TURUNCU_CHANGING_LANE_TO_LEFT, State.OVERTAKING_TURUNCU_PASSING_IN_LEFT_LANE,
                State.OVERTAKING_TURUNCU_CHANGING_LANE_TO_RIGHT
            ]
            if can_check_ground_crossing:
                # gcd.detect_crossing, boolean döndürür ve verilen frame üzerine çizer
                is_gc_detected_this_frame = gcd.detect_crossing(
                    frame_rgb_original, # Tespit için ham frame
                    display_frame_processed, # Çizim için en son işlenmiş frame
                    method=GROUND_CROSSING_DETECTION_METHOD
                ) # Bu display_frame_processed'i yerinde günceller
            
            time_since_last_gc_stop = time.time() - last_ground_crossing_stop_time
            if is_gc_detected_this_frame and time_since_last_gc_stop > GROUND_CROSSING_COOLDOWN_S:
                if current_state != State.GROUND_CROSSING_APPROACHING: # Tekrar tekrar girmesini engelle
                    print(f"ZEMİN GEÇİDİ ALGILANDI (Cooldown sonrası: {time_since_last_gc_stop:.1f}s). Duruluyor.")
                    current_state = State.GROUND_CROSSING_APPROACHING
                    mc.stop() # Anında dur
            
            # --- Ana Durum Makinesi ---
            # print(f"Durum: {current_state}, Offset: {lane_offset_m:.2f}m, Curv: {lane_curvature:.0f}, Hız: {effective_base_speed:.2f}")

            if current_state == State.LANE_FOLLOWING:
                time_since_last_overtake = time.time() - last_overtake_finish_time
                can_attempt_overtake_now = time_since_last_overtake > OVERTAKE_COOLDOWN_S

                approaching_turuncu_obstacle_to_overtake = None
                if can_attempt_overtake_now:
                    for obs_t in turuncu_engeller:
                        # obstacle_detector'daki approach_threshold_ratio_turuncu değeri YÜKSEK olmalı
                        if obs_t.get('is_in_my_lane') and obs_t.get('is_approaching'):
                            approaching_turuncu_obstacle_to_overtake = obs_t
                            break 
                
                if approaching_turuncu_obstacle_to_overtake:
                    print(f"TURUNCU ENGEL YAKINDA (Alan Oranı: {approaching_turuncu_obstacle_to_overtake['area_ratio']:.3f}). Sollama için hazırlanılıyor.")
                    current_state = State.OBSTACLE_APPROACHING_TURUNCU
                    mc.stop() # Engeli fark edince dur
                else:
                    # Normal şerit takibi ve sarı engel kontrolü
                    steering_val = calculate_steering(lane_offset_m, lane_curvature)
                    current_speed_final = effective_base_speed

                    is_sari_engel_on_left_active_now = False
                    for obs_s in sari_engeller:
                        if obs_s.get('is_on_left_lane') and obs_s.get('is_approaching'):
                            is_sari_engel_on_left_active_now = True
                            sari_engel_aktif_ts = time.time()
                            break
                    
                    is_sari_engel_still_in_cooldown = (sari_engel_aktif_ts > 0 and (time.time() - sari_engel_aktif_ts < SARI_ENGEL_AKTIF_SURESI))
                    
                    if is_sari_engel_on_left_active_now or is_sari_engel_still_in_cooldown:
                        if is_sari_engel_on_left_active_now: print("SARI ENGEL SOLDA AKTİF!")
                        else: print("Sarı engel için cooldown aktif, sağda kal.")
                        
                        current_speed_final = min(effective_base_speed, SARI_ENGEL_YAVASLAMA_HIZI)
                        if lane_offset_m < -0.05 or steering_val < -0.05: # Sola kaymışsak veya hafif sola dönüyorsak
                            steering_val += SARI_ENGEL_KACINMA_DIREKSIYONU_ITME
                            steering_val = min(steering_val, 0.4) # Çok keskin sağa gitmesin
                    
                    apply_motor_speeds(mc, current_speed_final, steering_val)

            elif current_state == State.OBSTACLE_APPROACHING_TURUNCU:
                # Bu durum sadece bir geçiş durumu, hemen bir sonraki adıma geçer.
                print("Turuncu engel için duruldu. Geri çekilme başlıyor.")
                current_state = State.OBSTACLE_AVOID_PREPARE_TURUNCU
                state_timer_start = time.time() # Geri çekilme süresi için
                mc.backward(TURUNCU_ENGEL_GERI_CEKILME_HIZI)

            elif current_state == State.OBSTACLE_AVOID_PREPARE_TURUNCU:
                if time.time() - state_timer_start > TURUNCU_ENGEL_GERI_CEKILME_SURESI:
                    mc.stop()
                    print("Geri çekilme tamamlandı. Sol şeride geçişe başlanıyor.")
                    current_state = State.OVERTAKING_TURUNCU_CHANGING_LANE_TO_LEFT
                    state_timer_start = time.time() # Sol şeride geçiş süresi için
                    overtake_maneuver_start_time = time.time() # Tüm sollama manevrasının genel başlangıcı
                # Geri gitmeye devam (mc.backward zaten çağrıldı)

            elif current_state == State.OVERTAKING_TURUNCU_CHANGING_LANE_TO_LEFT:
                apply_motor_speeds(mc, SOLLAMA_SERIT_DEGISTIRME_HIZI_SOLA, SOLLAMA_SERIT_DEGISTIRME_DIREKSIYON_SOL)
                if time.time() - state_timer_start > SOLLAMA_SERIT_DEGISTIRME_SURESI_SOLA:
                    print("Sol şeride geçiş tamamlandı (süre bazlı). Sol şeritte engeli geçmek için ilerleniyor.")
                    current_state = State.OVERTAKING_TURUNCU_PASSING_IN_LEFT_LANE
                    state_timer_start = time.time() # Sol şeritte ilerleme süresi için
            
            elif current_state == State.OVERTAKING_TURUNCU_PASSING_IN_LEFT_LANE:
                apply_motor_speeds(mc, SOLLAMA_SOL_SERITTE_ILERLEME_HIZI, 0) # Düz ve hızlı git
                if time.time() - state_timer_start > SOLLAMA_SOL_SERITTE_ILERLEME_SURESI:
                    print("Engeli geçme süresi doldu (tahmini). Sağ şeride dönülüyor.")
                    current_state = State.OVERTAKING_TURUNCU_CHANGING_LANE_TO_RIGHT
                    state_timer_start = time.time() # Sağ şeride dönüş süresi için

            elif current_state == State.OVERTAKING_TURUNCU_CHANGING_LANE_TO_RIGHT:
                apply_motor_speeds(mc, SOLLAMA_SERIT_DEGISTIRME_HIZI_SAGA, SOLLAMA_SERIT_DEGISTIRME_DIREKSIYON_SAG)
                if time.time() - state_timer_start > SOLLAMA_SERIT_DEGISTIRME_SURESI_SAGA:
                    print("Sağ şeride dönüş tamamlandı (süre bazlı). Sollama bitti.")
                    current_state = State.LANE_FOLLOWING
                    last_overtake_finish_time = time.time() # Sollama bitiş zamanını kaydet (cooldown için)
            
            elif current_state == State.GROUND_CROSSING_APPROACHING:
                mc.stop() # Zaten durdurulmuştu, emin olmak için.
                print("Zemin geçidinde duruldu, bekleme durumuna geçiliyor.")
                current_state = State.GROUND_CROSSING_WAITING
                state_timer_start = time.time() # Bekleme süresini burada başlat
                last_ground_crossing_stop_time = time.time() # Son duruş zamanını kaydet (cooldown için)

            elif current_state == State.GROUND_CROSSING_WAITING:
                mc.stop() # Durmaya devam et
                if time.time() - state_timer_start >= GROUND_CROSSING_WAIT_TIME_S:
                    print(f"Zemin geçidinde {GROUND_CROSSING_WAIT_TIME_S}sn bekleme süresi doldu. Şerit takibine devam ediliyor.")
                    current_state = State.LANE_FOLLOWING
            
            elif current_state == State.ERROR:
                print("HATA durumunda. Motorlar durduruldu."); mc.stop(); running = False
            
            # --- Görselleştirme ---
            # Durum, offset vb. bilgileri display_frame_processed üzerine çiz
            cv2.putText(display_frame_processed,f"D: {current_state}",(5,20),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0),2)
            cv2.putText(display_frame_processed,f"D: {current_state}",(5,20),cv2.FONT_HERSHEY_SIMPLEX,0.4,(50,205,50),1)
            cv2.putText(display_frame_processed, f"O:{lane_offset_m:.2f} C:{lane_curvature:.0f} S:{effective_base_speed:.2f}",(5,40),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0),2)
            cv2.putText(display_frame_processed, f"O:{lane_offset_m:.2f} C:{lane_curvature:.0f} S:{effective_base_speed:.2f}",(5,40),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,0),1)

            # Sarı engel uyarısı
            if sari_engel_aktif_ts > 0 and (time.time() - sari_engel_aktif_ts < SARI_ENGEL_AKTIF_SURESI):
                 cv2.putText(display_frame_processed, "SARI SOLDA!",(CAMERA_WIDTH-150,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
                 cv2.putText(display_frame_processed, "SARI SOLDA!",(CAMERA_WIDTH-150,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
            
            # Zemin geçidi bekleme sayacı
            if current_state == State.GROUND_CROSSING_WAITING:
                 remaining_wait = max(0, GROUND_CROSSING_WAIT_TIME_S - (time.time() - state_timer_start))
                 cv2.putText(display_frame_processed, f"ZG Bekleme: {remaining_wait:.1f}s",(CAMERA_WIDTH//2-80,CAMERA_HEIGHT-15),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)
                 cv2.putText(display_frame_processed, f"ZG Bekleme: {remaining_wait:.1f}s",(CAMERA_WIDTH//2-80,CAMERA_HEIGHT-15),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),1)

            # Teşhis görüntüsünü (lane detector) sağ üste ekle
            try:
                if diag_img_lanes_display is not None and diag_img_lanes_display.size > 0:
                    h_d, w_d = diag_img_lanes_display.shape[:2]
                    # Çakışmaması için sol üste alalım
                    display_frame_processed[5 : 5+h_d, 5 : 5+w_d] = diag_img_lanes_display
            except Exception: pass # Boyut uyumsuzluğu veya başka bir sorun olursa çökmesin

            cv2.imshow("Otonom Araç Kontrol", display_frame_processed)
            key = cv2.waitKey(1) & 0xFF # GUI olaylarını işlemesi için çok önemli!
            if key == ord('q'):
                running = False
            
            # Manevraların çok uzun sürmesini engellemek için güvenlik zaman aşımı
            MAX_MANEUVER_TIME_S = 22 # saniye (biraz daha kısa)
            current_maneuver_time_reference = state_timer_start # Çoğu durum için
            if current_state.startswith("OVERTAKING"):
                current_maneuver_time_reference = overtake_maneuver_start_time
            
            if current_state not in [State.LANE_FOLLOWING, State.INITIALIZING, State.ERROR] and \
               (time.time() - current_maneuver_time_reference) > MAX_MANEUVER_TIME_S :
                print(f"UYARI: Durum '{current_state}' çok uzun sürdü ({MAX_MANEUVER_TIME_S}s). Güvenlik için LANE_FOLLOWING'e dönülüyor.")
                current_state = State.LANE_FOLLOWING
                mc.stop() # Her ihtimale karşı motorları durdur
                if current_state.startswith("OVERTAKING"): # Eğer sollama takıldıysa, sollama cooldown'ını başlat
                    last_overtake_finish_time = time.time()


    except KeyboardInterrupt:
        print("Program kullanıcı tarafından sonlandırıldı (Ctrl+C).")
    except Exception as e:
        print(f"Ana döngüde beklenmedik bir genel hata oluştu: {e}")
        import traceback
        traceback.print_exc()
        # Hata durumunda motorları durdurmaya çalış
        if mc: mc.stop() 
        current_state = State.ERROR # Hata durumuna geç (döngü sonlanabilir)
    finally:
        print("Program sonlandırılıyor. Kaynaklar temizleniyor...")
        if mc:
            mc.stop()
            mc.cleanup()
        if ld:
            ld.cleanup()
        # gcd için özel cleanup yok (eğer kamera kullanmıyorsa)
        cv2.destroyAllWindows()
        print("Tüm kaynaklar temizlendi. Çıkıldı.")

if __name__ == "__main__":
    main()