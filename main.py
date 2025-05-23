# main.py

import cv2
import time
import numpy as np

# Kendi modüllerimizi import ediyoruz
from motor_kontrol import MotorController
from lane_detector import LaneDetector
from obstacle_detector import ObstacleDetector
from ground_crossing_detector import GroundCrossingDetector # Yeni modül

# --- Durum Tanımları ---
class State:
    INITIALIZING = "INITIALIZING"
    LANE_FOLLOWING = "LANE_FOLLOWING"
    OBSTACLE_APPROACHING_TURUNCU = "OBSTACLE_APPROACHING_TURUNCU"
    OBSTACLE_AVOID_PREPARE_TURUNCU = "OBSTACLE_AVOID_PREPARE_TURUNCU"
    OVERTAKING_TURUNCU_CHANGING_LANE = "OVERTAKING_TURUNCU_CHANGING_LANE"
    OVERTAKING_TURUNCU_IN_LEFT_LANE = "OVERTAKING_TURUNCU_IN_LEFT_LANE"
    RETURNING_TO_RIGHT_LANE_AFTER_TURUNCU = "RETURNING_TO_RIGHT_LANE_AFTER_TURUNCU"
    # Sarı engel LANE_FOLLOWING içinde ele alınıyor
    GROUND_CROSSING_APPROACHING = "GROUND_CROSSING_APPROACHING" # Zemin geçidine yaklaşma
    GROUND_CROSSING_WAITING = "GROUND_CROSSING_WAITING"     # Zemin geçidinde bekleme
    # TODO: Park durumları eklenecek
    ERROR = "ERROR"
    MISSION_COMPLETE = "MISSION_COMPLETE" # Pist sonu için

# --- Genel Parametreler ve Ayarlar ---
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Şerit takibi ve hız
BASE_SPEED = 0.35
MAX_SPEED_CAP = 0.45 # Genel hız sınırı (0.0 - 1.0)
MIN_SPEED_IN_CURVE = 0.20
CURVATURE_THRESHOLD_FOR_SLOWDOWN = 200 # Eğrilik yarıçapı (m) bunun altına düşerse yavaşla

# Direksiyon
STEERING_GAIN_P_STRAIGHT = 0.9
STEERING_GAIN_P_CURVE = 1.3
CURVATURE_THRESHOLD_FOR_GAIN_ADJUST = 250

# Sollama
TURUNCU_ENGEL_GERI_CEKILME_SURESI = 0.6
TURUNCU_ENGEL_GERI_CEKILME_HIZI = 0.25
SOLLAMA_SOL_SERITTE_KALMA_SURESI = 2.8
OVERTAKE_CLEARANCE_CHECK_DURATION = 0.7
SERIT_DEGISTIRME_HIZI = BASE_SPEED * 0.85
SERIT_DEGISTIRME_DIREKSIYON_SOL = -0.70
SERIT_DEGISTIRME_DIREKSIYON_SAG = 0.70

TARGET_LANE_OFFSET_M = 0.0
TARGET_LANE_OFFSET_LEFT_LANE_M = -0.40 # Yaklaşık -1 şerit genişliği

# Sarı Engel
SARI_ENGEL_YAVASLAMA_HIZI = 0.20
SARI_ENGEL_KACINMA_DIREKSIYONU_ITME = 0.35
SARI_ENGEL_AKTIF_SURESI = 2.5 # Sarı engel uyarısının ne kadar süreyle aktif kalacağı

# Zemin Geçidi
GROUND_CROSSING_WAIT_TIME_S = 5.0
GROUND_CROSSING_COOLDOWN_S = 15.0 # Aynı geçitte tekrar durmamak için (saniye)
GROUND_CROSSING_DETECTION_METHOD = "contour" # "hough" veya "contour"

# --- Yardımcı Fonksiyonlar ---
def get_dynamic_speed(current_curvature, base_speed_ref=BASE_SPEED):
    if 0 < current_curvature < CURVATURE_THRESHOLD_FOR_SLOWDOWN :
        if current_curvature < 100 : return MIN_SPEED_IN_CURVE
        else: return (MIN_SPEED_IN_CURVE + base_speed_ref) / 2
    return min(base_speed_ref, MAX_SPEED_CAP)

def get_dynamic_steering_gain(current_curvature):
    if 0 < current_curvature < CURVATURE_THRESHOLD_FOR_GAIN_ADJUST:
        return STEERING_GAIN_P_CURVE
    return STEERING_GAIN_P_STRAIGHT

def calculate_steering(offset_m, current_curvature):
    dynamic_gain = get_dynamic_steering_gain(current_curvature)
    steering_p = -offset_m * dynamic_gain
    return max(-1.0, min(1.0, steering_p))

def apply_motor_speeds(mc, base_speed_val, steering_val):
    # Bu direksiyon mantığı, direksiyon değeri arttıkça bir tekeri yavaşlatır,
    # diğeri sabit kalır veya hafif hızlanır gibi bir model izler.
    # steering_val < 0 (sola dön): sol motor yavaşlar/geri, sağ motor ileri
    # steering_val > 0 (sağa dön): sağ motor yavaşlar/geri, sol motor ileri
    # Bu daha çok tank dönüşü gibi olur.
    # Daha yaygın bir diferansiyel sürüş:
    l_s = base_speed_val - steering_val * (base_speed_val * 0.75) # direksiyon hassasiyeti
    r_s = base_speed_val + steering_val * (base_speed_val * 0.75)

    # Hızları [-MAX_SPEED_CAP, MAX_SPEED_CAP] aralığına kliple
    l_s = max(-MAX_SPEED_CAP, min(MAX_SPEED_CAP, l_s))
    r_s = max(-MAX_SPEED_CAP, min(MAX_SPEED_CAP, r_s))
    mc.set_speeds(l_s, r_s)

# --- Ana Program ---
def main():
    current_state = State.INITIALIZING
    mc = None
    ld = None
    od = None
    gcd = None # Ground Crossing Detector

    # Durumlar arası geçişlerde kullanılacak zamanlayıcılar veya sayaçlar
    state_timer_start = 0
    overtake_maneuver_start_time = 0 # Sollama manevrasının toplam süresi için
    last_ground_crossing_stop_time = 0 # En son ne zaman geçitte duruldu
    sari_engel_aktif_ts = 0 # Sarı engel uyarısının ne zaman başladığı

    running = True

    try:
        print("Ana program başlatılıyor...")
        mc = MotorController()
        ld = LaneDetector(image_width=CAMERA_WIDTH, image_height=CAMERA_HEIGHT, camera_fps=CAMERA_FPS)
        od = ObstacleDetector()
        gcd = GroundCrossingDetector(image_width=CAMERA_WIDTH, image_height=CAMERA_HEIGHT)
        
        current_state = State.LANE_FOLLOWING
        print("Tüm modüller yüklendi. Ana döngü başlıyor.")

        while running:
            frame_rgb = ld.capture_frame()
            if frame_rgb is None:
                print("Kameradan frame alınamadı, 10ms bekleniyor...")
                time.sleep(0.01)
                continue

            # Üzerine çizim yapılacak ana frame (her döngü başında ham frame'den kopyala)
            display_frame = frame_rgb.copy()

            # --- 1. Şerit Tespiti ---
            try:
                # LaneDetector kendi çizimlerini processed_lane_frame'e yapar
                processed_lane_frame, lane_offset_m, lane_curvature, diag_img_lanes = ld.detect_lanes(display_frame)
                display_frame = processed_lane_frame # Güncel çizilmiş frame'i al
            except Exception as e:
                print(f"Şerit tespiti sırasında hata: {e}")
                lane_offset_m, lane_curvature = 0, float('inf') # Varsayılan değerler
                diag_img_lanes = np.zeros((100,100,3), dtype=np.uint8) # Boş teşhis görüntüsü

            # --- 2. Engel Tespiti ---
            try:
                # ObstacleDetector kendi çizimlerini processed_obstacle_frame'e yapar
                # Şeritler çizilmiş frame'i girdi olarak veriyoruz
                turuncu_engeller, sari_engeller, processed_obstacle_frame = od.find_obstacles(
                    display_frame,
                    lane_info={"offset_m": lane_offset_m, "image_width_px": CAMERA_WIDTH}
                )
                display_frame = processed_obstacle_frame # Güncel çizilmiş frame'i al
            except Exception as e:
                print(f"Engel tespiti sırasında hata: {e}")
                turuncu_engeller, sari_engeller = [], []

            # --- Dinamik Hız Ayarı ---
            effective_base_speed = get_dynamic_speed(lane_curvature, BASE_SPEED)

            # --- 3. Zemin Geçidi Tespiti (Öncelikli Kontrol) ---
            can_check_ground_crossing = current_state not in [
                State.GROUND_CROSSING_WAITING,
                State.OBSTACLE_AVOID_PREPARE_TURUNCU, # Sollama manevraları sırasında kontrol etme
                State.OVERTAKING_TURUNCU_CHANGING_LANE,
                State.OVERTAKING_TURUNCU_IN_LEFT_LANE,
                State.RETURNING_TO_RIGHT_LANE_AFTER_TURUNCU
            ]

            is_ground_crossing_detected_now = False
            if can_check_ground_crossing:
                # Zemin geçidi tespiti için ham frame'i (frame_rgb) kullanmak daha iyi olabilir,
                # çünkü diğer çizimler tespiti etkileyebilir.
                # gcd.detect_crossing, çizimlerini display_frame'e yapacak.
                is_ground_crossing_detected_now = gcd.detect_crossing(
                    frame_rgb, debug_frame=display_frame, method=GROUND_CROSSING_DETECTION_METHOD
                )

            time_since_last_gc_stop = time.time() - last_ground_crossing_stop_time
            
            if is_ground_crossing_detected_now and time_since_last_gc_stop > GROUND_CROSSING_COOLDOWN_S:
                if current_state != State.GROUND_CROSSING_APPROACHING: # Tekrar tekrar girmesini engelle
                    print(f"!!! Zemin geçidi tespit edildi. Cooldown ({time_since_last_gc_stop:.1f}s) dolmuş.")
                    current_state = State.GROUND_CROSSING_APPROACHING
                    mc.stop() # Anında dur (daha sonra yavaşça durma eklenebilir)
                    # state_timer_start ve last_ground_crossing_stop_time bir sonraki durumda ayarlanacak
            
            # --- Ana Durum Makinesi ---
            # print(f"Durum: {current_state}, Offset: {lane_offset_m:.2f}m, Curv: {lane_curvature:.0f}, Hız: {effective_base_speed:.2f}")

            if current_state == State.LANE_FOLLOWING:
                approaching_turuncu_in_my_lane = None
                for obs_t in turuncu_engeller: # obstacle_detector'dan gelen liste
                    if obs_t.get('is_in_my_lane', False) and obs_t.get('is_approaching', False):
                        approaching_turuncu_in_my_lane = obs_t
                        break

                if approaching_turuncu_in_my_lane:
                    print(f"Turuncu engel ({approaching_turuncu_in_my_lane['rect']}) yaklaşıyor. Sollamaya hazırlanılıyor.")
                    current_state = State.OBSTACLE_APPROACHING_TURUNCU
                    mc.stop()
                else:
                    steering_val = calculate_steering(lane_offset_m, lane_curvature)
                    current_speed_final = effective_base_speed

                    # Sarı engel kontrolü
                    is_sari_engel_on_left_active_now = False
                    for obs_s in sari_engeller:
                        if obs_s.get('is_on_left_lane', False) and obs_s.get('is_approaching', False):
                            is_sari_engel_on_left_active_now = True
                            sari_engel_aktif_ts = time.time() # Zamanı güncelle (en son görülme)
                            break
                    
                    # Eğer yeni sarı engel yoksa ama cooldown içindeysek de aktif say
                    is_sari_engel_still_active_cooldown = (time.time() - sari_engel_aktif_ts < SARI_ENGEL_AKTIF_SURESI)
                    
                    if is_sari_engel_on_left_active_now or (sari_engel_aktif_ts > 0 and is_sari_engel_still_active_cooldown) :
                        if is_sari_engel_on_left_active_now:
                             print("Sol şeritte sarı engel aktif! Sağda kalınıyor.")
                        else:
                             print("Sol şeritteki sarı engel için cooldown aktif! Sağda kalınıyor.")

                        current_speed_final = min(effective_base_speed, SARI_ENGEL_YAVASLAMA_HIZI)
                        if lane_offset_m < -0.05 or steering_val < -0.1: # Sola kaymışsak veya sola dönüyorsak
                            steering_val += SARI_ENGEL_KACINMA_DIREKSIYONU_ITME
                            steering_val = min(steering_val, 0.5) # Çok keskin sağa gitmesin
                    
                    apply_motor_speeds(mc, current_speed_final, steering_val)

            elif current_state == State.OBSTACLE_APPROACHING_TURUNCU:
                print("Turuncu engele yaklaşıldı. Geri çekilme başlıyor.")
                current_state = State.OBSTACLE_AVOID_PREPARE_TURUNCU
                state_timer_start = time.time() # Geri çekilme süresi için
                mc.backward(TURUNCU_ENGEL_GERI_CEKILME_HIZI)

            elif current_state == State.OBSTACLE_AVOID_PREPARE_TURUNCU:
                if time.time() - state_timer_start > TURUNCU_ENGEL_GERI_CEKILME_SURESI:
                    mc.stop()
                    print("Geri çekilme tamamlandı. Sol şeride geçişe başlanıyor.")
                    current_state = State.OVERTAKING_TURUNCU_CHANGING_LANE
                    overtake_maneuver_start_time = time.time() # Tüm sollama manevrasının başlangıcı
                # Geri gitmeye devam (MotorController'da zaten komut verildi)

            elif current_state == State.OVERTAKING_TURUNCU_CHANGING_LANE:
                if lane_offset_m > TARGET_LANE_OFFSET_LEFT_LANE_M + 0.10: # Hedef sol offsete ulaşana kadar
                    apply_motor_speeds(mc, SERIT_DEGISTIRME_HIZI, SERIT_DEGISTIRME_DIREKSIYON_SOL)
                else:
                    print("Sol şeride ulaşıldı. Sol şeritte ilerlemeye başlanıyor.")
                    current_state = State.OVERTAKING_TURUNCU_IN_LEFT_LANE
                    state_timer_start = time.time() # Sol şeritte kalma süresi için
                    steering_val = calculate_steering(lane_offset_m - TARGET_LANE_OFFSET_LEFT_LANE_M, lane_curvature)
                    apply_motor_speeds(mc, effective_base_speed, steering_val)

            elif current_state == State.OVERTAKING_TURUNCU_IN_LEFT_LANE:
                total_time_in_left_lane = SOLLAMA_SOL_SERITTE_KALMA_SURESI + OVERTAKE_CLEARANCE_CHECK_DURATION
                if time.time() - state_timer_start > total_time_in_left_lane:
                    print("Engeli geçme ve güvenlik süresi doldu. Sağ şeride dönülüyor.")
                    current_state = State.RETURNING_TO_RIGHT_LANE_AFTER_TURUNCU
                    # overtake_maneuver_start_time burada sıfırlanabilir veya başarılı sollama sayacı artırılabilir.
                else:
                    steering_val = calculate_steering(lane_offset_m - TARGET_LANE_OFFSET_LEFT_LANE_M, lane_curvature)
                    apply_motor_speeds(mc, effective_base_speed, steering_val)

            elif current_state == State.RETURNING_TO_RIGHT_LANE_AFTER_TURUNCU:
                if lane_offset_m < TARGET_LANE_OFFSET_M - 0.10: # Hedef sağ offsete ulaşana kadar
                    apply_motor_speeds(mc, SERIT_DEGISTIRME_HIZI, SERIT_DEGISTIRME_DIREKSIYON_SAG)
                else:
                    print("Sağ şeride başarıyla dönüldü.")
                    current_state = State.LANE_FOLLOWING

            elif current_state == State.GROUND_CROSSING_APPROACHING:
                mc.stop() # Durduğundan emin ol
                print("Zemin geçidinde duruldu, bekleniyor...")
                current_state = State.GROUND_CROSSING_WAITING
                state_timer_start = time.time() # Bekleme süresini burada başlat
                last_ground_crossing_stop_time = time.time() # Son duruş zamanını kaydet (cooldown için)

            elif current_state == State.GROUND_CROSSING_WAITING:
                mc.stop() # Durmaya devam et
                if time.time() - state_timer_start >= GROUND_CROSSING_WAIT_TIME_S:
                    print(f"Zemin geçidinde {GROUND_CROSSING_WAIT_TIME_S}sn bekleme süresi doldu. Devam ediliyor.")
                    current_state = State.LANE_FOLLOWING
                # else: Kalan süreyi yazdırmak için
                    # remaining_wait = max(0, GROUND_CROSSING_WAIT_TIME_S - (time.time() - state_timer_start))
                    # print(f"ZG Bekleniyor: {remaining_wait:.1f}s")

            elif current_state == State.ERROR:
                print("HATA durumunda. Motorlar durduruldu.")
                mc.stop()
                running = False # Döngüden çık

            # --- Görselleştirme ---
            # Durum, offset, hız vb. bilgileri display_frame üzerine çiz
            cv2.putText(display_frame, f"D: {current_state}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2) # Gölge
            cv2.putText(display_frame, f"D: {current_state}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,205,50), 1) # Yeşil

            cv2.putText(display_frame, f"Ofst:{lane_offset_m:.2f} Crv:{lane_curvature:.0f} Spd:{effective_base_speed:.2f}", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
            cv2.putText(display_frame, f"Ofst:{lane_offset_m:.2f} Crv:{lane_curvature:.0f} Spd:{effective_base_speed:.2f}", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1) # Açık mavi

            if sari_engel_aktif_ts > 0 and (time.time() - sari_engel_aktif_ts < SARI_ENGEL_AKTIF_SURESI):
                 cv2.putText(display_frame, "SARI ENGEL SOLDA!", (CAMERA_WIDTH - 220, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
                 cv2.putText(display_frame, "SARI ENGEL SOLDA!", (CAMERA_WIDTH - 220, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1) # Sarı

            if current_state == State.GROUND_CROSSING_WAITING:
                 remaining_wait = max(0, GROUND_CROSSING_WAIT_TIME_S - (time.time() - state_timer_start))
                 cv2.putText(display_frame, f"Gecit Bekleme: {remaining_wait:.1f}s", (CAMERA_WIDTH // 2 - 100, CAMERA_HEIGHT - 20),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
                 cv2.putText(display_frame, f"Gecit Bekleme: {remaining_wait:.1f}s", (CAMERA_WIDTH // 2 - 100, CAMERA_HEIGHT - 20),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 1) # Koyu Mavi

            # LaneDetector'dan gelen teşhis görüntüsünü (eğer varsa) sağ üste ekle
            h_main, w_main = display_frame.shape[:2]
            try:
                if diag_img_lanes is not None and diag_img_lanes.size > 0: # diag_img_lanes dolu mu?
                    h_diag, w_diag = diag_img_lanes.shape[:2]
                    target_diag_w = w_main // 4 # Ana görüntünün 1/4 genişliğinde
                    if w_diag > 0 : # Sıfıra bölme hatası olmasın
                        target_diag_h = int(h_diag * (target_diag_w / w_diag))
                        if target_diag_h > 0 and target_diag_w > 0: # Boyutlar geçerli mi?
                            small_diag_img = cv2.resize(diag_img_lanes, (target_diag_w, target_diag_h))
                            display_frame[10 : 10+target_diag_h, w_main-target_diag_w-10 : w_main-10] = small_diag_img
            except Exception as e_diag_draw:
                # print(f"Teşhis görüntüsü eklenirken hata: {e_diag_draw}") # Opsiyonel log
                pass

            cv2.imshow("Otonom Araç Kontrol", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                running = False
            # Diğer debug tuşları eklenebilir (örn: durumu manuel değiştirmek için)

            # Manevraların çok uzun sürmesini engellemek için bir güvenlik önlemi
            MAX_MANEUVER_TIME_S = 25 # saniye
            current_maneuver_start_time = overtake_maneuver_start_time if current_state.startswith("OVERTAKING") else state_timer_start
            if current_state not in [State.LANE_FOLLOWING, State.INITIALIZING, State.ERROR] and \
               (time.time() - current_maneuver_start_time) > MAX_MANEUVER_TIME_S :
                print(f"UYARI: Durum {current_state} çok uzun sürdü ({MAX_MANEUVER_TIME_S}s). Güvenlik için LANE_FOLLOWING'e dönülüyor.")
                current_state = State.LANE_FOLLOWING
                mc.stop() # Her ihtimale karşı motorları durdur

    except KeyboardInterrupt:
        print("Program kullanıcı tarafından sonlandırıldı (Ctrl+C).")
    except Exception as e:
        print(f"Ana döngüde beklenmedik bir genel hata oluştu: {e}")
        import traceback
        traceback.print_exc()
        current_state = State.ERROR # Hata durumuna geç
    finally:
        print("Program sonlandırılıyor. Kaynaklar temizleniyor...")
        if mc:
            mc.stop()
            mc.cleanup()
        if ld:
            ld.cleanup()
        # gcd'nin cleanup'ı yok, sadece Picamera2 varsa ld.cleanup() yeterli olur.
        cv2.destroyAllWindows()
        print("Tüm kaynaklar temizlendi. Çıkıldı.")

if __name__ == "__main__":
    main()