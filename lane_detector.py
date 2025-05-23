# lane_detector.py

import cv2
import numpy as np
from picamera2 import Picamera2, Preview
from libcamera import controls # Aydınlanma gibi kontroller için
import time

class LaneDetector:
    def __init__(self, image_width=640, image_height=480, camera_fps=30):
        """
        Şerit tespit algoritması için gerekli parametreleri ve kamera ayarlarını başlatır.
        """
        print("Şerit Tespit Modülü Başlatılıyor...")
        self.image_width = image_width
        self.image_height = image_height
        self.camera_fps = camera_fps

        # Picamera2 kurulumu
        self.picam2 = Picamera2()
        # Ana akış için yapılandırma (görüntü işleme için)
        config = self.picam2.create_video_configuration(
            main={"size": (self.image_width, self.image_height), "format": "RGB888"},
            lores={"size": (320, 240), "format": "YUV420"}, # İsteğe bağlı düşük çözünürlüklü önizleme akışı
            controls={"FrameRate": self.camera_fps}
        )
        self.picam2.configure(config)
        # Otomatik odaklama ve pozlama ayarlarını yapabiliriz
        # self.picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous, "AwbEnable": True, "AeEnable": True})
        # Parlama için pozlamayı manuel düşürmek gerekebilir:
        # self.picam2.set_controls({"ExposureTime": 10000, "AnalogueGain": 1.0}) # Değerler deneme yanılma ile bulunur

        self.picam2.start()
        time.sleep(1) # Kameranın ısınması için kısa bir bekleme
        print(f"Kamera {self.image_width}x{self.image_height} @ {self.camera_fps}FPS ile başlatıldı.")

        # Perspektif dönüşümü için kaynak ve hedef noktaları
        # BU NOKTALAR KAMERA AÇINIZA VE YÜKSEKLİĞİNİZE GÖRE AYARLANMALIDIR!
        # Örnek değerler: Görüntünün alt orta kısmındaki bir yamuğu,
        # düz bir dikdörtgene dönüştüreceğiz.
        # (sol-alt, sağ-alt, sağ-üst, sol-üst)
        # Bu noktaları bulmak için bir test görüntüsü üzerinde manuel işaretleme yapmak en iyisidir.
        # Pist genişliği 100cm, şeritler 40cm. Araç ortada olacak.
        # Kamera yerden 23cm yüksekte.
        roi_bottom_offset = 50 # Görüntünün en altından ne kadar yukarıda ROI başlasın
        roi_top_offset = int(self.image_height * 0.60) # ROI'nin üst sınırı (görüntünün % kaçı)
        roi_side_margin = int(self.image_width * 0.1) # Kenarlardan ne kadar içerde

        self.src_points = np.float32([
            (roi_side_margin, self.image_height - roi_bottom_offset),  # Sol Alt
            (self.image_width - roi_side_margin, self.image_height - roi_bottom_offset),  # Sağ Alt
            (self.image_width // 2 - int(self.image_width*0.20), roi_top_offset), # Sağ Üst (ufuk çizgisine yakın daralan kısım)
            (self.image_width // 2 + int(self.image_width*0.20), roi_top_offset)  # Sol Üst (ufuk çizgisine yakın daralan kısım)
        ])
        # Hata: src_points sıralaması (sol-alt, sağ-alt, SOL-ÜST, SAĞ-ÜST) olmalı genelde, ya da cv2.getPerspectiveTransform
        # beklentisine göre (sol-üst, sağ-üst, sol-alt, sağ-alt).
        # Tipik sıralama: (üst-sol, üst-sağ, alt-sağ, alt-sol)
        self.src_points = np.float32([
            (self.image_width // 2 - int(self.image_width*0.25), roi_top_offset), # Sol Üst (ufuk çizgisine yakın)
            (self.image_width // 2 + int(self.image_width*0.25), roi_top_offset), # Sağ Üst (ufuk çizgisine yakın)
            (self.image_width - roi_side_margin, self.image_height - roi_bottom_offset),  # Sağ Alt
            (roi_side_margin, self.image_height - roi_bottom_offset)  # Sol Alt
        ])


        # Hedef noktalar (kuşbakışı görüntüde nasıl görünecek)
        # Genellikle düz bir dikdörtgen
        dst_width = int(self.image_width * 0.6) # Kuşbakışı görüntünün genişliği
        dst_height = self.image_height # Kuşbakışı görüntünün yüksekliği (veya ROI yüksekliği)

        self.dst_points = np.float32([
            [0, 0],                         # Sol Üst
            [dst_width - 1, 0],             # Sağ Üst
            [dst_width - 1, dst_height - 1],# Sağ Alt
            [0, dst_height - 1]             # Sol Alt
        ])

        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.Minv = cv2.getPerspectiveTransform(self.dst_points, self.src_points)

        # Metre başına piksel (KALİBRASYON GEREKTİRİR!)
        # Bu değerler, kuşbakışı görüntüdeki şerit genişliğine ve
        # ROI'nin gerçek dünyadaki uzunluğuna göre ayarlanmalıdır.
        # Örnek: Pistin iki şeridi toplam 80cm, kuşbakışında 300 piksel ise:
        self.xm_per_pix = 0.80 / dst_width # 80 cm (iki şerit) / kuşbakışı piksel genişliği
        # Örnek: İleri doğru baktığımız ROI'nin derinliği 50cm, kuşbakışında 400 piksel ise:
        self.ym_per_pix = 1.0 / dst_height # 1 metre / kuşbakışı piksel yüksekliği (bu değer daha zor belirlenir)
                                            # Genellikle yolda belli bir mesafeyi ölçüp piksel karşılığını bulmak gerekir.

        # Şerit takibi için önceki frame'den kalan bilgiler
        self.left_fit = None
        self.right_fit = None
        self.detected_once = False # İlk tespitten sonra daha hızlı arama yapılabilir


    def capture_frame(self):
        """Kameradan bir frame yakalar."""
        # `capture_array` doğrudan bir NumPy dizisi döndürür.
        # "main" akışını kullanıyoruz.
        frame = self.picam2.capture_array("main")
        # Picamera2 RGB888 formatında BGR sırasında değil, RGB sırasında verir.
        # OpenCV genellikle BGR bekler, ama biz burada RGB ile devam edebiliriz
        # veya cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) ile dönüştürebiliriz.
        # Şimdilik RGB kalsın, renk maskelemede ona göre davranırız.
        return frame

    def _preprocess_image(self, img):
        """Görüntüyü şerit tespiti için ön işler."""
        # 1. ROI (İlgi Alanı) - Perspektif dönüşümü öncesi de uygulanabilir, sonrası da.
        # Şimdilik tüm görüntüye uygulayalım, perspektif zaten bir nevi ROI yapıyor.

        # 2. Gri Tonlama
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 3. Gaussian Blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 4. Renk Maskeleme / Eşikleme (Beyaz şeritler için)
        # Siyah zemin (0-50), beyaz şerit (200-255) varsayımıyla
        # Eşik değerlerini ortam ışığına göre ayarlamak gerekebilir.
        #_, thresholded = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)
        # Adaptive thresholding parlama ve gölgelere daha dayanıklı olabilir:
        thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2) # Blok boyutu ve C sabiti ayarlanabilir

        # İsteğe bağlı: Morfolojik operasyonlar (gürültü temizleme)
        # kernel = np.ones((3,3), np.uint8)
        # thresholded = cv2.erode(thresholded, kernel, iterations=1)
        # thresholded = cv2.dilate(thresholded, kernel, iterations=1)

        return thresholded

    def _perspective_warp(self, img):
        """Görüntüye kuşbakışı perspektif dönüşümü uygular."""
        warped_img = cv2.warpPerspective(img, self.M, (int(self.dst_points[1][0]), int(self.dst_points[2][1])), flags=cv2.INTER_LINEAR)
        return warped_img

    def _find_lane_pixels_sliding_window(self, warped_img):
        """
        Kayar pencere yöntemiyle şerit piksellerini bulur.
        Kesikli şeritler ve virajlar için uygundur.
        """
        histogram = np.sum(warped_img[warped_img.shape[0]//2:, :], axis=0) # Alt yarı histogramı
        midpoint = np.int32(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        nwindows = 9  # Pencere sayısı
        margin = 100  # Pencere genişliği (+/- piksel)
        minpix = 50   # Bir pencerede şerit pikseli olarak kabul edilecek minimum piksel sayısı

        window_height = np.int32(warped_img.shape[0]//nwindows)
        nonzero = warped_img.nonzero() # Görüntüdeki beyaz piksellerin koordinatları
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []

        # Görselleştirme için (isteğe bağlı)
        out_img = np.dstack((warped_img, warped_img, warped_img)) * 255 # 3 kanallı yap

        for window in range(nwindows):
            win_y_low = warped_img.shape[0] - (window+1)*window_height
            win_y_high = warped_img.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Pencereleri çiz (görselleştirme)
            cv2.rectangle(out_img,(win_xleft_low,win_y_low), (win_xleft_high,win_y_high),(0,255,0), 2)
            cv2.rectangle(out_img,(win_xright_low,win_y_low), (win_xright_high,win_y_high),(0,255,0), 2)

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > minpix:
                leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))

        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError: # Eğer hiç piksel bulunamazsa
            pass

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, out_img


    def _fit_polynomial(self, leftx, lefty, rightx, righty, warped_shape):
        """Tespit edilen piksellere polinom uydurur."""
        left_fit, right_fit = None, None
        left_fit_cr, right_fit_cr = None, None # Gerçek dünya birimlerinde katsayılar
        ploty = np.linspace(0, warped_shape[0]-1, warped_shape[0])

        try:
            if len(leftx) > 0 and len(lefty) > 0:
                left_fit = np.polyfit(lefty, leftx, 2) # Piksel uzayında
                left_fit_cr = np.polyfit(lefty * self.ym_per_pix, leftx * self.xm_per_pix, 2) # Metrik uzayda
            if len(rightx) > 0 and len(righty) > 0:
                right_fit = np.polyfit(righty, rightx, 2) # Piksel uzayında
                right_fit_cr = np.polyfit(righty * self.ym_per_pix, rightx * self.xm_per_pix, 2) # Metrik uzayda
        except TypeError as e:
            print(f"Polinom uydurma hatası: {e}. Yetersiz piksel olabilir.")
            return None, None, None, None, ploty

        return left_fit, right_fit, left_fit_cr, right_fit_cr, ploty


    def _calculate_curvature_offset(self, left_fit_cr, right_fit_cr, ploty, warped_shape):
        """Şerit eğriliğini ve aracın şerit ortasına olan mesafesini hesaplar."""
        # ploty'nin en alt noktasını (araca en yakın) değerlendirme için alalım
        y_eval = np.max(ploty) * self.ym_per_pix # Metrik

        left_curverad, right_curverad = 0, 0

        if left_fit_cr is not None:
            # R_curve = (1 + (2Ay + B)^2)^(3/2) / |2A|
            A = left_fit_cr[0]
            B = left_fit_cr[1]
            if A != 0 : # Düz çizgi değilse
                 left_curverad = ((1 + (2*A*y_eval + B)**2)**1.5) / np.absolute(2*A)

        if right_fit_cr is not None:
            A = right_fit_cr[0]
            B = right_fit_cr[1]
            if A != 0:
                right_curverad = ((1 + (2*A*y_eval + B)**2)**1.5) / np.absolute(2*A)

        # Ortalama eğrilik yarıçapı (veya sadece sol/sağ kullanılabilir)
        # Eğer biri 0 ise (düz çizgi veya tespit yok), diğerini kullan
        if left_curverad != 0 and right_curverad != 0:
            curvature = (left_curverad + right_curverad) / 2
        elif left_curverad != 0:
            curvature = left_curverad
        elif right_curverad != 0:
            curvature = right_curverad
        else:
            curvature = float('inf') # Düz çizgi için sonsuz eğrilik yarıçapı

        # Aracın şerit merkezine olan uzaklığı (offset)
        offset_pixels = 0
        lane_center_px = 0
        vehicle_center_px = warped_shape[1] / 2 # Görüntünün ortası

        if left_fit_cr is not None and right_fit_cr is not None:
            # y_eval'deki (en alt) x pozisyonları
            left_x_pos = left_fit_cr[0]*(y_eval**2) + left_fit_cr[1]*y_eval + left_fit_cr[2]
            right_x_pos = right_fit_cr[0]*(y_eval**2) + right_fit_cr[1]*y_eval + right_fit_cr[2]
            lane_center_metric = (left_x_pos + right_x_pos) / 2
            vehicle_center_metric = (warped_shape[1] * self.xm_per_pix) / 2
            offset_m = vehicle_center_metric - lane_center_metric
        elif left_fit_cr is not None: # Sadece sol şerit varsa, şerit genişliğini (40cm) kullanarak sağ şeridi tahmin et
            assumed_lane_width_m = 0.40 # Tek şerit genişliği
            left_x_pos = left_fit_cr[0]*(y_eval**2) + left_fit_cr[1]*y_eval + left_fit_cr[2]
            lane_center_metric = left_x_pos + assumed_lane_width_m / 2
            vehicle_center_metric = (warped_shape[1] * self.xm_per_pix) / 2
            offset_m = vehicle_center_metric - lane_center_metric
        elif right_fit_cr is not None: # Sadece sağ şerit varsa
            assumed_lane_width_m = 0.40
            right_x_pos = right_fit_cr[0]*(y_eval**2) + right_fit_cr[1]*y_eval + right_fit_cr[2]
            lane_center_metric = right_x_pos - assumed_lane_width_m / 2
            vehicle_center_metric = (warped_shape[1] * self.xm_per_pix) / 2
            offset_m = vehicle_center_metric - lane_center_metric
        else: # Hiç şerit yoksa
            offset_m = 0 # Veya bir hata durumu belirt

        # Offset: pozitif ise araç şeridin sağında, negatif ise solunda
        return curvature, offset_m


    def _draw_lanes(self, original_img, warped_img, left_fit, right_fit, ploty):
        """Tespit edilen şeritleri orijinal görüntüye çizer."""
        if left_fit is None and right_fit is None:
            return original_img # Çizilecek bir şey yoksa orijinali döndür

        warp_zero = np.zeros_like(warped_img).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        try:
            if left_fit is not None:
                left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
                pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
                cv2.polylines(color_warp, np.int_([pts_left]), isClosed=False, color=(255,0,0), thickness=10) # Mavi sol

            if right_fit is not None:
                right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
                pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
                cv2.polylines(color_warp, np.int_([pts_right]), isClosed=False, color=(0,0,255), thickness=10) # Kırmızı sağ

            # İki şerit arasını yeşile boya (isteğe bağlı)
            if left_fit is not None and right_fit is not None:
                left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2] # Yeniden hesapla
                right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2] # Yeniden hesapla
                pts = np.hstack((pts_left, pts_right[:, ::-1, :])) # İki şeridi birleştir
                cv2.fillPoly(color_warp, np.int_([pts]), (0,255,0)) # Yeşil alan

        except Exception as e:
            print(f"Şerit çizerken hata: {e}")
            return original_img


        # Kuşbakışı çizimi orijinal görüntüye geri dönüştür
        new_warp = cv2.warpPerspective(color_warp, self.Minv, (original_img.shape[1], original_img.shape[0]))
        result_img = cv2.addWeighted(original_img, 1, new_warp, 0.3, 0)
        return result_img


    def detect_lanes(self, frame):
        """
        Ana şerit tespit fonksiyonu.
        Bir frame alır, işler ve şerit bilgileriyle birlikte işlenmiş frame'i döndürür.
        Döndürülenler: (processed_image, lane_offset_m, lane_curvature, diagnostic_warped_img)
        """
        # 1. Ön işleme
        preprocessed_img = self._preprocess_image(frame)

        # 2. Perspektif Dönüşümü
        warped_img = self._perspective_warp(preprocessed_img)

        # 3. Şerit Piksellerini Bulma
        # Eğer daha önce şerit bulunduysa, önceki fit'in etrafında arama yapılabilir (daha hızlı)
        # Şimdilik her zaman sliding window kullanalım
        leftx, lefty, rightx, righty, diagnostic_sliding_window_img = self._find_lane_pixels_sliding_window(warped_img)

        # 4. Polinom Uydurma
        left_fit, right_fit, left_fit_cr, right_fit_cr, ploty = self._fit_polynomial(leftx, lefty, rightx, righty, warped_img.shape)
        self.left_fit = left_fit # Son başarılı fit'i sakla
        self.right_fit = right_fit

        # 5. Sapma ve Eğrilik Hesaplanması
        curvature, offset_m = 0, 0
        if left_fit_cr is not None or right_fit_cr is not None: # En az bir şerit bulunduysa
             curvature, offset_m = self._calculate_curvature_offset(left_fit_cr, right_fit_cr, ploty, warped_img.shape)

        # 6. Görselleştirme
        # Orijinal görüntü RGB, OpenCV BGR bekler çizim için, ama addWeighted RGB ile çalışır
        # frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # Eğer OpenCV çizim fonksiyonları direkt kullanılacaksa
        final_image = self._draw_lanes(frame.copy(), warped_img, left_fit, right_fit, ploty)

        # Ekrana bilgi yazdırma (isteğe bağlı)
        cv2.putText(final_image, f"Offset: {offset_m:.2f}m", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(final_image, f"Curvature: {curvature:.0f}m", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        # Teşhis için kuşbakışı görüntüyü de döndürebiliriz (kayar pencereli olanı)
        # diagnostic_warped_img'i yeniden boyutlandırıp final_image'a ekleyebiliriz
        # Veya ayrı döndürebiliriz:
        h, w = diagnostic_sliding_window_img.shape[:2]
        small_diag = cv2.resize(diagnostic_sliding_window_img, (w//3, h//3))


        return final_image, offset_m, curvature, small_diag # warped_img yerine diagnostic_sliding_window_img döndürülüyor


    def cleanup(self):
        """Kamera ve diğer kaynakları serbest bırakır."""
        print("Şerit Tespit Modülü Kapatılıyor...")
        self.picam2.stop()
        print("Kamera durduruldu.")


# Bu dosya doğrudan çalıştırıldığında test kodu çalışır
if __name__ == "__main__":
    detector = None
    try:
        # Düşük çözünürlükte test için: image_width=320, image_height=240
        # Daha yüksek: image_width=640, image_height=480
        detector = LaneDetector(image_width=320, image_height=240, camera_fps=30)
        print("Şerit Tespiti Testi Başlatılıyor. Çıkmak için Ctrl+C.")
        print("Kameranın perspektif ayarları için 'src_points' ve 'dst_points' değerlerini ayarlamanız gerekebilir.")

        # Perspektif dönüşümü kaynak noktalarını görselleştirmek için:
        # test_frame = detector.capture_frame()
        # for point in detector.src_points:
        #     cv2.circle(test_frame, (int(point[0]), int(point[1])), 5, (0,0,255), -1) # Kırmızı noktalar
        # cv2.imshow("Kaynak Noktalar (Ayarlayın!)", test_frame)
        # cv2.waitKey(0) # Bir tuşa basana kadar beklet

        while True:
            frame = detector.capture_frame()
            if frame is None:
                print("Kameradan frame alınamadı.")
                time.sleep(0.1)
                continue

            processed_frame, offset, curvature, diag_img = detector.detect_lanes(frame.copy()) # frame'in kopyasını gönder

            # Teşhis görüntüsünü ana görüntüye ekle (isteğe bağlı)
            ph, pw = processed_frame.shape[:2]
            dh, dw = diag_img.shape[:2]
            combined_display = np.zeros((max(ph, dh), pw + dw, 3), dtype=np.uint8)
            combined_display[0:ph, 0:pw] = processed_frame
            # diag_img 3 kanallı değilse: diag_img_color = cv2.cvtColor(diag_img, cv2.COLOR_GRAY2BGR)
            # Zaten 3 kanallı dönderiyoruz
            combined_display[0:dh, pw:pw+dw] = diag_img


            cv2.imshow("Şerit Tespiti", combined_display)
            # cv2.imshow("İşlenmiş Kuşbakışı", diag_img) # Ayrı pencerede de gösterilebilir

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'): # Ayar noktalarını kaydetmek için bir kısayol (örnek)
                print("Mevcut src_points:", detector.src_points)
                # np.save("src_points.npy", detector.src_points) gibi kaydedebilirsiniz.

    except KeyboardInterrupt:
        print("Program sonlandırılıyor.")
    except Exception as e:
        print(f"Ana test döngüsünde hata: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if detector:
            detector.cleanup()
        cv2.destroyAllWindows()
        print("Pencereler kapatıldı.")