# motor_kontrol.py

from gpiozero import Motor, Device
from gpiozero.pins.pigpio import PiGPIOFactory # Daha stabil PWM için pigpio kullanılabilir
from time import sleep
import RPi.GPIO as GPIO # BOARD numaralandırmasını ayarlamak için

# pigpio kullanmak için:
# sudo systemctl start pigpiod
# Eğer PiGPIOFactory kullanacaksanız, pigpio daemon'ının çalışıyor olması gerekir.
# Device.pin_factory = PiGPIOFactory() # Eğer pigpio kullanmak isterseniz bu satırı aktif edin

# BOARD pin numaralandırmasını kullanacağımızı belirtiyoruz
# Bu, gpiozero'nun da bu modu kullanmasını sağlar.
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False) # GPIO uyarılarını kapatabiliriz, isteğe bağlı

# --- PIN KONFİGÜRASYONU (BOARD NUMARALANDIRMASI) ---
# Sol Motor
L_MOTOR_ENA = 12  # PWM Hız Kontrolü (Enable A)
L_MOTOR_IN1 = 16  # Yön Pini 1
L_MOTOR_IN2 = 18  # Yön Pini 2

# Sağ Motor
R_MOTOR_ENB = 32  # PWM Hız Kontrolü (Enable B)
R_MOTOR_IN3 = 36  # Yön Pini 3
# !!! DİKKAT: Orijinal isteğinizde IN4 için de 36 belirtilmişti.
# Bu bir çakışmadır. IN4 için farklı bir pin olmalı.
# Örnek olarak 38 kullanıyorum. Lütfen kendi bağlantınızı kontrol edin!
R_MOTOR_IN4 = 38  # Yön Pini 4 (ÖRNEK - LÜTFEN KONTROL EDİN)

class MotorController:
    def __init__(self):
        """
        Motor kontrolcüsünü başlatır.
        Belirtilen BOARD pin numaralarına göre motorları tanımlar.
        """
        print("Motor kontrolcüsü başlatılıyor...")
        print(f"Sol Motor Pinleri: ENA={L_MOTOR_ENA}, IN1={L_MOTOR_IN1}, IN2={L_MOTOR_IN2}")
        print(f"Sağ Motor Pinleri: ENB={R_MOTOR_ENB}, IN3={R_MOTOR_IN3}, IN4={R_MOTOR_IN4}")

        if R_MOTOR_IN3 == R_MOTOR_IN4:
            print("\033[91mUYARI: Sağ motor için IN3 ve IN4 pinleri aynı ({R_MOTOR_IN3}). Bu durum motorun düzgün çalışmasını engeller.\033[0m")
            print("\033[91mLütfen sağ motorun IN4 pini için farklı bir GPIO pini bağlayıp kodu güncelleyin.\033[0m")
            # raise ValueError("Sağ motor IN3 ve IN4 pinleri aynı olamaz.") # Programı durdurmak için

        try:
            # gpiozero Motor(forward_pin, backward_pin, enable_pin)
            # forward_pin'e HIGH, backward_pin'e LOW verildiğinde motor ileri döner.
            # enable_pin PWM ile hız kontrolü sağlar.
            self.left_motor = Motor(forward=L_MOTOR_IN1, backward=L_MOTOR_IN2, enable=L_MOTOR_ENA, pwm=True)
            self.right_motor = Motor(forward=R_MOTOR_IN3, backward=R_MOTOR_IN4, enable=R_MOTOR_ENB, pwm=True)
            print("Motorlar başarıyla tanımlandı.")
        except Exception as e:
            print(f"\033[91mMotor tanımlanırken hata oluştu: {e}\033[0m")
            print("Lütfen pin bağlantılarını ve GPIO kütüphanesi kurulumunu kontrol edin.")
            print("Eğer pigpio kullanıyorsanız 'sudo systemctl start pigpiod' komutunu çalıştırdığınızdan emin olun.")
            raise # Hatayı yeniden yükselterek programın durmasını sağlayabiliriz veya farklı bir işlem yapabiliriz.

    def set_speeds(self, left_speed, right_speed):
        """
        Sol ve sağ motor hızlarını ayarlar.
        Hız değerleri -1.0 (tam geri) ile 1.0 (tam ileri) arasında olmalıdır.
        0 dur anlamına gelir.
        """
        # Hız değerlerini -1.0 ile 1.0 arasında sınırla
        left_speed = max(-1.0, min(1.0, left_speed))
        right_speed = max(-1.0, min(1.0, right_speed))

        if left_speed > 0:
            self.left_motor.forward(left_speed)
        elif left_speed < 0:
            self.left_motor.backward(abs(left_speed))
        else:
            self.left_motor.stop()

        if right_speed > 0:
            self.right_motor.forward(right_speed)
        elif right_speed < 0:
            self.right_motor.backward(abs(right_speed))
        else:
            self.right_motor.stop()
        # print(f"Hızlar ayarlandı: Sol={left_speed:.2f}, Sağ={right_speed:.2f}")

    def forward(self, speed=0.5):
        """Aracı belirtilen hızda ileri hareket ettirir."""
        print(f"İleri hareket, hız: {speed}")
        self.set_speeds(speed, speed)

    def backward(self, speed=0.5):
        """Aracı belirtilen hızda geri hareket ettirir."""
        print(f"Geri hareket, hız: {speed}")
        self.set_speeds(-speed, -speed)

    def turn_left(self, speed=0.5, intensity=1.0):
        """
        Aracı sola döndürür.
        intensity: 0 (yerinde döner) ile 1 (geniş kavis) arasında.
                   intensity = 1: Sağ motor tam hız, sol motor (1-intensity)*sağ_hız
                   intensity = 0: Sağ motor tam hız, sol motor tam ters hız (pivot dönüş)
                   intensity = 0.5 (varsayılan): Sağ motor tam hız, sol motor yarım hız ters
        Bu fonksiyonu projenizin dinamiklerine göre ayarlayabilirsiniz.
        Şimdilik basit bir versiyon: sol motor yavaş/geri, sağ motor ileri
        """
        print(f"Sola dönüş, hız: {speed}, intensity: {intensity}")
        # Basit bir sola dönüş: Sağ teker ileri, sol teker yavaş ileri veya duruk veya geri.
        # Yerinde keskin dönüş için: self.set_speeds(-speed, speed)
        # Daha yumuşak dönüş için:
        right_turn_speed = speed
        left_turn_speed = speed * (1.0 - (intensity * 1.5)) # intensity arttıkça sol teker hızı azalır, hatta eksiye düşer
        left_turn_speed = max(-1.0, min(1.0, left_turn_speed)) # Sınırları koru
        self.set_speeds(left_turn_speed, right_turn_speed)


    def turn_right(self, speed=0.5, intensity=1.0):
        """
        Aracı sağa döndürür.
        intensity: 0 (yerinde döner) ile 1 (geniş kavis) arasında.
        """
        print(f"Sağa dönüş, hız: {speed}, intensity: {intensity}")
        # Basit bir sağa dönüş: Sol teker ileri, sağ teker yavaş ileri veya duruk veya geri.
        # Yerinde keskin dönüş için: self.set_speeds(speed, -speed)
        # Daha yumuşak dönüş için:
        left_turn_speed = speed
        right_turn_speed = speed * (1.0 - (intensity * 1.5))
        right_turn_speed = max(-1.0, min(1.0, right_turn_speed)) # Sınırları koru
        self.set_speeds(left_turn_speed, right_turn_speed)

    def stop(self):
        """Aracı durdurur."""
        print("Motorlar durduruluyor.")
        self.left_motor.stop()
        self.right_motor.stop()
        # self.set_speeds(0, 0) # Alternatif olarak

    def cleanup(self):
        """GPIO kaynaklarını serbest bırakır."""
        print("Motor kaynakları temizleniyor.")
        self.stop() # Önce motorları durdur
        self.left_motor.close()
        self.right_motor.close()
        GPIO.cleanup() # RPi.GPIO tarafından kullanılan pinleri temizler
        print("Motor kaynakları başarıyla temizlendi.")

# Bu dosya doğrudan çalıştırıldığında test kodu çalışır
if __name__ == "__main__":
    print("Motor Kontrol Test Başlatılıyor...")
    print("LÜTFEN ARACIN TEKERLEKLERİNİN BOŞTA OLDUĞUNDAN EMİN OLUN (ÖRNEĞİN TAKOZ ÜZERİNDE)!")
    
    # Sağ motor pinlerini kontrol etmeyi unutmayın!
    # Eğer R_MOTOR_IN3 ve R_MOTOR_IN4 aynıysa (örn. 36), aşağıdaki satır hata verecektir
    # veya motorlar beklenmedik şekilde çalışacaktır.
    # Lütfen R_MOTOR_IN4 için farklı bir pin (örn: 38 veya 40) kullandığınızdan emin olun.
    if R_MOTOR_IN3 == R_MOTOR_IN4:
        print("\033[91m\nUYARI: Sağ motor pinleri (IN3 ve IN4) aynı. Test düzgün çalışmayabilir veya hata verebilir.\033[0m")
        print("\033[91mLütfen motor_kontrol.py dosyasındaki R_MOTOR_IN4 pinini düzeltin.\n\033[0m")
        # Testin devam etmesini engellemek için çıkış yapabilir veya kullanıcıdan onay alabilirsiniz.
        # exit()

    motors = None
    try:
        motors = MotorController()
        
        print("\n--- Test Başlıyor ---")
        
        print("\n1. İleri (Hız: 0.6) - 2 saniye")
        motors.forward(0.6)
        sleep(2)
        motors.stop()
        sleep(1)

        print("\n2. Geri (Hız: 0.4) - 2 saniye")
        motors.backward(0.4)
        sleep(2)
        motors.stop()
        sleep(1)

        print("\n3. Sola Dönüş (Keskin - Yerinde, Hız: 0.5) - 2 saniye")
        # Yerinde keskin dönüş için bir yöntem:
        motors.set_speeds(-0.5, 0.5) # Sol geri, Sağ ileri
        # Veya turn_left fonksiyonunu buna göre ayarlayabilirsiniz:
        # motors.turn_left(speed=0.5, intensity=0) # intensity=0 ile pivot
        sleep(2)
        motors.stop()
        sleep(1)

        print("\n4. Sağa Dönüş (Keskin - Yerinde, Hız: 0.5) - 2 saniye")
        # Yerinde keskin dönüş için bir yöntem:
        motors.set_speeds(0.5, -0.5) # Sol ileri, Sağ geri
        # Veya turn_right fonksiyonunu buna göre ayarlayabilirsiniz:
        # motors.turn_right(speed=0.5, intensity=0) # intensity=0 ile pivot
        sleep(2)
        motors.stop()
        sleep(1)

        print("\n5. Sola Yumuşak Dönüş (Hız: 0.6, Intensity: 0.7) - 2 saniye")
        motors.turn_left(speed=0.6, intensity=0.7) # Sağ teker daha hızlı
        sleep(2)
        motors.stop()
        sleep(1)

        print("\n6. Sağa Yumuşak Dönüş (Hız: 0.6, Intensity: 0.7) - 2 saniye")
        motors.turn_right(speed=0.6, intensity=0.7) # Sol teker daha hızlı
        sleep(2)
        motors.stop()
        sleep(1)

        print("\n--- Test Tamamlandı ---")

    except Exception as e:
        print(f"\033[91mTest sırasında bir hata oluştu: {e}\033[0m")
    finally:
        if motors:
            motors.cleanup()
        else: # MotorController başlatılamadıysa bile GPIO temizliği deneyelim
            GPIO.cleanup()
        print("Test programı sonlandırıldı.")