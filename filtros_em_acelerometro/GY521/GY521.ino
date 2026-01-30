#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

Adafruit_MPU6050 mpu;
int med = 100; // Define o valor da media
float accz = 0, accy = 0;
float speedz = 0, speedy = 0;
float distz = 0, disty = 0;
float media_accz = 0, media_accy = 0;
float tempoAtual = 0;
float intervalo = 0;
float ultimoTempo = 0;
float anterior_accz = 0, anterior_accy = 0;
float anterior_speedz = 0, anterior_speedy = 0;
bool parado;

void setup(void) {
  Serial.begin(115200);
  while (!Serial)
    delay(10); // will pause Zero, Leonardo, etc until serial console opens

  // Try to initialize!
  if (!mpu.begin()) {
    while (1) {
      delay(10);
    }
  }

  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  delay(100);
}

void loop() {
  accz = 0;
  accy = 0;
  media_accz = 0;
  media_accy = 0;
  tempoAtual = millis();
  for(int i=0; i<med; i++){
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);
    media_accz += a.acceleration.z - 7.8;
    media_accy += a.acceleration.y + 0.11;
  }
  intervalo = (tempoAtual - ultimoTempo)/1000;
  accz = media_accz/med;
  accy = media_accy/med;
  speedz = (accz - anterior_accz)/intervalo;
  speedy = (accy - anterior_accy)/intervalo;
  distz = (speedz - anterior_speedz)/intervalo;
  disty = (speedy - anterior_speedy)/intervalo;
  
  Serial.print(accy);
  Serial.print(",");
  Serial.print(accz);
  //Serial.print(speedy);
  //Serial.print(",");
  //Serial.print(disty);
/*
  Serial.print(",");
  
  Serial.print(",");
  Serial.print(speedz);
  Serial.print(",");
  Serial.print(distz);
*/
  anterior_accz = accz;
  anterior_accy = accy;
  anterior_speedz = speedz;
  anterior_speedy = speedy;
  ultimoTempo = tempoAtual;
  parado = 0;
  Serial.println("");
}
