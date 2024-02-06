/*
 * HX711 simple print weight in grams developed using Arduino Uno R3
 * 
 * Library dependencies:
 * - HX711 v0.7.5 https://github.com/bogde/HX711
 */
#include "HX711.h"

const int DOUT_PIN = 2;
const int SCK_PIN = 3;

HX711 scale;

void setup() {
  Serial.begin(57600);
  Serial.println("Initializing the scale");

  scale.begin(DOUT_PIN, SCK_PIN);        
  scale.set_scale(395.74);
  // reset the scale to 0
  scale.tare(); 
}

void loop() {
  Serial.print("grams:");
  Serial.println(scale.get_units(), 1);
  delay(100);
}
