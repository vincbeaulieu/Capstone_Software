#include <Arduino.h>

int val_a0 = 0;
int val_a1 = 0;
int val_a2 = 0;
int val_a3 = 0;
int val_a4 = 0;

void setup() {
  Serial.begin(9600);
}

void loop() {
  val_a0 = analogRead(PIN_A0);
  val_a1 = analogRead(PIN_A1);
  val_a2 = analogRead(PIN_A2);
  val_a3 = analogRead(PIN_A3);
  val_a4 = analogRead(PIN_A4);

  Serial.println("----------------------------");
  Serial.print("Pin 0: ");
  Serial.println(val_a0);
  Serial.print("Pin 1: ");
  Serial.println(val_a1);  
  Serial.print("Pin 1: ");
  Serial.println(val_a2);
  Serial.print("Pin 3: ");
  Serial.println(val_a3);
  Serial.print("Pin 4: ");
  Serial.println(val_a4);
  Serial.println("----------------------------");

  delay(200);
}