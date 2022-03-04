#include <Arduino.h>

int valArray[5];
int maxValue;

void setup() {
  Serial.begin(9600);

  analogWrite(PIN_A5, 255);
}

void loop() {

  for (size_t i = 0; i < 5; i++)
  {
    valArray[i] = analogRead(i);
  }

  maxValue = valArray[0];
  for (size_t i = 1; i < 5; i++)
  {
    if (valArray[i] > maxValue)
    {
      maxValue = valArray[i];
    }
  }
  
  if (maxValue > 300)
  {
    analogWrite(PIN_A5, 255/2);
    Serial.print("ON, ");
    Serial.println(maxValue);
  }
  else 
  {
    analogWrite(PIN_A5, 255);
    Serial.print("OFF, ");
    Serial.println(maxValue);
  }

  delay(200);
}