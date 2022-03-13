#include <Arduino.h>

void readSensors();
void findMaxValue();
void rumbleOn();
void rumbleOff();

int valArray[5];
int maxValue;
int serialRead = -1;
bool isEnabled = false;

void setup() {
  Serial.begin(9600);
  analogWrite(PIN_A5, 255);
}

void loop() {

  serialRead = Serial.read();
  // Serial.println(serialRead);

  if (serialRead == 1)
  {
    isEnabled = true;
  }
  else if(serialRead == 0)
  {
    isEnabled = false;
  }

  if (isEnabled)
  {
    readSensors();
    findMaxValue();
    
    if (maxValue > 300)
    {
      rumbleOn();
    }
    else 
    {
      rumbleOff();
    }
  }
  else 
  {
    rumbleOff();
  }

  delay(10);
}

void readSensors()
{
  for (size_t i = 0; i < 5; i++)
  {
    valArray[i] = analogRead(i);
  }
}

void findMaxValue()
{
  maxValue = valArray[0];
  for (size_t i = 1; i < 5; i++)
  {
    if (valArray[i] > maxValue)
    {
      maxValue = valArray[i];
    }
  }
}

void rumbleOn()
{
  analogWrite(PIN_A5, 0);
  // Serial.print("ON, ");
  // Serial.println(maxValue);
}

void rumbleOff()
{
  analogWrite(PIN_A5, 255);
  // Serial.print("OFF, ");
  // Serial.println(maxValue);
}