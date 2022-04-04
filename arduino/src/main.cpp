#include <Arduino.h>

void readSensors();
void findMaxValue();
void rumbleOn(int);
void rumbleOff();

int valArray[5];
int maxValue;
int serialRead = -1;
bool isEnabled = false;
int valueSum = 0;
int valueAvg = 0;
int valueOutput = 0;

void setup() {
  Serial.begin(9600);
  analogWrite(6, 255);
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
    valueSum = 0;
    for (size_t i = 0; i < 16; i++)
    {
      readSensors();
      findMaxValue();
      valueSum += maxValue;
      delay(10);
    }
    
    valueAvg = valueSum >> 4; // Divide by 16

    if (valueAvg > 100)
    {
      valueOutput = 255 - map(valueAvg, 100, 1023, 190, 255);
      rumbleOn(valueOutput);
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

void rumbleOn(int value)
{
  analogWrite(6, value);
  // Serial.print("ON, ");
  // Serial.print(maxValue);
  // Serial.print(", output=");
  // Serial.println(value);
}

void rumbleOff()
{
  analogWrite(6, 255);
  // Serial.print("OFF, ");
  // Serial.println(maxValue);
}