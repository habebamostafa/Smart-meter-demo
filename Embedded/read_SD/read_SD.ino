#include <SPI.h>
#include <SD.h>
#include <SoftwareSerial.h>


// Pins
#define PLC_LED           3      // PLC mode indicator
#define RF_LED            2      // RF mode indicator
#define chipSelect        10     // SD card CS pin

#define SEND_INTERVAL     1000   // Delay between transmissions (ms)

// SoftwareSerial ports for PLC/RF simulation
SoftwareSerial PLC_softSerial(8, 9);  // RX, TX (PLC)
SoftwareSerial RF_softSerial(6, 7);   // RX, TX (RF)

void setup() {

  Serial.begin(9600);
  PLC_softSerial.begin(9600);
  RF_softSerial.begin(9600);

  pinMode(PLC_LED, OUTPUT);
  pinMode(RF_LED, OUTPUT);

  // Initialize SD card
  if (!SD.begin(chipSelect)) 
  {
    return;
  }

  File csvFile = SD.open("data.csv");

  if (!csvFile) 
  {
    return;
  }

  
  char c;

  while (csvFile.available()) 
  {
    String line = csvFile.readStringUntil('\n');
    Serial.println(line);

    while (1)
    {
      // Wait for mode selection (PLC/RF)
      c = Serial.read();
      if(c == 'P') // PLC mode
      {
        digitalWrite(PLC_LED, HIGH);
        digitalWrite(RF_LED, LOW);
        PLC_softSerial.println(line);
      }
      else if(c == 'R') // RF mode
      {
        digitalWrite(RF_LED, HIGH);
        digitalWrite(PLC_LED, LOW);
        RF_softSerial.println(line);
      }
      else if (c == '\n') 
      {
        break;
      }
    }
    delay(SEND_INTERVAL);
  }

  csvFile.close();

}

void loop() {
  // Do nothing
}
