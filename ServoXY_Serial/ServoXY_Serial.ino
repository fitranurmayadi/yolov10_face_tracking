//MCU Board : ESP32C3
//Servo X pin D2(GPIO4) and Servo Y pin D3(GPIO5)


#include <pwmWrite.h>

Pwm pwm = Pwm();

const int servoX = 4;
const int servoY = 5;
int angleX = 0;
int angleY = 0;

void setup() {
  Serial.begin(9600); // Start serial communication at 9600 baud rate
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n'); // Read input from serial monitor until newline character
    int xIndex = input.indexOf("x=");
    int yIndex = input.indexOf("y=");
    
    if (xIndex != -1 && yIndex != -1) {
      angleX = input.substring(xIndex + 2, input.indexOf(' ', xIndex)).toInt();
      angleY = input.substring(yIndex + 2).toInt();
      
      if (angleX >= 0 && angleX <= 180) {
        pwm.writeServo(servoX, angleX);
      } else {
        Serial.println("Invalid angle for servoX. Please enter a value between 0 and 180.");
      }
      
      if (angleY >= 0 && angleY <= 180) {
        pwm.writeServo(servoY, angleY);
      } else {
        Serial.println("Invalid angle for servoY. Please enter a value between 0 and 180.");
      }
    } else {
      Serial.println("Invalid command. Use 'x=<angle> y=<angle>'.");
    }
  }
}
