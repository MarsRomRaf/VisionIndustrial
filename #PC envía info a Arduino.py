#PC envía info a Arduino
import serial
import time

arduino=serial.Serial('COM6',9600)
time.sleep(2)

arduino.write(bytes('1','utf-8'))
time.sleep(2)
arduino.write(bytes('0','utf-8'))
time.sleep(2)
