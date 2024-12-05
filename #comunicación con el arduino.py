#comunicación con el arduino
import serial
import time

arduino = serial.Serial('COM5',9600)
time.sleep(2)
data = arduino.readline()
print(data)
arduino.close()