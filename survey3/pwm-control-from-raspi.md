---
description: Python scripts to control the Survey3 using a Raspberry Pi computer
---

# PWM Control (from RasPi)

```python
import RPi.GPIO as GPIO
import time

GPIO.setwarnings(False)

pin = 18  #set (BCM) GPIO pin to send GPIO.HIGH pulse

try:
    while True:

        print("Command Sent\n")

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(0.002) #2000us pulse width
        GPIO.cleanup()

        time.sleep(0.1)

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(pin, GPIO.OUT)
        time.sleep(0.001) #1000us pulse width (do nothing)
        GPIO.cleanup()

        time.sleep(1.5) #time delay between GPIO.HIGH pulses, >1.5s recommended

#to exit type CTRL+Z
except KeyboardInterrupt:
    pass
```

You could also set it up to take user keyboard inputs:

```python
GPIO.setwarnings(False)

pin = 18  #set (BCM) GPIO pin to send GPIO.HIGH pulse

GPIO.setmode(GPIO.BCM)
GPIO.setup(pin, GPIO.OUT)
time.sleep(0.001)
GPIO.cleanup()

print("\nCommands:")
print("t + Enter: Trigger (2000us)")
print("s + Enter: SD Card Mount/Unmount (1500us)")
print("e + Enter: Exit\n")

while True:
    key = input(">")
    if key == "t":
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(0.002)
        GPIO.cleanup()

        time.sleep(0.1)
        print("Trigger")

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(pin, GPIO.OUT)
        time.sleep(0.001)
        GPIO.cleanup()

    if key == "s":
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(0.0015)
        GPIO.cleanup()

        time.sleep(0.1)
        print("SD Mount/UnMount")

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(pin, GPIO.OUT)
        time.sleep(0.001)
        GPIO.cleanup()

    if key == "e":
        print("Exiting\n")
        break

    else:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(pin, GPIO.OUT)
        time.sleep(0.001)
        GPIO.cleanup()
```

The code above allows you to type the letter "t" and then press the "Enter" key
and it will take a photo. You can type "s + Enter" and it will unmount the uSD
card from the camera and the RasPi will then show it as a removable drive.
Typing "s + Enter" again will unmount from RasPi and mount the uSD back to the
camera. Typing "e + Enter" exits the program.
