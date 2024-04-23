import RPi.GPIO as gp
import os
import cv2
import subprocess
import time
import signal
from PIL import Image
from pathlib import Path

def main():
    gp.setwarnings(False)
    gp.setmode(gp.BOARD)
    
    # Setup GPIO
    
    gp.setup(7, gp.OUT)
    gp.setup(11, gp.OUT)
    gp.setup(12, gp.OUT)

    # Setup button

    button = 37 # BOARD pin, not BCM
    gp.setup(button,gp.IN,pull_up_down=gp.PUD_UP)
    
 	# Setup Lights

    gp.setup(36, gp.OUT)
    gp.setup(38, gp.OUT)
    gp.setup(40, gp.OUT)
    
    # Clear lights
    
    lightOff("all")
    
    while True:
        buttonState = gp.input(button)
        lightOn('green')
        if buttonState == 0:
            lightOff('all')
            lightOn('blue')
            print('Button is pressed')
            try:
                imager = subprocess.Popen(['python', '/home/picam/Desktop/CameraNEW.py'], start_new_session=True)
                imager.wait(timeout = 13)
                lightOn('green')
                print('Saving images')
                try:
                    save('/media/picam/Samsung256GB/Captures/', cv2.imread('/media/picam/Samsung256GB/Buffer/capture_1.tiff'), 'RGB')
                    save('/media/picam/Samsung256GB/Captures/', cv2.imread('/media/picam/Samsung256GB/Buffer/capture_2.tiff'), 'NIR')
                    save('/media/picam/Samsung256GB/Captures/', cv2.imread('/media/picam/Samsung256GB/Buffer/capture_4.tiff'), 'RED')
                    print('Images saved successfully')
                    os.remove('/media/picam/Samsung256GB/Buffer/capture_1.tiff')
                    os.remove('/media/picam/Samsung256GB/Buffer/capture_2.tiff')
                    os.remove('/media/picam/Samsung256GB/Buffer/capture_4.tiff')
                    attempt = 0
                except:
                    print('Failed to open image file')
                    lightOn('red')
                    subprocess.run(['python', '/home/picam/Desktop/Reset.py'], start_new_session=True)
                    exit()
            except:
                print('Camera timeout, terminating Capture')
                gp.output(7, False)
                gp.output(11, True)
                gp.output(12, True)
                lightOn('red')
                os.killpg(os.getpgid(imager.pid), signal.SIGTERM)
                subprocess.run(['python', '/home/picam/Desktop/Reset.py'], start_new_session=True)
                exit()
            lightOff('blue')
            

def lightOn(color):
    if color == "red":
        gp.output(36, True)
    if color == "green":
        gp.output(38, True)
    if color == "blue":
        gp.output(40, True)
        
def lightOff(color):
    if color == "red":
        gp.output(36, False)
    if color == "green":
        gp.output(38, False)
    if color == "blue":
        gp.output(40, False)
    if color == "all":
        gp.output(36, False)
        gp.output(38, False)
        gp.output(40, False)

def save(image_path, image, imtype):
    i = 1
    image_path = Path(image_path + imtype)
    while Path((str(image_path) + str(i) + '.tiff')).is_file():
        i = i + 1
    cv2.imwrite(str(image_path) + str(i) + '.tiff', image)

if __name__ == "__main__":
    main()

    #gp.output(7, False)
    #gp.output(11, False)
    #gp.output(12, True)
