import time
import subprocess
import os

def main():
	# Sleep, then reopen program
	time.sleep(2)
	os.system('python /home/picam/Desktop/Camera.py')
	#subprocess.Popen(['python', '/home/picam/Desktop/Camera.py'])

if __name__ == "__main__":
    main()
