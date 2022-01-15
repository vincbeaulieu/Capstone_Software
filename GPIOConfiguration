from gpiozero import PWMLED
from signal import pause

# define part with PWMLED Object for a PWM signal and as a argument there can be multiple formats.
# >>> led = LED(17)
# >>> led = LED("GPIO17")
# >>> led = LED("BCM17")
# >>> led = LED("BOARD11")
# >>> led = LED("WPI0")
# >>> led = LED("J8:11")
# All of these refer to the same GPIO 17 definition.

LED = PWMLED("BOARD3")


# method used to control the LED
def pulseLED():
    LED.blink()
    pause()


# main function of the program.
if __name__ == "__main__":
    pulseLED()
