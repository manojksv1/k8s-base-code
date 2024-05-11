import time

class IntegerCapture:
    def __init__(self, capture_threshold, release_threshold):
        self.capture_threshold = capture_threshold
        self.release_threshold = release_threshold
        self.captured_integers = []

    def monitor_integers(self):
        current_integer = self.get_current_integer()  # Function to get current integer value
        if current_integer < self.capture_threshold:
            self.capture_integer(current_integer)
        elif current_integer > self.release_threshold:
            self.release_integer(current_integer)
        else:
            print("Integer within thresholds:", current_integer)

    def capture_integer(self, integer):
        print("Capturing integer:", integer)
        self.captured_integers.append(integer)

    def release_integer(self, integer):
        if integer in self.captured_integers:
            print("Releasing integer:", integer)
            self.captured_integers.remove(integer)

    def get_current_integer(self):
        # Simulated function to get current integer value
        return 10  # Placeholder value for demonstration purposes

if __name__ == "__main__":
    capture_threshold = 5
    release_threshold = 15
    integer_capturer = IntegerCapture(capture_threshold, release_threshold)

    # Continuously monitor integers
    while True:
        integer_capturer.monitor_integers()
        time.sleep(2)  # Simulate 2 seconds interval for integer change
