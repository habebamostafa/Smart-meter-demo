import serial
import time

# --- CONFIGURATION ---
SERIAL_PORT = 'COM5'   # Change to match your port (e.g., 'COM3' on Windows or '/dev/ttyUSB0' on Linux/Mac)
BAUD_RATE = 9600
TIMEOUT = 1             # Timeout in seconds

# --- SERIAL SETUP ---
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
time.sleep(2)  # Give Arduino time to reset

# --- FUNCTIONS ---

def read_from_arduino():
    """Waits until a line is received from the Arduino."""
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            return line


def send_to_arduino(message):
    """Sends a string to the Arduino with newline."""
    ser.write((message + '\n').encode('utf-8'))

# --- MAIN INTERACTION LOOP ---

print("Serial communication started. Type and send data to Arduino.")
try:
    while True:
        # Check for incoming data
        response = read_from_arduino()
        if response:
            print(f"Arduino says: {response}")

        # Send user input
        user_input = input("Enter a command to send (or 'q' to quit): ")
        if user_input.lower() == 'q':
            break
        send_to_arduino(user_input)

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    ser.close()
    print("Serial connection closed.")
