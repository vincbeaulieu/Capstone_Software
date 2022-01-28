from myo import init, Hub, ApiDeviceListener
import time

def main():
  init()
  hub = Hub()
  listener = ApiDeviceListener()
  with hub.run_in_background(listener.on_event):
    print("Waiting for both Myos to connect ...")
    devices = listener.connected_devices
    if len(devices) != 2:
      print("Both Myos not connected.")
      return
    print("Hello, Myo! Requesting EMG ...")
    for device in devices:
      device.request_rssi()
      device.stream_emg(True)
      while hub.running and device.connected and not device.rssi:
        print("Waiting for RRSI...")
        time.sleep(0.001)
      print("RSSI:", device.rssi)
    while(True):
      time.sleep(0.008)
      for (i, device) in enumerate(devices, start=1):
        print(f'EMG Device {i}: {device.emg}')

if __name__ == "__main__":
  main()