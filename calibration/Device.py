from bitalino import BITalino

class Device:
    """
    Interface for controlling a BITalino biosignal acquisition device.
    """

    def __init__(self, macAddress, samplingRate=1000, acqChannels=[1, 2]):
        """
        Initializes the BITalino device.
        
        :param macAddress: MAC address of the BITalino device.
        :param samplingRate: Sampling rate for data acquisition.
        :param acqChannels: List of acquisition channels to be used.
        """
        self.device = None
        self.macAddress = macAddress
        self.samplingRate = samplingRate
        self.acqChannels = acqChannels
        self.reset()

    def is_connected(self):
        """Returns whether the device is connected."""
        return self.device is not None

    def read(self, nSamples):
        """Reads data from the device."""
        if self.device:
            return self.device.read(nSamples)
        else:
            self.reset()
            return self.device.read(nSamples) if self.device else None

    def start(self):
        """Starts the device acquisition."""
        if self.device:
            self.device.start(self.samplingRate, self.acqChannels)

    def stop(self):
        """Stops the device acquisition."""
        if self.device:
            self.device.stop()

    def close(self):
        """Closes the device connection."""
        if self.device:
            self.device.close()
            self.device = None

    def reset(self):
        """Resets the device by stopping, closing, and reinitializing it."""
        if self.device:
            try:
                self.device.stop()
                self.device.close()
            except Exception as e:
                print(f"[Device.reset] Error stopping or closing device: {e}")
        self.device = None
        try:
            self.device = BITalino(self.macAddress)
            self.start()
        except Exception as e:
            print(f"[Device.reset] Error reinitializing device: {e}")
