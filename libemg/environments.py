from abc import ABC, abstractmethod
import socket
from multiprocessing import Process
from collections import deque
from typing import overload


class Controller(ABC, Process):
    def __init__(self):
        super().__init__(daemon=True)
        self.action = None

    @abstractmethod
    def parse_predictions(self) -> list[float]:
        # Grab latest prediction (should we keep track of all or deque?)
        ...

    @abstractmethod
    def parse_proportional_control(self) -> list[float]:
        # Grab latest prediction (should we keep track of all or deque?)
        ...

    @abstractmethod
    def get_action(self):
        # Freeze single action so all data is parsed from that
        ...

    @overload
    def get_data(self, info: list[str]) -> tuple:
        ...

    @overload
    def get_data(self, info: str) -> list[float]:
        ...

    def get_data(self, info: list[str] | str) -> tuple | list[float]:
        # Take in which types of info you need (predictions, PC), call the methods, then return
        # This method was needed because we're making this a process, so it's possible that separate calls to
        # parse_predictions and parse_proportional_control would operate on different packets
        if isinstance(info, str):
            # Cast to list
            info = [info]

        action = self.get_action()
        self.action = action
        
        info_function_map = {
            'predictions': self.parse_predictions,
            'pc': self.parse_proportional_control
        }

        data = []
        for info_type in info:
            try:
                parse_function = info_function_map[info_type]
                result = parse_function()           
            except KeyError as e:
                raise ValueError(f"Unexpected value for info type. Accepted parameters are: {list(info_function_map.keys())}. Got: {info_type}.") from e

            data.append(result)

        data = tuple(data)  # convert to tuple so unpacking can be used if desired
        if len(data) == 1:
            data = data[0]
        return data


class SocketController(Controller):
    def __init__(self, ip: str = '127.0.0.1', port: int = 12346) -> None:
        super().__init__()
        self.ip = ip
        self.port = port
        self.data = deque(maxlen=1) # only want to read a single message at a time

    @abstractmethod
    def parse_predictions(self) -> list[float]:
        # Grab latest prediction (should we keep track of all or deque?)
        # Will be specific to controller
        ...

    @abstractmethod
    def parse_proportional_control(self) -> list[float]:
        # Grab latest prediction (should we keep track of all or deque?)
        # Will be specific to controller
        ...

    def get_action(self):
        if len(self.data) > 0:
            # Grab latest prediction and remove from queue so it isn't repeated
            return self.data.pop()

    def run(self) -> None:
        # Create UDP port for reading predictions
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.ip, self.port))

        while True:
            bytes, _ = self.sock.recvfrom(1024)
            message = str(bytes.decode('utf-8'))
            if message:
                # Data received
                self.data.append(message)
    

# Not sure if controllers should go in here or have their own file...
# Environment base class that takes in controller and has a run method

# Fitts should have the option for rotational.