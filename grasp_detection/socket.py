from typing import Optional, List, Union

import zmq
import numpy as np

class ZmqSocket:
    def __init__(self, cfgs):
        # init socket with port number
        zmq_context = zmq.Context()
        self.socket = zmq_context.socket(zmq.REP)
        self.socket.bind("tcp://*:" + str(cfgs.port))

    def send_array(
        self, 
        data: np.ndarray, 
        flags: int = 0, 
        copy: bool = True, 
        track: bool = False
    ) ->  Optional[int]:
        """send a numpy array with metadata"""
        md = dict(
            dtype = str(data.dtype),
            shape = data.shape,
        )
        self.socket.send_json(md, flags|zmq.SNDMORE)
        
        return self.socket.send(np.ascontiguousarray(data), flags, copy=copy, track=track)

    def recv_array(
        self,
        flags: int = 0,
        copy: bool = True,
        track: bool = False
    ) -> np.ndarray:
        """Receive a NumPy array."""
        md = self.socket.recv_json(flags=flags)
        msg = self.socket.recv(flags=flags, copy=copy, track=track)
        data = np.frombuffer(msg, dtype=md['dtype'])

        return data.reshape(md['shape'])
    
    def send_msgs(
        self,
        msgs: List[Union[List[float], str]]
    ) -> Optional[bool]:
        """Send list of messages - list of numbers or a string"""
        for msg in msgs:
            if isinstance(msg, list) and all(isinstance(num, float) for num in msg):
                # if the message is a list of floats
                data = np.ndarray(msg)
                self.send_array(data)
            elif isinstance(msg, str):
                # if the message 
                self.socket.send_string(msg)
        