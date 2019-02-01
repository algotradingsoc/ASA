#!/usr/bin/env python3

import socket
import struct

HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 65432        # Port to listen on (non-privileged ports are > 1023)
FMT = '?i'

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    count = 0
    while True:
        conn, addr = s.accept()
        with conn:
            print('Connected by', addr, count)
            count += 1
            data = conn.recv(1024)
            msg = struct.unpack(FMT, data)
            print(msg)
            conn.sendall(data)
            if not msg[0]:
                break
            