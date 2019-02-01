#!/usr/bin/env python3

import socket
import struct

import argparse

def send_message(msg, HOST='127.0.0.1', PORT=65432):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(msg)
        data = s.recv(1024)
        print('Received', repr(data))
    return repr(data)
    

    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Send message to socket.')
    parser.add_argument('-e', '--exit', action='store_true',
                        help='sends exit message to socket')
    parser.add_argument('-i', '--in', default=0, type=int,
                        help='input for message')
    
    FMT = '?i'
    
    args = vars(parser.parse_args())
    exit = args['exit']
    val = args['in']
    
    if not exit:
        msg = struct.pack(FMT, True, val)
        send_message(msg)
    else:
        msg = struct.pack(FMT, False, 0)
        send_message(msg)
    print("Sent",msg)