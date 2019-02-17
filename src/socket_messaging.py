import socket
from struct import pack


HOST = '127.0.0.1'
PORT = 65430


def send_to_socket(msg):
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((HOST, PORT))
                s.sendall(msg)
        except ConnectionResetError:
            continue
        except ConnectionRefusedError:
            continue
        break
        
    
def send_data(PACKING, *args): 
    msg = pack(PACKING, *args)
    send_to_socket(msg)
    
    
if __name__=='__main__':
    send_data('?ifi', True, 3, 0, 0)
    