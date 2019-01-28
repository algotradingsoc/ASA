import socket
from struct import pack


HOST = '127.0.0.1'
PORT = 65430
PACKING = '?ifi'


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

        
def init_bokeh(agents):
    msg = pack(PACKING, True, agents, 0, 0)
    send_to_socket(msg)
    
    
def send_agent_data(agent_id, value, time):
    msg = pack(PACKING, False, agent_id, value, time)
    send_to_socket(msg)
    
    
if __name__=='__main__':
    init_bokeh(3)
    