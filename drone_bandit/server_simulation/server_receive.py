import socket
import torch
import io
import struct 
from torchsummary import summary
import torch.nn.functional as F
from simple_model import simple_model

SERVER_IP = '0.0.0.0'  
PORT = 5000
BUFFER_SIZE = 1024  

CP_MAP = [ 0, 0, 0, 1, 2]

model = simple_model()

print(model)
print(summary(model, (1, 200, 200)))


def run_back_inf_time (p, intermediate_output):

    features_output = model.conv2_block[CP_MAP[p]:](intermediate_output)
    features_output = features_output.view(features_output.size(0), -1)
    output = model.exit2_block(features_output)

    # hacer softmax de output
    output = F.softmax(output, dim=1)    
    print(f"Output final: {output}")

def start_server():
    # Create a TCP/IP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Bind the socket to the IP and port
    server_socket.bind((SERVER_IP, PORT))
    
    # Listen for incoming connections
    server_socket.listen(5)  # Allow up to 5 connections to queue
    print(f"Server listening on {SERVER_IP}:{PORT}")
    
    while True:  # Keep the server running to accept multiple clients
        # Accept a new connection
        conn, addr = server_socket.accept()
        print(f"Connected by {addr}")
        
        # Receive the integer first (int is 4 bytes)
        int_size = struct.calcsize('i')
        int_data = conn.recv(int_size)
        p = struct.unpack('i', int_data)[0]
        print(f"Received integer: {p}")
    
        # Receive the tensor data
        data = b''
        while True:
            packet = conn.recv(BUFFER_SIZE)
            if not packet:
                break
            data += packet  # Concatenate incoming packets
    
        # Deserialize the tensor from the received data
        intermediate_output = torch.load(io.BytesIO(data))

        # Run the inference logic
        run_back_inf_time(p, intermediate_output)
        
        # Clean up the connection
        conn.close()
        print("Connection closed, waiting for the next connection...")

if __name__ == "__main__":
    start_server()

