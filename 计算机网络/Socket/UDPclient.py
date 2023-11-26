from socket import *

serverName = "hostname"
serverPort = 12000
clientSocket = socket.socket(socket.AF_INET, SOCK_DGRAM)
message = input("message")
clientSocket.sendto(message.encode(),(serverName, serverPort))
recv_message,recv_adress = clientSocket.recvfrom(1024)
print("received message")
clientSocket.close()


