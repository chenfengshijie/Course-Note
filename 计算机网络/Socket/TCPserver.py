from socket import *

serverPort = 12000
serverSocket = socket(AF_INET, SOCK_STREAM)
serverSocket.blind(('',serverPort))
serverSocket.listen(1)
print("server is listening on")
while True:
    connectionSocket,addr = serverSocket.accept()
    sentence = connectionSocket.recv(1024).decode()
    upper_sentence = sentence.upper()
    connectionSocket.send(upper_sentence.encode())
    connectionSocket.close()
    
    