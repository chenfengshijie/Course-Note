from socket import *

serverName = 'serverName'
serverPort = 12000
clientSocket = socket(AF_INET, SOCK_STREAM)
clientSocket.connect(serverName, serverPort)
sentence = input("input something!")
clientSocket.send(sentence.encode())
modified = clientSocket.recv(1024)
print("recieve message from server")
clientSocket.close()
