import atexit
import sys
import time
from streamlit import cli as stcli
from threading import Thread
import socket

def stop_server():
    stcli._stop_streamlit()

def listen_for_shutdown_signal():
    shutdown_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    shutdown_socket.bind(('localhost', 8502))
    shutdown_socket.listen(1)
    conn, addr = shutdown_socket.accept()
    with conn:
        print('Shutdown signal received')
        stop_server()

atexit.register(stop_server)

shutdown_listener = Thread(target=listen_for_shutdown_signal)
shutdown_listener.start()

if __name__ == '__main__':
    sys.argv = ["streamlit", "run", "app.py", "--browser.serverAddress", "0.0.0.0", "--server.port", "8501", "--server.headless", "true"]
    sys.exit(stcli.main())