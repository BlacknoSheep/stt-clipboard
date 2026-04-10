from pythonosc import udp_client


class OSCClient:
    def __init__(self, ip: str = "127.0.0.1", port: int = 9000):
        self.client = udp_client.SimpleUDPClient(ip, port)

    def chatbox_input(self, text: str) -> None:
        if len(text) == 0:
            return
        self.client.send_message("/chatbox/input", [text, True, True])
