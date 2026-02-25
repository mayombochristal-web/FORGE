import re

class Demodulator:
    def __init__(self):
        self.alphabet = "abcdefghijklmnopqrstuvwxyz"

    def decode_stream(self, m_history):
        substance = []
        for i in range(1, len(m_history)):
            delta = abs(m_history[i] - m_history[i-1])
            if delta > 0.05:
                char_idx = int(delta * 10) % 26
                substance.append(self.alphabet[char_idx])
            else:
                substance.append(" ")
        text = "".join(substance)
        text = re.sub(r'(.)\1{2,}', r'\1', text)
        text = re.sub(r'(\w{5})', r'\1 ', text)
        return text.capitalize().strip()[:100] + "."