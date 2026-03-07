from memory_storage import save_memory
from memory_compressor import compress_memory


class OracleMemoryManager:

    def __init__(self):

        self.attention_threshold = 12

    def evaluate_memory(self, text_score, content):

        if text_score >= self.attention_threshold:

            path = save_memory(content, text_score)

            compressed = compress_memory(path)

            print("Memory stored:", compressed)

            return compressed

        return None