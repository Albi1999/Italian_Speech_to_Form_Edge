import base64
import os

class AudioEncoder:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def encode(self) -> str:
        """
        Encodes the audio file to a Base64 string (UTF-8 encoding).

        Returns:
            str: The Base64 encoded audio data, or None if an error occurs.
        """
        try:
            #Check file size
            file_size_bytes = os.path.getsize(self.file_path)
            max_size_bytes = 10 * 1024 * 1024 # 10MB limit
            if file_size_bytes > max_size_bytes:
                print(f"Error: File size exceeds maximum allowed size ({max_size_bytes} bytes).")
                return None

            with open(self.file_path, "rb") as f:
                data = f.read()
            return base64.b64encode(data).decode("utf-8")
        except FileNotFoundError:
            print(f"Error: File not found at path: {self.file_path}")
            return None
        except Exception as e:
            print(f"An error occurred during encoding: {e}")
            return None

    def __call__(self) -> str:
         return self.encode()