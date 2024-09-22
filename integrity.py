import hashlib
import base64

def calculate_sha384_integrity(file_path):
    # Open the file in binary mode
    with open(file_path, 'rb') as file:
        file_content = file.read()

    # Calculate the SHA-384 hash
    sha384_hash = hashlib.sha384(file_content).digest()

    # Base64 encode the hash
    base64_encoded_hash = base64.b64encode(sha384_hash).decode()

    return f"sha384-{base64_encoded_hash}"

# Example usage
file_path = 'd:/bootstrap.min.css'
integrity_value = calculate_sha384_integrity(file_path)
print(integrity_value)
