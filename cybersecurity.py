import os
import json
import base64
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding

AES_KEY = b'x' * 32  
HMAC_KEY = b'y' * 32  

def encrypt_data(ai_json):
    """Encrypt AI output JSON with AES-256-CBC and HMAC-SHA256"""
    plaintext = json.dumps(ai_json).encode('utf-8')

    #Random IV
    iv = os.urandom(16)

    cipher = Cipher(
        algorithms.AES(AES_KEY),
        modes.CBC(iv),
        backend=default_backend()
    )
    encryptor = cipher.encryptor()

    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(plaintext) + padder.finalize()

    ciphertext = encryptor.update(padded_data) + encryptor.finalize()

    h = hmac.HMAC(HMAC_KEY, hashes.SHA256(), backend=default_backend())
    h.update(plaintext)
    signature = h.finalize()

    return {
        "ciphertext": base64.b64encode(ciphertext).decode('utf-8'),
        "hmac": base64.b64encode(signature).decode('utf-8'),
        "iv": base64.b64encode(iv).decode('utf-8')
    }

if __name__ == "__main__":
    input_path = "syn_data.json"  

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            ai_output = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: {input_path} not found.")
        exit(1)
    except json.JSONDecodeError:
        print(f"❌ Error: Failed to parse {input_path}. Ensure it's valid JSON.")
        exit(1)

    
    encrypted = encrypt_data(ai_output)

    os.makedirs("web", exist_ok=True)
    with open("web/data/encrypted_dataa.json", "w") as f:
        json.dump(encrypted, f, indent=2)
        f.flush()

    print("=" * 50)
    print("✅ Encryption successful! Output saved to web/encrypted_dataa.json")
    print(f"AES Key: {AES_KEY} ({len(AES_KEY)} bytes)")
    print(f"HMAC Key: {HMAC_KEY} ({len(HMAC_KEY)} bytes)")
    print("=" * 50)
