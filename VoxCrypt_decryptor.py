#!/usr/bin/env python3
"""
VoxCrypt-V2 Decryptor - Decryptor for VoxCrypt (.vxc)
"""

import os
import sys
import argparse
from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

# Constants matching the encryptor EXACTLY
HKDF_INFO = b"VoxCrypt v2 Key Derivation"
SALT_SIZE = 32
NONCE_SIZE = 12
HMAC_SIZE = 64

class VoxDecrypt:
    @staticmethod
    def derive_keys(seed, salt=None, info=HKDF_INFO, length=32):
        """HKDF with BLAKE2b for key derivation"""
        hkdf = HKDF(
            algorithm=hashes.BLAKE2b(64),
            length=length,
            salt=salt,
            info=info,
            backend=default_backend()
        )
        return hkdf.derive(seed)

    @staticmethod
    def load_private_key(key_path):
        """Load private key from file"""
        with open(key_path, 'rb') as kf:
            key_data = kf.read().strip()
        
        try:
            if key_data.startswith(b'-----BEGIN'):
                return serialization.load_pem_private_key(
                    key_data,
                    password=None,
                    backend=default_backend()
                )
            elif len(key_data) == 32:
                return x25519.X25519PrivateKey.from_private_bytes(key_data)
            else:
                raise ValueError(f"Unknown key format: {len(key_data)} bytes")
        except Exception as e:
            raise ValueError(f"Failed to load key: {str(e)}")

    @staticmethod
    def read_pem_object(file_obj, start_marker):
        """Read a PEM object from file until end marker"""
        data = b''
        line = file_obj.readline()
        
        # Read until we find the start marker
        while line and start_marker not in line:
            line = file_obj.readline()
        
        if not line:
            raise ValueError(f"PEM start marker {start_marker} not found")
        
        data += line
        
        # Read until we find the end marker
        end_marker = start_marker.replace(b'BEGIN', b'END')
        while line and end_marker not in line:
            line = file_obj.readline()
            if line:
                data += line
        
        if end_marker not in data:
            raise ValueError(f"PEM end marker {end_marker} not found")
        
        return data

    @staticmethod
    def is_text_data(data):
        """Check if data is likely text (not binary)"""
        if len(data) == 0:
            return False
        
        # Check for common binary file signatures
        binary_signatures = [
            b'\xff\xd8\xff',  # JPEG
            b'\x89PNG',        # PNG
            b'%PDF',           # PDF
            b'\x50\x4B\x03',   # ZIP/Office
            b'\x49\x49\x2A',   # TIFF
            b'\x4D\x4D\x00',   # TIFF
            b'\x47\x49\x46',   # GIF
            b'\x52\x49\x46\x46',  # WEBP, AVI
            b'\x1A\x45\xDF\xA3',  # WebM, MKV
            b'\x00\x00\x00\x20',  # MP4
            b'\x49\x44\x33',      # MP3
            b'\xFF\xFB',          # MP3
            b'\xFF\xF3',          # MP3
            b'\xFF\xF2',          # MP3
        ]
        
        for sig in binary_signatures:
            if data.startswith(sig):
                return False
        
        # Check if data contains many non-printable characters
        printable_count = 0
        for byte in data[:1000]:  # Check first 1000 bytes
            if 32 <= byte <= 126 or byte in (9, 10, 13):  # Printable ASCII + tab, LF, CR
                printable_count += 1
        
        return printable_count / min(len(data), 1000) > 0.8  # 80% printable

    @staticmethod
    def decrypt_file(input_path, key_path, output_path):
        """Decrypt .vxc file"""
        try:
            with open(input_path, 'rb') as f:
                header = f.read(5)
                
                if header == b'VXC3H':
                    print("Detected VXC3H format (static file)")
                    return VoxDecrypt._decrypt_static(f, key_path, output_path)
                else:
                    raise ValueError(f"Invalid file format header: {header}")
        except Exception as e:
            raise ValueError(f"File reading failed: {str(e)}")

    @staticmethod
    def _decrypt_static(file_obj, key_path, output_path):
        """Decrypt static .vxc file with proper PEM handling"""
        try:
            print("Reading file components...")
            
            # Read fixed-size components
            salt = file_obj.read(SALT_SIZE)
            if len(salt) != SALT_SIZE:
                raise ValueError(f"Invalid salt size: {len(salt)} bytes")
            print(f"Salt: {salt.hex()[:16]}...")
            
            hmac_value = file_obj.read(HMAC_SIZE)
            if len(hmac_value) != HMAC_SIZE:
                raise ValueError(f"Invalid HMAC size: {len(hmac_value)} bytes")
            print(f"HMAC: {hmac_value.hex()[:16]}...")
            
            # Read PEM-formatted ephemeral public key (variable length)
            print("Reading ephemeral public key (PEM format)...")
            ephemeral_pub = VoxDecrypt.read_pem_object(file_obj, b'-----BEGIN PUBLIC KEY-----')
            print(f"Ephemeral pub length: {len(ephemeral_pub)} bytes")
            
            # Read fixed-size nonce
            nonce = file_obj.read(NONCE_SIZE)
            if len(nonce) != NONCE_SIZE:
                raise ValueError(f"Invalid nonce size: {len(nonce)} bytes")
            print(f"Nonce: {nonce.hex()}")
            
            # Read remaining data as ciphertext
            ciphertext = file_obj.read()
            if len(ciphertext) == 0:
                raise ValueError("Ciphertext is empty")
            print(f"Ciphertext: {len(ciphertext)} bytes")
            
            # Load private key
            print("Loading private key...")
            private_key = VoxDecrypt.load_private_key(key_path)
            print("Private key loaded successfully")
            
            # Verify HMAC first
            print("Verifying HMAC...")
            data_to_hmac = ephemeral_pub + nonce + ciphertext
            hmac_key = VoxDecrypt.derive_keys(salt, info=b"HMAC key")
            h = hmac.HMAC(hmac_key, hashes.BLAKE2b(64), backend=default_backend())
            h.update(data_to_hmac)
            h.verify(hmac_value)
            print("HMAC verified successfully")
            
            # Derive decryption key
            print("Deriving decryption key...")
            public_key = serialization.load_pem_public_key(ephemeral_pub)
            shared_key = private_key.exchange(public_key)
            dec_key = VoxDecrypt.derive_keys(shared_key, salt)
            
            print(f"Shared key: {shared_key.hex()[:16]}...")
            print(f"Decryption key: {dec_key.hex()[:16]}...")
            
            # Decrypt
            print("Decrypting...")
            cipher = ChaCha20Poly1305(dec_key)
            plaintext = cipher.decrypt(nonce, ciphertext, None)
            print("Decryption successful!")
            
            # Verify the decrypted data
            if len(plaintext) == 0:
                raise ValueError("Decrypted data is empty")
            
            # Detect file type and handle appropriately
            if VoxDecrypt.is_text_data(plaintext):
                try:
                    decoded_text = plaintext.decode('utf-8')
                    print(f"Decrypted text: {decoded_text[:100]}..." if len(decoded_text) > 100 else decoded_text)
                except UnicodeDecodeError:
                    print("Decrypted data appears to be binary (UTF-8 decode failed)")
            else:
                print("Decrypted data appears to be binary")
                
                # Check for common file signatures
                file_signatures = {
                    b'\xff\xd8\xff': 'JPEG',
                    b'\x89PNG': 'PNG', 
                    b'%PDF': 'PDF',
                    b'\x50\x4B\x03': 'ZIP/Office',
                    b'\x49\x49\x2A': 'TIFF',
                    b'\x4D\x4D\x00': 'TIFF',
                    b'\x47\x49\x46': 'GIF',
                    b'\x52\x49\x46\x46': 'WEBP/AVI',
                    b'\x1A\x45\xDF\xA3': 'WebM/MKV',
                    b'\x00\x00\x00\x20': 'MP4',
                    b'\x49\x44\x33': 'MP3',
                }
                
                for sig, filetype in file_signatures.items():
                    if plaintext.startswith(sig):
                        print(f"✓ Detected file type: {filetype}")
                        break
                else:
                    print("⚠ File type not recognized")
            
            # Write output
            print("Writing output file...")
            with open(output_path, 'wb') as f:
                f.write(plaintext)
            print("✓ Decryption completed successfully")
                
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            raise ValueError(f"Static decryption failed: {str(e)}\n{error_details}")

def main():
    parser = argparse.ArgumentParser(description="VoxCrypt File Decryptor")
    parser.add_argument("-i", "--input", help="Input file (.vxc)", required=True)
    parser.add_argument("-k", "--key", help="Private key file", required=True)
    parser.add_argument("-o", "--output", help="Output file path", required=True)
    
    args = parser.parse_args()
    
    try:
        if not os.path.exists(args.input):
            print(f"[!] Input file not found: {args.input}")
            sys.exit(1)
            
        if not os.path.exists(args.key):
            print(f"[!] Key file not found: {args.key}")
            sys.exit(1)
            
        input_size = os.path.getsize(args.input)
        print(f"▓▓▓ DECRYPTING: {args.input} ({input_size} bytes) ▓▓▓")
        print(f"▓▓▓ USING KEY: {args.key} ▓▓▓")
        
        VoxDecrypt.decrypt_file(args.input, args.key, args.output)
        print(f"▓▓▓ SUCCESS: Decrypted to {args.output} ▓▓▓")
        
        # Verify the output file
        output_size = os.path.getsize(args.output)
        print(f"▓▓▓ OUTPUT SIZE: {output_size} bytes ▓▓▓")
        
    except Exception as e:
        print(f"[!] DECRYPTION FAILED: {str(e)}")
        if os.path.exists(args.output):
            try:
                os.remove(args.output)
                print("▓▓▓ Cleaned up partial output ▓▓▓")
            except:
                pass
        sys.exit(1)

if __name__ == "__main__":
    main()
