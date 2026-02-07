#!/usr/bin/env python3
"""
VoxCrypt Decryptor - Decrypts and replaces .vxc files with original content
Removes .vxc extension after successful decryption
"""

import os
import sys
import argparse
import shutil
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
    def decrypt_and_rename(input_path, key_path):
        """Decrypt .vxc file and rename to original name (without .vxc)"""
        try:
            with open(input_path, 'rb') as f:
                header = f.read(5)
                
                if header == b'VXC3H':
                    print("Detected VXC3H format (static file)")
                    return VoxDecrypt._decrypt_static_and_rename(f, key_path, input_path)
                else:
                    raise ValueError(f"Invalid file format header: {header}")
        except Exception as e:
            raise ValueError(f"File reading failed: {str(e)}")

    @staticmethod
    def _decrypt_static_and_rename(file_obj, key_path, input_path):
        """Decrypt static .vxc file and rename to original name"""
        temp_path = None
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
            
            if len(plaintext) == 0:
                raise ValueError("Decrypted data is empty")
            
            # Create output filename without .vxc extension
            if input_path.lower().endswith('.vxc'):
                output_path = input_path[:-4]  # Remove .vxc
            else:
                output_path = input_path + ".decrypted"
            
            # Write decrypted content to output file
            print(f"Writing decrypted content to: {output_path}")
            temp_path = output_path + ".temp"
            with open(temp_path, 'wb') as f:
                f.write(plaintext)
            
            # Verify the temp file
            if os.path.getsize(temp_path) == len(plaintext):
                # Rename temp file to final output
                os.replace(temp_path, output_path)
                print("✓ Decrypted file created successfully")
                
                # Remove the original .vxc file
                if input_path != output_path and os.path.exists(input_path):
                    os.remove(input_path)
                    print(f"✓ Removed encrypted file: {input_path}")
            else:
                raise ValueError("Temporary file size mismatch")
                
        except Exception as e:
            # Clean up temp file if it exists
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            raise ValueError(f"Decryption failed: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="VoxCrypt File Decryptor - Creates decrypted file without .vxc extension")
    parser.add_argument("-i", "--input", help="Input .vxc file to decrypt", required=True)
    parser.add_argument("-k", "--key", help="Private key file", required=True)
    parser.add_argument("-o", "--output", help="Output file path (optional, defaults to input without .vxc)")
    
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
        
        # Determine output filename
        if args.output:
            output_path = args.output
        else:
            # Remove .vxc extension if present
            if args.input.lower().endswith('.vxc'):
                output_path = args.input[:-4]
            else:
                output_path = args.input + ".decrypted"
        
        # Check if output file already exists
        if os.path.exists(output_path):
            print(f"[!] Warning: Output file already exists: {output_path}")
            print("    It will be overwritten.")
        
        VoxDecrypt.decrypt_and_rename(args.input, args.key)
        
        # Verify output
        if os.path.exists(output_path):
            restored_size = os.path.getsize(output_path)
            print(f"▓▓▓ OUTPUT SIZE: {restored_size} bytes ▓▓▓")
            print(f"▓▓▓ SUCCESS: Decrypted file created: {output_path} ▓▓▓")
        else:
            print("[!] ERROR: Output file was not created")
            sys.exit(1)
        
    except Exception as e:
        print(f"[!] DECRYPTION FAILED: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
