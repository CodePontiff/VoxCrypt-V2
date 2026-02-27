#!/usr/bin/env python3
"""
VoxCrypt Decryptor - Decrypts and replaces .vxc files with original content
Supports both VXC3H (standard) and VXC3S (streaming) formats
Removes .vxc extension after successful decryption
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

# ========== CONSTANTS MATCHING ENCRYPTOR EXACTLY ==========
HKDF_INFO = b"VoxCrypt v2 Key Derivation"
SALT_SIZE = 32
NONCE_SIZE = 12
HMAC_SIZE = 64
STREAM_CHUNK_SIZE = 256  # CRITICAL: Must match encryptor's STREAM_CHUNK_SIZE
STREAM_HEADER = b'VXC3S'
STANDARD_HEADER = b'VXC3H'

class VoxDecrypt:
    @staticmethod
    def derive_keys(seed, salt=None, info=None, length=32):
        """HKDF with BLAKE2b for key derivation - EXACT match to encryptor"""
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
                
                if header == STANDARD_HEADER:
                    print("Detected VXC3H format (standard encryption)")
                    return VoxDecrypt._decrypt_standard_and_rename(f, key_path, input_path)
                elif header == STREAM_HEADER:
                    print("Detected VXC3S format (streaming encryption)")
                    return VoxDecrypt._decrypt_streaming_and_rename(f, key_path, input_path)
                else:
                    raise ValueError(f"Invalid file format header: {header}")
        except Exception as e:
            raise ValueError(f"File reading failed: {str(e)}")

    @staticmethod
    def _decrypt_standard_and_rename(file_obj, key_path, input_path):
        """Decrypt standard .vxc file (VXC3H format)"""
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
            
            # Read PEM-formatted ephemeral public key
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
            
            # Derive shared key and decryption key first
            print("Deriving decryption key...")
            public_key = serialization.load_pem_public_key(ephemeral_pub)
            shared_key = private_key.exchange(public_key)
            
            # Derive decryption key
            hkdf = HKDF(
                algorithm=hashes.BLAKE2b(64),
                length=32,
                salt=salt,
                info=HKDF_INFO,
                backend=default_backend()
            )
            dec_key = hkdf.derive(shared_key)
            
            # Derive HMAC key
            hmac_key = hkdf.derive(shared_key + b"HMAC key")
            
            # Verify HMAC
            print("Verifying HMAC...")
            data_to_hmac = ephemeral_pub + nonce + ciphertext
            h = hmac.HMAC(hmac_key, hashes.BLAKE2b(64), backend=default_backend())
            h.update(data_to_hmac)
            h.verify(hmac_value)
            print("HMAC verified successfully")
            
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
                output_path = input_path[:-4]
            else:
                output_path = input_path + ".decrypted"
            
            # Write decrypted content to output file
            print(f"Writing decrypted content to: {output_path}")
            temp_path = output_path + ".temp"
            with open(temp_path, 'wb') as f:
                f.write(plaintext)
            
            # Verify the temp file
            if os.path.getsize(temp_path) == len(plaintext):
                os.replace(temp_path, output_path)
                print("✓ Decrypted file created successfully")
                
                # Remove the original .vxc file
                if input_path != output_path and os.path.exists(input_path):
                    os.remove(input_path)
                    print(f"✓ Removed encrypted file: {input_path}")
            else:
                raise ValueError("Temporary file size mismatch")
                
        except Exception as e:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            raise ValueError(f"Standard decryption failed: {str(e)}")

    @staticmethod
    def _decrypt_streaming_and_rename(file_obj, key_path, input_path):
        """Decrypt streaming .vxc file (VXC3S format) - FIXED nonce handling"""
        temp_path = None
        try:
            print("Reading streaming file components...")
            
            # Read salt (32 bytes)
            salt = file_obj.read(SALT_SIZE)
            if len(salt) != SALT_SIZE:
                raise ValueError(f"Invalid salt size: {len(salt)} bytes")
            print(f"Salt: {salt.hex()[:16]}...")
            
            # Read PEM-formatted ephemeral public key
            print("Reading ephemeral public key (PEM format)...")
            ephemeral_pub = VoxDecrypt.read_pem_object(file_obj, b'-----BEGIN PUBLIC KEY-----')
            print(f"Ephemeral pub length: {len(ephemeral_pub)} bytes")
            
            # Read initial nonce
            initial_nonce = file_obj.read(NONCE_SIZE)
            if len(initial_nonce) != NONCE_SIZE:
                raise ValueError(f"Invalid initial nonce size: {len(initial_nonce)} bytes")
            print(f"Initial nonce: {initial_nonce.hex()}")
            
            # Read remaining data as streaming ciphertext
            streaming_ciphertext = file_obj.read()
            if len(streaming_ciphertext) == 0:
                raise ValueError("Streaming ciphertext is empty")
            print(f"Streaming ciphertext: {len(streaming_ciphertext)} bytes")
            
            # Load private key
            print("Loading private key...")
            private_key = VoxDecrypt.load_private_key(key_path)
            print("Private key loaded successfully")
            
            # Derive decryption key
            print("Deriving decryption key...")
            public_key = serialization.load_pem_public_key(ephemeral_pub)
            shared_key = private_key.exchange(public_key)
            
            hkdf = HKDF(
                algorithm=hashes.BLAKE2b(64),
                length=32,
                salt=salt,
                info=b"VoxCrypt Real-time Stream",
                backend=default_backend()
            )
            dec_key = hkdf.derive(shared_key)
            
            print(f"Shared key: {shared_key.hex()[:16]}...")
            print(f"Decryption key: {dec_key.hex()[:16]}...")
            
            # Setup cipher
            cipher = ChaCha20Poly1305(dec_key)
            
            # Create output filename without .vxc extension
            if input_path.lower().endswith('.vxc'):
                output_path = input_path[:-4]
            else:
                output_path = input_path + ".decrypted"
            
            # Prepare for streaming decryption
            temp_path = output_path + ".temp"
            with open(temp_path, 'wb') as out_file:
                # Process ciphertext in chunks
                chunk_index = 0
                processed_bytes = 0
                
                # Each encrypted chunk is STREAM_CHUNK_SIZE + 16 (Poly1305 tag)
                encrypted_chunk_size = STREAM_CHUNK_SIZE + 16
                i = 0
                
                # We need to try all possible nonce variations because of audio entropy
                # The audio entropy is 8 bytes, so there are 2^64 possibilities - impossible to brute force
                # But in practice, the audio entropy is hashed from microphone input, so we can't know it
                # However, the encryption should still work because we're using the same initial nonce
                # and chunk counter. The audio entropy just adds an extra XOR layer, but since we can't
                # know it, we assume it was 0 for nonce calculation. This is a limitation of the design.
                
                print("  [NOTE] Audio entropy in nonce is ignored during decryption")
                print("  [NOTE] This is a limitation of the streaming format")
                
                while i < len(streaming_ciphertext):
                    # Calculate current chunk size
                    remaining = len(streaming_ciphertext) - i
                    current_encrypted_size = min(encrypted_chunk_size, remaining)
                    
                    # Get next chunk
                    ciphertext_chunk = streaming_ciphertext[i:i + current_encrypted_size]
                    
                    # Generate nonce for this chunk - ONLY XOR WITH CHUNK COUNTER
                    # We CANNOT know the audio entropy, so we assume it was 0
                    nonce = bytearray(initial_nonce)
                    
                    # XOR with chunk counter (first 4 bytes, little-endian)
                    for j in range(min(len(nonce), 4)):
                        nonce[j] ^= ((chunk_index >> (j * 8)) & 0xFF)
                    
                    # Debug info for first few chunks
                    if chunk_index < 10:
                        print(f"  Chunk {chunk_index}: offset={i}, size={current_encrypted_size}, nonce={bytes(nonce).hex()}")
                    
                    try:
                        # Decrypt chunk
                        plaintext_chunk = cipher.decrypt(bytes(nonce), ciphertext_chunk, None)
                    except Exception as e:
                        # If decryption fails, try with audio entropy variations? 
                        # That's not feasible - 2^64 possibilities
                        print(f"  [!] Decryption failed for chunk {chunk_index}: {str(e)}")
                        print(f"  Nonce: {bytes(nonce).hex()}, Ciphertext size: {len(ciphertext_chunk)}")
                        print(f"  Ciphertext (first 16 bytes): {ciphertext_chunk[:16].hex()}")
                        print(f"  [ERROR] Cannot decrypt without knowing audio entropy values")
                        raise ValueError("Streaming decryption requires audio entropy values which are not stored in the file. This is a design limitation.")
                    
                    # Write decrypted chunk to output
                    out_file.write(plaintext_chunk)
                    
                    processed_bytes += len(plaintext_chunk)
                    chunk_index += 1
                    i += current_encrypted_size
                    
                    # Progress update
                    if chunk_index % 50 == 0:
                        print(f"Progress: {chunk_index} chunks - {processed_bytes} bytes")
            
            print("Streaming decryption successful!")
            print(f"Total chunks processed: {chunk_index}")
            print(f"Total bytes decrypted: {processed_bytes}")
            
            # Verify the temp file
            if os.path.getsize(temp_path) == processed_bytes:
                os.replace(temp_path, output_path)
                print(f"✓ Decrypted file created: {output_path} ({processed_bytes} bytes)")
                
                # Remove the original .vxc file
                if input_path != output_path and os.path.exists(input_path):
                    os.remove(input_path)
                    print(f"✓ Removed encrypted file: {input_path}")
            else:
                raise ValueError(f"Temporary file size mismatch: {os.path.getsize(temp_path)} != {processed_bytes}")
                
        except Exception as e:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            raise ValueError(f"Streaming decryption failed: {str(e)}")

    @staticmethod
    def detect_file_format(input_path):
        """Detect the format of a .vxc file without fully processing it"""
        try:
            with open(input_path, 'rb') as f:
                header = f.read(5)
                return header
        except:
            return None

def main():
    parser = argparse.ArgumentParser(
        description="VoxCrypt File Decryptor - Supports both standard (VXC3H) and streaming (VXC3S) formats",
        epilog="Examples:\n"
               "  Decrypt any .vxc file: voxdecrypt -i file.vxc -k private.key\n"
               "  Auto-detect format: voxdecrypt -i encrypted.vxc -k key.pem"
    )
    
    parser.add_argument("-i", "--input", help="Input .vxc file to decrypt", required=True)
    parser.add_argument("-k", "--key", help="Private key file", required=True)
    parser.add_argument("-o", "--output", help="Output file path (optional, defaults to input without .vxc)")
    parser.add_argument("--debug", help="Enable debug output for troubleshooting", action="store_true")
    
    args = parser.parse_args()
    
    try:
        if not os.path.exists(args.input):
            print(f"[!] Input file not found: {args.input}")
            sys.exit(1)
            
        if not os.path.exists(args.key):
            print(f"[!] Key file not found: {args.key}")
            sys.exit(1)
        
        # Detect format
        print("▓▓▓ ANALYZING ENCRYPTED FILE ▓▓▓")
        format_header = VoxDecrypt.detect_file_format(args.input)
        
        if format_header == STANDARD_HEADER:
            print("▓▓▓ FORMAT: VXC3H (Standard Encryption) ▓▓▓")
        elif format_header == STREAM_HEADER:
            print("▓▓▓ FORMAT: VXC3S (Streaming/Real-time Encryption) ▓▓▓")
            print(f"▓▓▓ CHUNK SIZE: {STREAM_CHUNK_SIZE} bytes ▓▓▓")
        else:
            print(f"[!] WARNING: Unknown format header: {format_header}")
            sys.exit(1)
        
        input_size = os.path.getsize(args.input)
        print(f"▓▓▓ INPUT: {args.input} ({input_size} bytes) ▓▓▓")
        print(f"▓▓▓ USING KEY: {args.key} ▓▓▓")
        
        # Determine output filename
        if args.output:
            output_path = args.output
        else:
            if args.input.lower().endswith('.vxc'):
                output_path = args.input[:-4]
            else:
                output_path = args.input + ".decrypted"
        
        # Check if output file already exists
        if os.path.exists(output_path):
            print(f"[!] Warning: Output file already exists: {output_path}")
            print("    It will be overwritten.")
        
        # Perform decryption
        print("\n▓▓▓ STARTING DECRYPTION ▓▓▓")
        VoxDecrypt.decrypt_and_rename(args.input, args.key)
        
        # Verify output
        if os.path.exists(output_path):
            restored_size = os.path.getsize(output_path)
            print(f"▓▓▓ OUTPUT SIZE: {restored_size} bytes ▓▓▓")
            print(f"▓▓▓ SUCCESS: Decrypted file created: {output_path} ▓▓▓")
        
    except Exception as e:
        print(f"[!] DECRYPTION FAILED: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
