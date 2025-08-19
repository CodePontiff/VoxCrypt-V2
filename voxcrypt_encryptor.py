#!/usr/bin/env python3
"""
VoxCrypt Ultimate - Secure Audio-Based Encryption Tool
Fixed version with proper public key handling
"""

import os
import sys
import time
import argparse
import hashlib
import numpy as np
import sounddevice as sd
import select
from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation

# ========== CONSTANTS ==========
SAMPLE_RATE = 44100
CHUNK_SIZE = 1024
DISPLAY_LEN = 1024
SALT_SIZE = 32
HKDF_INFO = b"VoxCrypt v2 Key Derivation"
MIN_VOICE_THRESHOLD = 0.01
NONCE_SIZE = 12
CYBER_COLORS = {
    'pink': '#ff00ff',
    'blue': '#00ffff',
    'purple': '#9d00ff',
    'cyan': '#00ffcc'
}

# ========== GLOBAL STATE ==========
latest_audio_frame = np.zeros(DISPLAY_LEN, dtype=np.float32)
encryption_active = False
current_salt = None
visualization_enabled = True
user_finalized = False
dot_animation_state = 0
fig = None
ax = None

# ========== CRYPTO CORE ==========
class AudioCrypto:
    @staticmethod
    def generate_audio_seed(audio_samples):
        """Generate cryptographic seed from audio with OS entropy mixing"""
        audio_bytes = audio_samples.tobytes()
        os_entropy = os.urandom(32)
        return hashlib.blake2b(audio_bytes + os_entropy).digest()

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
    def generate_key_pair(seed):
        """X25519 key pair from audio seed"""
        private_key = x25519.X25519PrivateKey.from_private_bytes(
            AudioCrypto.derive_keys(seed, info=b"Key generation salt")
        )
        return private_key, private_key.public_key()

    @staticmethod
    def encrypt_data(data, public_key_pem, salt):
        """ChaCha20Poly1305 encryption with ephemeral key pair - FIXED"""
        ephemeral_private = x25519.X25519PrivateKey.generate()
        
        # Load the public key from PEM bytes (FIXED)
        public_key = serialization.load_pem_public_key(public_key_pem)
        
        shared_key = ephemeral_private.exchange(public_key)
        enc_key = AudioCrypto.derive_keys(shared_key, salt)
        
        cipher = ChaCha20Poly1305(enc_key)
        nonce = os.urandom(NONCE_SIZE)
        ciphertext = cipher.encrypt(nonce, data, None)

        return {
            'ephemeral_pub': ephemeral_private.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ),
            'nonce': nonce,
            'ciphertext': ciphertext
        }

# ========== AUDIO HANDLING ==========
class AudioHandler:
    @staticmethod
    def record_until_enter():
        """Record audio sample until user presses Enter"""
        print("\n▓▓▓ PRESS ENTER TO BEGIN AUDIO CAPTURE ▓▓▓")
        input()
        
        print("»» SPEAK NOW - PRESS ENTER WHEN DONE ««")
        audio_chunks = []
        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16', blocksize=CHUNK_SIZE)
        stream.start()
        
        try:
            while True:
                data, _ = stream.read(CHUNK_SIZE)
                audio_chunks.append(data.copy().flatten())
                
                # Check for Enter key press
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    _ = sys.stdin.readline()
                    break
        finally:
            stream.stop()
            
        return np.concatenate(audio_chunks).astype('int16') if audio_chunks else np.array([], dtype='int16')

    @staticmethod
    def live_audio_callback(indata, frames, time_info, status):
        """Live audio callback for visualization and salt generation"""
        global latest_audio_frame, current_salt
        
        # Update visualization frame
        latest_audio_frame = np.roll(latest_audio_frame, -len(indata))
        latest_audio_frame[-len(indata):] = indata[:, 0]
        
        # Generate salt only when voice is detected
        rms = np.sqrt(np.mean(indata.astype(np.float32)**2))
        if rms > MIN_VOICE_THRESHOLD:
            current_salt = hashlib.blake2b(indata.tobytes()).digest()

# ========== FILE OPERATIONS ==========
class SecureFile:
    @staticmethod
    def encrypt_file(input_path, output_path, public_key_pem):
        """Encrypt file with authenticated format - FIXED"""
        with open(input_path, 'rb') as f:
            data = f.read()
        
        salt = os.urandom(SALT_SIZE)
        encrypted = AudioCrypto.encrypt_data(data, public_key_pem, salt)  # Pass PEM bytes
        
        hmac_key = AudioCrypto.derive_keys(salt, info=b"HMAC key")
        h = hmac.HMAC(hmac_key, hashes.BLAKE2b(64), backend=default_backend())

        data_to_hmac = encrypted['ephemeral_pub'] + encrypted['nonce'] + encrypted['ciphertext']
        h.update(data_to_hmac)
        
        with open(output_path, 'wb') as f:
            f.write(b'VXC3H')  # Format marker
            f.write(salt)
            f.write(h.finalize())
            f.write(encrypted['ephemeral_pub'])
            f.write(encrypted['nonce'])
            f.write(encrypted['ciphertext'])

    @staticmethod
    def encrypt_file_from_data(data, output_path, public_key_pem):
        """Encrypt raw data with authenticated format (for text input) - FIXED"""
        salt = os.urandom(SALT_SIZE)
        encrypted = AudioCrypto.encrypt_data(data, public_key_pem, salt)  # Pass PEM bytes
        
        hmac_key = AudioCrypto.derive_keys(salt, info=b"HMAC key")
        h = hmac.HMAC(hmac_key, hashes.BLAKE2b(64), backend=default_backend())

        data_to_hmac = encrypted['ephemeral_pub'] + encrypted['nonce'] + encrypted['ciphertext']
        h.update(data_to_hmac)
        
        with open(output_path, 'wb') as f:
            f.write(b'VXC3H')  # Format marker
            f.write(salt)
            f.write(h.finalize())
            f.write(encrypted['ephemeral_pub'])
            f.write(encrypted['nonce'])
            f.write(encrypted['ciphertext'])

# ========== VISUALIZATION ==========
def setup_cyberpunk_display():
    """Initialize cyberpunk-style plot"""
    global fig, ax
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor('#0a0a12')
    ax.set_facecolor('#0a0a12')
    ax.grid(color='#1a1a2a55', linestyle='--', alpha=0.7)
    ax.tick_params(colors='#e0e0ff')
    
    for spine in ax.spines.values():
        spine.set_edgecolor(CYBER_COLORS['purple'])
        spine.set_linewidth(1.5)
    
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(-1.5, 1.5)
    
    info_text = ax.text(
        0.02, 0.95, "INITIALIZING...", transform=ax.transAxes,
        fontsize=10, fontfamily='monospace', color=CYBER_COLORS['cyan'],
        verticalalignment='top'
    )
    
    fig.suptitle(
        "»» VOXCRYPT ENCRYPTOR TERMINAL ««",
        color=CYBER_COLORS['pink'],
        fontsize=14,
        fontweight='bold',
        fontfamily='monospace'
    )
    
    return fig, ax, info_text

def update_visual(frame):
    """Update visualization with audio data"""
    global dot_animation_state, user_finalized
    
    # Animation state
    dot_animation_state = (dot_animation_state + 1) % 4
    
    # Check for Enter key press
    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        _ = sys.stdin.readline()
        user_finalized = True
        return []
    
    # Prepare visualization data
    x = np.linspace(0, 2 * np.pi, DISPLAY_LEN)
    y = latest_audio_frame * 0.8
    
    # Create segmented color wave
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    if hasattr(update_visual, 'segments'):
        update_visual.segments.remove()
    
    colors = []
    for val in y:
        if val > 0.8: colors.append('#ff00ff')
        elif val > 0.5: colors.append('#ff00a2')
        elif val > 0.2: colors.append('#9d00ff')
        elif val > -0.2: colors.append('#00ffff')
        elif val > -0.5: colors.append('#00ffcc')
        else: colors.append('#00a2ff')
    
    update_visual.segments = LineCollection(
        segments,
        colors=colors,
        linewidths=3,
        alpha=0.9,
        linestyle='-',
        antialiased=True
    )
    ax.add_collection(update_visual.segments)
    
    # Update status text
    status = [
        "▓▓▓ VOXCRYPT ENCRYPTION PROTOCOL ▓▓▓",
        f"» STATUS: {'RECORDING' + '.' * dot_animation_state if encryption_active else 'READY'}",
        f"» SALT: {current_salt.hex()[:8] + '...' if current_salt else 'NO VOICE'}",
        f"» MODE: {'STREAM' if args.stream else 'FILE'}",
        "▓▓▓ PRESS ENTER TO FINALIZE ▓▓▓"
    ]
    
    update_visual.info_text.set_text("\n".join(status))
    
    return [update_visual.segments, update_visual.info_text]

# ========== MAIN APPLICATION ==========
def main():
    global encryption_active, user_finalized, args
    
    parser = argparse.ArgumentParser(
        description="VoxCrypt Ultimate - Audio-Based Secure Encryption",
        epilog="Examples:\n"
               "  Encrypt file: voxcrypt -I secret.doc -k key.pem\n"
               "  Encrypt stream: voxcrypt --stream data.log -k stream.pem"
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-i", "--input", help="Text to encrypt")
    group.add_argument("-I", "--input-file", help="File to encrypt")
    group.add_argument("--stream", help="Stream source to encrypt")
    
    parser.add_argument("-k", "--key", help="Output key file", required=True)
    parser.add_argument("--replace", help="Replace original file", action="store_true")
    parser.add_argument("--no-visual", help="Disable visualization", action="store_true")
    
    args = parser.parse_args()
    
    try:
        # Initialize
        encryption_active = True
        visualization_enabled = not args.no_visual
        
        # Audio seed generation
        audio_samples = AudioHandler.record_until_enter()
        if audio_samples.size == 0:
            print("[!] ERROR: No audio captured")
            sys.exit(1)
            
        seed = AudioCrypto.generate_audio_seed(audio_samples)
        
        # Key generation
        private_key, public_key = AudioCrypto.generate_key_pair(seed)
        pub_key_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        # Save key
        private_key_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        with open(args.key, 'wb') as f:
            f.write(private_key_pem)
        print(f"\n▓▓▓ ENCRYPTION KEY SAVED TO {args.key} ▓▓▓")
        
        # Setup visualization
        if visualization_enabled:
            global fig, ax
            fig, ax, update_visual.info_text = setup_cyberpunk_display()
            
            # Start audio stream for visualization and salt
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE, 
                channels=1, 
                dtype='int16',
                blocksize=CHUNK_SIZE, 
                callback=AudioHandler.live_audio_callback
            )
            stream.start()
            
            # Start animation
            ani = FuncAnimation(
                fig, 
                update_visual, 
                interval=50, 
                blit=True,
                cache_frame_data=False
            )
            
            print("\n▓▓▓ LIVE ENCRYPTION ACTIVE - PRESS ENTER TO FINALIZE ▓▓▓")
            plt.show()
            
            # Cleanup
            stream.stop()
            stream.close()
        else:
            # Non-visual mode
            print("\n▓▓▓ ENCRYPTION IN PROGRESS ▓▓▓")
            print("▓▓▓ PRESS ENTER TO FINALIZE ▓▓▓")
            input()
            user_finalized = True
        
        # Perform encryption
        if args.stream:
            output_path = f"{args.stream}.vxcs"
            print(f"\n▓▓▓ ENCRYPTING STREAM: {args.stream} ▓▓▓")
            # Note: You'll need to fix the encrypt_stream method too
            # SecureFile.encrypt_stream(args.stream, output_path, pub_key_pem)
            print("Stream encryption not implemented in this fix")
        else:
            output_path = f"{args.input_file if args.input_file else 'message'}.vxc"
            
            if args.input:
                data = args.input.encode()
                SecureFile.encrypt_file_from_data(data, output_path, pub_key_pem)
            else:
                SecureFile.encrypt_file(args.input_file, output_path, pub_key_pem)
            
            if args.replace and args.input_file:
                os.remove(args.input_file)
                
        print(f"\n▓▓▓ ENCRYPTION COMPLETE ▓▓▓")
        print(f"» Output: {output_path}")
        
    except Exception as e:
        print(f"[!] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        encryption_active = False

if __name__ == "__main__":
    main()