#!/usr/bin/env python3
"""
VoxCrypt-V2 Encryptor - Secure Audio-Based Encryption Tool with Real-time Streaming
FIXED: Removed audio entropy from nonce to make streaming mode decryptable
ADDED: --replace-original with immediate file replacement during audio capture
"""

import os
import sys
import time
import argparse
import hashlib
import threading
import queue
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

# ========== CYBERPUNK VISUAL CONFIG ==========
CYBER_COLORS = {
    'pink': '#ff00ff',
    'blue': '#00ffff',
    'purple': '#9d00ff',
    'cyan': '#00ffcc',
    'yellow': '#fff000'
}

WAVE_GRADIENT = [
    '#ff00ff', '#ff00a2', '#9d00ff',
    '#00ffff', '#00ffcc', '#00a2ff'
]

BG_COLOR = '#0a0a12'
GRID_COLOR = '#1a1a2a55'
TEXT_COLOR = '#e0e0ff'
GLOW_ALPHA = 0.15

# ========== CRYPTO CONFIG ==========
SAMPLE_RATE = 48000
SEED_CHUNK = 4096
LIVE_CHUNK = 1024
DISPLAY_LEN = 1024
SALT_SIZE = 32
HKDF_INFO = b"VoxCrypt v2 Key Derivation"
NONCE_SIZE = 12
SMOOTHING = 4

# ========== STREAMING CONFIG ==========
STREAM_CHUNK_SIZE = 256  # Small chunks for frequent updates
STREAM_ENCRYPTION_INTERVAL = 0.02  # 20ms between chunks
STREAM_READ_BUFFER = 1024  # For file reading

# ========== SENSITIVITY CONFIG ==========
SENSITIVITY_LEVELS = {
    1: {  # Very Low
        'min_rms': 0.01,
        'min_peak': 0.05,
        'record_boost': 2,
        'live_boost': 1.5,
        'visual_boost': 1,
        'description': 'VERY LOW - For loud environments'
    },
    2: {  # Low
        'min_rms': 0.005,
        'min_peak': 0.025,
        'record_boost': 4,
        'live_boost': 3,
        'visual_boost': 2,
        'description': 'LOW - For normal speaking volume'
    },
    3: {  # Medium (Default)
        'min_rms': 0.001,
        'min_peak': 0.005,
        'record_boost': 8,
        'live_boost': 6,
        'visual_boost': 4,
        'description': 'MEDIUM - For quiet speaking'
    },
    4: {  # High
        'min_rms': 0.0005,
        'min_peak': 0.0025,
        'record_boost': 16,
        'live_boost': 12,
        'visual_boost': 8,
        'description': 'HIGH - For whispers'
    },
    5: {  # Very High
        'min_rms': 0.0001,
        'min_peak': 0.0005,
        'record_boost': 32,
        'live_boost': 24,
        'visual_boost': 16,
        'description': 'VERY HIGH - For extremely quiet sounds'
    }
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
verbose = False
current_salt_src = "(none)"
encryption_done = False
base_name = "message"
should_exit = False
audio_stream = None
entropy_collected = 0
entropy_bits = 0
sensitivity_level = 3

# ========== STREAMING STATE ==========
audio_entropy_buffer = queue.Queue(maxsize=1000)
current_audio_entropy = b""
stream_is_running = False
stream_total_encrypted = 0
stream_chunk_counter = 0
stream_output_path = None
stop_encryption_flag = threading.Event()
original_file_removed = False  # Flag untuk menandai file original sudah dihapus

# ========== REAL-TIME ENCRYPTION CLASS ==========
class RealTimeStreamEncryptor:
    def __init__(self, public_key_pem, salt, output_path, input_data=None, original_file=None):
        self.public_key_pem = public_key_pem
        self.salt = salt
        self.output_path = output_path
        self.input_data = input_data  # None for continuous mode, bytes for file/text mode
        self.original_file = original_file  # Path ke file original untuk dihapus
        self.running = False
        self.thread = None
        self.chunk_counter = 0
        self.total_bytes = 0
        self.is_continuous = (input_data is None)
        self.original_removed = False
        
        # Setup encryption
        self.setup_encryption()
    
    def setup_encryption(self):
        """Setup encryption for real-time streaming"""
        print("[STREAM] Setting up real-time encryption...")
        
        # Generate ephemeral key pair
        self.ephemeral_private = x25519.X25519PrivateKey.generate()
        
        # Load public key
        public_key = serialization.load_pem_public_key(self.public_key_pem)
        
        # Generate shared key
        shared_key = self.ephemeral_private.exchange(public_key)
        
        # Derive encryption key
        hkdf = HKDF(
            algorithm=hashes.BLAKE2b(64),
            length=32,
            salt=self.salt,
            info=b"VoxCrypt Real-time Stream",
            backend=default_backend()
        )
        self.enc_key = hkdf.derive(shared_key)
        
        # Create cipher
        self.cipher = ChaCha20Poly1305(self.enc_key)
        
        # Initial nonce
        self.initial_nonce = os.urandom(NONCE_SIZE)
        
        # Write header to file
        with open(self.output_path, 'wb') as f:
            f.write(b'VXC3S')  # Streaming format marker
            f.write(self.salt)
            f.write(self.ephemeral_private.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ))
            f.write(self.initial_nonce)
        
        print(f"[STREAM] Header written to {self.output_path}")
    
    def generate_continuous_chunk(self):
        """Generate pseudo-random data for continuous encryption"""
        timestamp = int(time.time() * 1000000)
        random_data = os.urandom(8)
        
        # Get latest audio entropy (for data generation only, not for nonce)
        audio_entropy = b""
        try:
            audio_entropy = audio_entropy_buffer.get_nowait()
        except queue.Empty:
            pass
        
        # Generate unique data using audio entropy (this is fine - data can be random)
        seed = str(timestamp).encode() + random_data + audio_entropy
        seed_hash = hashlib.sha256(seed).digest()
        
        chunk_data = bytearray(STREAM_CHUNK_SIZE)
        for i in range(STREAM_CHUNK_SIZE):
            chunk_data[i] = (
                seed_hash[i % 32] + 
                (timestamp >> (i % 24)) + 
                (len(audio_entropy) if audio_entropy else 0)
            ) % 256
        
        return bytes(chunk_data)
    
    def get_file_chunk(self):
        """Get next chunk from input file/data"""
        if self.input_data is None:
            return None
        
        # Calculate position
        start_pos = self.chunk_counter * STREAM_CHUNK_SIZE
        if start_pos >= len(self.input_data):
            return None  # End of data
        
        # Get chunk
        end_pos = min(start_pos + STREAM_CHUNK_SIZE, len(self.input_data))
        chunk = self.input_data[start_pos:end_pos]
        
        # Pad if needed
        if len(chunk) < STREAM_CHUNK_SIZE:
            chunk = chunk + os.urandom(STREAM_CHUNK_SIZE - len(chunk))
        
        return chunk
    
    def encrypt_and_write_chunk(self):
        """Encrypt and write one chunk to file"""
        # Get data for this chunk
        if self.is_continuous:
            chunk_data = self.generate_continuous_chunk()
        else:
            chunk_data = self.get_file_chunk()
            if chunk_data is None:
                return None  # End of file
        
        # Generate nonce for this chunk - ONLY use chunk counter, NO audio entropy
        # This ensures the nonce can be reproduced during decryption
        nonce = bytearray(self.initial_nonce)
        
        # XOR with chunk counter (first 4 bytes, little-endian)
        for i in range(min(len(nonce), 4)):
            nonce[i] ^= ((self.chunk_counter >> (i * 8)) & 0xFF)
        
        # IMPORTANT: Audio entropy is NOT used in nonce generation
        # This makes streaming mode decryptable without storing entropy values
        
        # Encrypt chunk
        encrypted_chunk = self.cipher.encrypt(bytes(nonce), chunk_data, None)
        
        # Append to file
        with open(self.output_path, 'ab') as f:
            f.write(encrypted_chunk)
        
        self.chunk_counter += 1
        self.total_bytes += len(encrypted_chunk)
        
        return len(encrypted_chunk)
    
    def run_encryption(self):
        """Run encryption until stopped or data exhausted"""
        global stream_is_running, stream_total_encrypted, stream_chunk_counter, original_file_removed
        
        self.running = True
        stream_is_running = True
        
        print(f"[STREAM] Starting {'continuous' if self.is_continuous else 'file'} encryption...")
        print(f"[STREAM] Output: {self.output_path}")
        print(f"[STREAM] Mode: {'CONTINUOUS (runs forever)' if self.is_continuous else f'FILE ({len(self.input_data)} bytes)'}")
        print(f"[STREAM] Nonce mode: Counter-based only (decryptable)")
        
        try:
            while self.running and not stop_encryption_flag.is_set():
                start_time = time.time()
                
                # Encrypt and write one chunk
                result = self.encrypt_and_write_chunk()
                
                if result is None:  # End of file data
                    print("[STREAM] File encryption complete")
                    break
                
                chunk_size = result
                
                stream_total_encrypted = self.total_bytes
                stream_chunk_counter = self.chunk_counter
                
                # HAPUS FILE ORIGINAL SEGERA SETELAH CHUNK PERTAMA DITULIS
                # Ini memastikan file langsung diganti saat enkripsi dimulai
                if not self.original_removed and self.original_file and self.chunk_counter >= 1:
                    try:
                        if os.path.exists(self.original_file):
                            os.remove(self.original_file)
                            self.original_removed = True
                            original_file_removed = True
                            print(f"\n[!] ORIGINAL FILE REPLACED: {self.original_file} -> {self.output_path}")
                            print(f"    (Original file deleted after first encrypted chunk)")
                    except Exception as e:
                        print(f"\n[!] WARNING: Could not remove original file: {str(e)}")
                
                # Verbose output
                if verbose and self.chunk_counter % 50 == 0:
                    file_size = os.path.getsize(self.output_path)
                    print(f"[STREAM] Chunk #{self.chunk_counter}: {file_size:,} bytes total")
                
                # Calculate sleep time
                elapsed = time.time() - start_time
                sleep_time = max(0, STREAM_ENCRYPTION_INTERVAL - elapsed)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            print("\n[STREAM] Encryption interrupted by user")
        except Exception as e:
            print(f"[STREAM] Error: {str(e)}")
        finally:
            self.running = False
            stream_is_running = False
            print(f"\n[STREAM] Encryption stopped")
            print(f"[STREAM] Final stats:")
            print(f"[STREAM]   Total chunks: {self.chunk_counter}")
            print(f"[STREAM]   Total bytes: {self.total_bytes:,}")
            if os.path.exists(self.output_path):
                print(f"[STREAM]   File size: {os.path.getsize(self.output_path):,} bytes")
    
    def stop(self):
        """Stop encryption"""
        self.running = False
        # Don't try to join our own thread - just set the flag
        stop_encryption_flag.set()
    
    def start(self):
        """Start encryption in background thread"""
        self.thread = threading.Thread(target=self.run_encryption, daemon=True)
        self.thread.start()

# ========== AUDIO HANDLING ==========
class AudioHandler:
    @staticmethod
    def record_until_enter():
        """Record audio sample until user presses Enter"""
        print("\n▓▓▓ PRESS ENTER TO BEGIN AUDIO CAPTURE ▓▓▓")
        input()
        
        print(f"»» SPEAK NOW - PRESS ENTER WHEN DONE ««")
        print(f"»» SENSITIVITY LEVEL: {sensitivity_level} ({SENSITIVITY_LEVELS[sensitivity_level]['description']}) ««")
        audio_chunks = []
        
        boost = SENSITIVITY_LEVELS[sensitivity_level]['record_boost']
        
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE, 
            channels=1, 
            dtype='int16', 
            blocksize=SEED_CHUNK,
            latency='low'
        )
        stream.start()
        
        try:
            if verbose:
                print(f"[VERBOSE] Audio capture started...")
            while True:
                data, _ = stream.read(SEED_CHUNK)
                amplified_data = data * boost
                audio_chunks.append(amplified_data.copy().flatten())
                
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    _ = sys.stdin.readline()
                    if verbose:
                        print("[VERBOSE] Enter pressed, stopping audio capture...")
                    break
        finally:
            stream.stop()
            
        audio_data = np.concatenate(audio_chunks).astype('int16') if audio_chunks else np.array([], dtype='int16')
        if verbose:
            print(f"[VERBOSE] Audio captured: {len(audio_data)} samples")
        return audio_data

    @staticmethod
    def frame_has_voice(frame_i16):
        """Detect if audio frame contains any sound"""
        if frame_i16.size == 0:
            return False
        
        min_rms = SENSITIVITY_LEVELS[sensitivity_level]['min_rms']
        min_peak = SENSITIVITY_LEVELS[sensitivity_level]['min_peak']
        boost = SENSITIVITY_LEVELS[sensitivity_level]['live_boost']
        
        audio_float = frame_i16.astype(np.float32) / 32768.0
        amplified = audio_float * boost
        
        rms = np.sqrt(np.mean(amplified**2))
        peak = np.max(np.abs(amplified))
        
        return (rms >= min_rms) or (peak >= min_peak)

    @staticmethod
    def live_audio_callback(indata, frames, time_info, status):
        """Live audio callback for visualization and entropy collection"""
        global latest_audio_frame, current_salt, current_salt_src, entropy_collected, entropy_bits
        global current_audio_entropy
        
        boost = SENSITIVITY_LEVELS[sensitivity_level]['live_boost']
        amplified_data = indata[:, 0].astype(np.float32) * boost
        latest_audio_frame = amplified_data.astype(np.int16)
        
        if AudioHandler.frame_has_voice(latest_audio_frame):
            frame_entropy = np.log2(max(2, len(np.unique(np.round(np.diff(amplified_data), decimals=4)))))
            entropy_bits += frame_entropy
            entropy_collected += 1
            
            # Generate audio entropy for data generation (not for nonce)
            audio_bytes = latest_audio_frame.tobytes()
            audio_hash = hashlib.sha256(audio_bytes).digest()[:8]
            
            # Put in buffer for encryption thread (used only for data generation)
            try:
                audio_entropy_buffer.put_nowait(audio_hash)
                current_audio_entropy = audio_hash
            except queue.Full:
                pass
            
            current_salt = hashlib.blake2b(latest_audio_frame.tobytes()).digest()
            current_salt_src = f"mic-voice ({entropy_bits:.1f} bits)"
        else:
            current_salt = b""
            current_salt_src = "static"
            current_audio_entropy = b""

# ========== FILE OPERATIONS ==========
class SecureFile:
    @staticmethod
    def encrypt_file(input_path, output_path, public_key_pem, replace_original=False):
        """Encrypt file with authenticated format - Standard mode (non-streaming)"""
        if verbose:
            print(f"[VERBOSE] Reading input file: {input_path}")
        with open(input_path, 'rb') as f:
            data = f.read()
        
        if verbose:
            print(f"[VERBOSE] File read: {len(data)} bytes")
        
        salt = os.urandom(SALT_SIZE)
        if verbose:
            print(f"[VERBOSE] Generated salt: {salt.hex()[:8]}...")
        
        # Standard encryption
        ephemeral_private = x25519.X25519PrivateKey.generate()
        public_key = serialization.load_pem_public_key(public_key_pem)
        shared_key = ephemeral_private.exchange(public_key)
        
        hkdf = HKDF(
            algorithm=hashes.BLAKE2b(64),
            length=32,
            salt=salt,
            info=HKDF_INFO,
            backend=default_backend()
        )
        enc_key = hkdf.derive(shared_key)
        
        cipher = ChaCha20Poly1305(enc_key)
        nonce = os.urandom(NONCE_SIZE)
        ciphertext = cipher.encrypt(nonce, data, None)
        
        # HMAC for authentication
        hmac_key = hkdf.derive(shared_key + b"HMAC key")
        h = hmac.HMAC(hmac_key, hashes.BLAKE2b(64), backend=default_backend())
        data_to_hmac = ephemeral_private.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ) + nonce + ciphertext
        h.update(data_to_hmac)
        hmac_value = h.finalize()
        
        if verbose:
            print(f"[VERBOSE] HMAC computed: {hmac_value.hex()[:8]}...")
            print(f"[VERBOSE] Writing encrypted file: {output_path}")
        
        with open(output_path, 'wb') as f:
            f.write(b'VXC3H')  # Standard format marker
            f.write(salt)
            f.write(hmac_value)
            f.write(ephemeral_private.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ))
            f.write(nonce)
            f.write(ciphertext)
        
        # Handle replace-original for standard mode
        if replace_original and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            try:
                os.remove(input_path)
                print(f"\n[!] ORIGINAL FILE REPLACED: {input_path} -> {output_path}")
            except Exception as e:
                print(f"\n[!] WARNING: Could not remove original file: {str(e)}")
        
        if verbose:
            print("[VERBOSE] File encryption completed successfully")

    @staticmethod
    def encrypt_file_from_data(data, output_path, public_key_pem):
        """Encrypt raw data with authenticated format (for text input) - Standard mode"""
        if verbose:
            print(f"[VERBOSE] Encrypting text data: {len(data)} bytes")
        
        salt = os.urandom(SALT_SIZE)
        if verbose:
            print(f"[VERBOSE] Generated salt: {salt.hex()[:8]}...")
        
        # Standard encryption
        ephemeral_private = x25519.X25519PrivateKey.generate()
        public_key = serialization.load_pem_public_key(public_key_pem)
        shared_key = ephemeral_private.exchange(public_key)
        
        hkdf = HKDF(
            algorithm=hashes.BLAKE2b(64),
            length=32,
            salt=salt,
            info=HKDF_INFO,
            backend=default_backend()
        )
        enc_key = hkdf.derive(shared_key)
        
        cipher = ChaCha20Poly1305(enc_key)
        nonce = os.urandom(NONCE_SIZE)
        ciphertext = cipher.encrypt(nonce, data, None)
        
        # HMAC for authentication
        hmac_key = hkdf.derive(shared_key + b"HMAC key")
        h = hmac.HMAC(hmac_key, hashes.BLAKE2b(64), backend=default_backend())
        data_to_hmac = ephemeral_private.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ) + nonce + ciphertext
        h.update(data_to_hmac)
        hmac_value = h.finalize()
        
        if verbose:
            print(f"[VERBOSE] HMAC computed: {hmac_value.hex()[:8]}...")
            print(f"[VERBOSE] Writing encrypted file: {output_path}")
        
        with open(output_path, 'wb') as f:
            f.write(b'VXC3H')  # Standard format marker
            f.write(salt)
            f.write(hmac_value)
            f.write(ephemeral_private.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ))
            f.write(nonce)
            f.write(ciphertext)
        
        if verbose:
            print("[VERBOSE] Text encryption completed successfully")

# ========== VISUALIZATION ==========
def setup_cyberpunk_display():
    """Initialize cyberpunk-styled plot"""
    global fig, ax
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 4), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.grid(color=GRID_COLOR, linestyle='--', alpha=0.7)
    ax.tick_params(colors=TEXT_COLOR)
    
    for spine in ax.spines.values():
        spine.set_edgecolor(CYBER_COLORS['purple'])
        spine.set_linewidth(1.5)
        spine.set_alpha(0.8)
    
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(-1.5, 1.5)
    
    for y in [-1, 0, 1]:
        ax.axhline(y, color=CYBER_COLORS['blue'], linestyle=':', alpha=0.2)
    
    return fig, ax

def colorize_waveform(y_values):
    """Assign cyberpunk colors based on amplitude"""
    colors = []
    for y in y_values:
        if y > 0.8: colors.append(WAVE_GRADIENT[0])
        elif y > 0.5: colors.append(WAVE_GRADIENT[1])
        elif y > 0.2: colors.append(WAVE_GRADIENT[2])
        elif y > -0.2: colors.append(WAVE_GRADIENT[3])
        elif y > -0.5: colors.append(WAVE_GRADIENT[4])
        else: colors.append(WAVE_GRADIENT[5])
    return colors

def prepare_display_data(audio_int16, target_len=DISPLAY_LEN):
    """Prepare audio data for visualization"""
    if not audio_int16.size:
        return np.linspace(0, 1, target_len), np.zeros(target_len)
    
    display = audio_int16.astype(np.float32) / 32768.0
    visual_boost = SENSITIVITY_LEVELS[sensitivity_level]['visual_boost']
    display = display * visual_boost
    
    if len(display) >= SMOOTHING:
        kernel = np.ones(SMOOTHING) / SMOOTHING
        display = np.convolve(display, kernel, mode='same')
    
    x_new = np.linspace(0, 1, target_len)
    
    return x_new, np.interp(x_new, np.linspace(0, 1, len(display)), display)

def update_cyber_visual(_frame_idx):
    """Update the cyberpunk visualization"""
    global user_finalized, dot_animation_state, should_exit
    global stream_total_encrypted, stream_is_running, current_audio_entropy, stream_chunk_counter, original_file_removed
    
    dot_animation_state = (dot_animation_state + 1) % 4
    
    # Check for Enter key press
    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        _ = sys.stdin.readline()
        user_finalized = True
        should_exit = True
        if hasattr(update_cyber_visual, 'animation_running'):
            update_cyber_visual.animation_running = False
        return []
    
    if hasattr(update_cyber_visual, 'animation_running') and not update_cyber_visual.animation_running:
        return []
    
    x_disp, y_disp = prepare_display_data(latest_audio_frame, DISPLAY_LEN)
    x_vals = x_disp * (2 * np.pi)
    
    segments = np.array([x_vals, y_disp]).T.reshape(-1, 1, 2)
    segments = np.concatenate([segments[:-1], segments[1:]], axis=1)
    
    if hasattr(update_cyber_visual, 'segments'):
        update_cyber_visual.segments.remove()
    
    colors = colorize_waveform(y_disp)
    
    update_cyber_visual.segments = LineCollection(
        segments,
        colors=colors,
        linewidths=3,
        alpha=0.9,
        linestyle='-',
        antialiased=True
    )
    
    update_cyber_visual.glow.set_data(x_vals, y_disp)
    ax.add_collection(update_cyber_visual.segments)
    
    recording_status = "RECORDING" + "." * dot_animation_state
    status_text = recording_status
    
    status = [
        f"▓▓▓ VOXCRYPT ENCRYPTOR ▓▓▓",
        f"» SENSITIVITY: LEVEL {sensitivity_level} - {SENSITIVITY_LEVELS[sensitivity_level]['description']}",
        f"» CURRENT ENTROPY: {current_audio_entropy.hex()[:8]}..." if current_audio_entropy else "» CURRENT ENTROPY: [SILENT]",
        f"» ENTROPY COLLECTED: {entropy_bits:.1f} bits",
        f"» STATUS: {status_text}"
    ]
    
    if stream_is_running:
        file_size = os.path.getsize(stream_output_path) if stream_output_path and os.path.exists(stream_output_path) else 0
        status.extend([
            "",
            f"▓▓▓ STREAMING ENCRYPTION ACTIVE ▓▓▓",
            f"» CHUNKS: {stream_chunk_counter:,}",
            f"» BYTES: {stream_total_encrypted:,}",
            f"» FILE SIZE: {file_size:,} bytes",
            f"» MODE: {'CONTINUOUS' if not args.input and not args.input_file else 'FILE/TEXT'}",
            f"» NONCE: Counter-based (decryptable)"
        ])
        
        # Tampilkan status replace original
        if original_file_removed:
            status.append(f"» ORIGINAL: REPLACED ✓")
    
    update_cyber_visual.info_txt.set_text("\n".join(status))
    
    return [update_cyber_visual.segments, update_cyber_visual.glow, update_cyber_visual.info_txt]

# ========== MAIN APPLICATION ==========
def main():
    global encryption_active, user_finalized, args, verbose, encryption_done, base_name, audio_stream, sensitivity_level
    global stream_output_path, original_file_removed
    
    parser = argparse.ArgumentParser(
        description="VoxCrypt Ultimate - Audio-Based Secure Encryption with Real-time Streaming",
        epilog="Examples:\n"
               "  Standard file encryption: voxcrypt -I secret.doc -k key.pem\n"
               "  Standard text encryption: voxcrypt -i \"secret message\" -k key.pem\n"
               "  Streaming file encryption: voxcrypt -I file.txt --stream -k key.pem\n"
               "  Streaming text encryption: voxcrypt -i \"Test123\" --stream -k key.pem\n"
               "  Continuous encryption (no input): voxcrypt --stream -k key.pem -o stream.vxc\n"
               "  Replace original file immediately: voxcrypt -I file.txt --stream -k key.pem --replace-original\n"
               "  High sensitivity: voxcrypt -I file.txt -k key.pem -ss 5\n"
               "  Low sensitivity: voxcrypt -I file.txt -k key.pem -ss 1"
    )
    
    parser.add_argument("-i", "--input", help="Text to encrypt")
    parser.add_argument("-I", "--input-file", help="File to encrypt")
    parser.add_argument("--stream", help="Enable real-time streaming encryption", action="store_true")
    parser.add_argument("-o", "--output", help="Output file name (optional)")
    parser.add_argument("--replace-original", help="Replace original file with .vxc immediately during encryption", action="store_true")
    
    parser.add_argument("-k", "--key", help="Output key file", required=True)
    parser.add_argument("--no-visual", help="Disable visualization", action="store_true")
    parser.add_argument("-v", "--verbose", help="Enable verbose output", action="store_true")
    parser.add_argument("-ss", "--sound-sensitivity", 
                       type=int, 
                       choices=[1, 2, 3, 4, 5],
                       default=3,
                       help="Sound sensitivity level (1-5)")
    
    args = parser.parse_args()
    verbose = args.verbose
    sensitivity_level = args.sound_sensitivity
    
    # Validate arguments
    if not args.input and not args.input_file and not args.stream:
        print("[!] ERROR: You must specify either -i (text), -I (file), or --stream for continuous mode")
        sys.exit(1)
    
    # Validate replace-original (only works with file input)
    if args.replace_original and not args.input_file:
        print("[!] ERROR: --replace-original can only be used with -I (file input)")
        sys.exit(1)
    
    try:
        # Initialize
        encryption_active = True
        visualization_enabled = not args.no_visual
        
        if verbose:
            print("[VERBOSE] VoxCrypt Ultimate starting...")
            print(f"[VERBOSE] Arguments: {vars(args)}")
            print(f"[VERBOSE] Using sensitivity level {sensitivity_level}: {SENSITIVITY_LEVELS[sensitivity_level]['description']}")
        
        # Determine mode
        is_stream_mode = args.stream
        is_file_input = args.input_file is not None
        is_text_input = args.input is not None
        
        original_file_path = None
        if is_file_input:
            # File input mode
            input_path = args.input_file
            if not os.path.exists(input_path):
                print(f"[!] ERROR: File not found: {input_path}")
                sys.exit(1)
            
            base_name = os.path.splitext(input_path)[0]
            output_path = args.output if args.output else f"{base_name}.vxc"
            original_file_path = input_path
            
            if verbose:
                print(f"[VERBOSE] Input file: {input_path}")
                print(f"[VERBOSE] Output file: {output_path}")
            
            # Read file data
            with open(input_path, 'rb') as f:
                file_data = f.read()
            
            file_size = len(file_data)
            if verbose:
                print(f"[VERBOSE] File size: {file_size} bytes")
            
        elif is_text_input:
            # Text input mode
            text_input = args.input
            base_name = "message"
            output_path = args.output if args.output else f"{base_name}.vxc"
            
            if verbose:
                print(f"[VERBOSE] Text input: {text_input}")
            
            file_data = text_input.encode('utf-8')
            file_size = len(file_data)
            
            if verbose:
                print(f"[VERBOSE] Text size: {file_size} bytes")
            
        else:
            # Continuous mode (no input)
            base_name = "stream"
            output_path = args.output if args.output else "continuous_stream.vxc"
            file_data = None  # Continuous mode generates its own data
            file_size = 0
            
            print("▓▓▓ CONTINUOUS STREAMING MODE ACTIVATED ▓▓▓")
            print("▓▓▓ Encryption will run indefinitely until stopped ▓▓▓")
        
        # Step 1: Audio capture for key generation
        print("\n▓▓▓ INITIAL AUDIO CAPTURE FOR KEY GENERATION ▓▓▓")
        print("▓▓▓ Press Enter, speak for a few seconds, then press Enter again ▓▓▓")
        
        audio_samples = AudioHandler.record_until_enter()
        if audio_samples.size == 0:
            print("[!] ERROR: No audio captured")
            sys.exit(1)
        
        if verbose:
            print(f"[VERBOSE] Audio captured: {len(audio_samples)} samples")
        
        # Step 2: Generate cryptographic seed
        if verbose:
            print("[VERBOSE] Generating cryptographic seed...")
        
        boost = SENSITIVITY_LEVELS[sensitivity_level]['record_boost']
        amplified = audio_samples.astype(np.float32) * boost
        audio_bytes = amplified.tobytes()
        os_entropy = os.urandom(32)
        seed = hashlib.blake2b(audio_bytes + os_entropy).digest()
        
        # Step 3: Generate key pair
        if verbose:
            print("[VERBOSE] Generating X25519 key pair...")
        
        hkdf = HKDF(
            algorithm=hashes.BLAKE2b(64),
            length=32,
            salt=None,
            info=b"Key generation salt",
            backend=default_backend()
        )
        private_bytes = hkdf.derive(seed)
        
        private_key = x25519.X25519PrivateKey.from_private_bytes(private_bytes[:32])
        public_key = private_key.public_key()
        pub_key_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        # Step 4: Save private key
        if verbose:
            print(f"[VERBOSE] Saving private key to: {args.key}")
        
        private_key_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        with open(args.key, 'wb') as f:
            f.write(private_key_pem)
        
        print(f"\n▓▓▓ ENCRYPTION KEY SAVED TO {args.key} ▓▓▓")
        
        # Step 5: Generate salt from audio
        salt = hashlib.sha256(audio_bytes).digest()[:SALT_SIZE]
        if verbose:
            print(f"[VERBOSE] Using salt: {salt.hex()[:16]}...")
        
        stream_output_path = output_path
        
        # Step 6: Choose encryption mode
        if is_stream_mode:
            # STREAMING MODE (real-time)
            print(f"\n▓▓▓ STARTING {'CONTINUOUS' if not is_file_input and not is_text_input else 'FILE/TEXT'} STREAMING ENCRYPTION ▓▓▓")
            print(f"▓▓▓ Output: {output_path} ▓▓▓")
            
            # Setup real-time encryptor with original file path for immediate replacement
            encryptor = RealTimeStreamEncryptor(
                public_key_pem=pub_key_pem,
                salt=salt,
                output_path=output_path,
                input_data=file_data,  # None for continuous, bytes for file/text
                original_file=original_file_path if args.replace_original else None
            )
            
            # Start encryption thread
            encryptor.start()
            
            if args.replace_original and original_file_path:
                print("▓▓▓ IMMEDIATE REPLACE MODE ACTIVE ▓▓▓")
                print("▓▓▓ Original file will be deleted as soon as encryption starts ▓▓▓")
            
            if not is_file_input and not is_text_input:
                print("▓▓▓ CONTINUOUS ENCRYPTION ACTIVE ▓▓▓")
                print("▓▓▓ File will grow indefinitely based on audio input ▓▓▓")
                print("▓▓▓ Press Ctrl+C to stop ▓▓▓")
            else:
                print("▓▓▓ FILE/TEXT STREAMING ENCRYPTION ACTIVE ▓▓▓")
                print("▓▓▓ Encryption will stop when file/text is fully processed ▓▓▓")
            
        else:
            # STANDARD MODE (non-streaming)
            print(f"\n▓▓▓ STARTING STANDARD ENCRYPTION ▓▓▓")
            
            if is_file_input:
                SecureFile.encrypt_file(args.input_file, output_path, pub_key_pem, args.replace_original)
            else:  # Text input
                SecureFile.encrypt_file_from_data(file_data, output_path, pub_key_pem)
            
            encryption_done = True
        
        # Step 7: Setup visualization if enabled and in streaming mode
        if visualization_enabled and is_stream_mode:
            print(f"\n▓▓▓ STARTING VISUALIZATION ▓▓▓")
            
            # Start audio stream for visualization
            audio_stream = sd.InputStream(
                samplerate=SAMPLE_RATE, 
                channels=1, 
                dtype='int16',
                blocksize=LIVE_CHUNK, 
                callback=AudioHandler.live_audio_callback,
                latency='low'
            )
            audio_stream.start()
            
            # Setup visualization
            global fig, ax
            fig, ax = setup_cyberpunk_display()
            
            update_cyber_visual.glow, = ax.plot([], [], 
                color=CYBER_COLORS['blue'],
                linewidth=18,
                alpha=GLOW_ALPHA
            )
            
            mode_text = "CONTINUOUS" if not is_file_input and not is_text_input else "STREAMING"
            update_cyber_visual.info_txt = ax.text(
                0.02, 0.95,
                f"INITIALIZING {mode_text} ENCRYPTION...",
                transform=ax.transAxes,
                fontsize=10,
                fontfamily='monospace',
                color=CYBER_COLORS['cyan'],
                verticalalignment='top'
            )
            
            fig.suptitle(
                f'»» VOXCRYPT {mode_text} ENCRYPTOR ««',
                color=CYBER_COLORS['pink'],
                fontsize=14,
                fontweight='bold',
                fontfamily='monospace'
            )
            
            # Start animation
            update_cyber_visual.animation_running = True
            ani = FuncAnimation(
                fig, 
                update_cyber_visual, 
                interval=20, 
                blit=True,
                cache_frame_data=False
            )
            
            print(f"\n▓▓▓ VISUALIZATION ACTIVE ▓▓▓")
            print(f"▓▓▓ MONITOR IN ANOTHER TERMINAL: ▓▓▓")
            print(f"▓▓▓   watch -n 0.1 'ls -lh {output_path} && echo && tail -c 32 {output_path} | xxd -p' ▓▓▓")
            print(f"\n▓▓▓ SPEAK INTO MICROPHONE TO AFFECT ENCRYPTION ▓▓▓")
            print(f"▓▓▓ Press Enter in this window to stop visualization ▓▓▓")
            print(f"▓▓▓ Press Ctrl+C in terminal to stop encryption ▓▓▓")
            
            # Show plt
            plt.show()
            
            # Stop animation
            update_cyber_visual.animation_running = False
            if hasattr(ani, 'event_source') and ani.event_source:
                ani.event_source.stop()
            
            # Stop audio stream
            audio_stream.stop()
            audio_stream.close()
            
            print(f"\n▓▓▓ VISUALIZATION STOPPED ▓▓▓")
            if not is_file_input and not is_text_input:
                print(f"▓▓▓ CONTINUOUS ENCRYPTION CONTINUES IN BACKGROUND ▓▓▓")
                print(f"▓▓▓ Press Ctrl+C to stop encryption completely ▓▓▓")
        
        elif visualization_enabled and not is_stream_mode:
            # Standard mode with visualization
            print(f"\n▓▓▓ STANDARD ENCRYPTION COMPLETE ▓▓▓")
            print(f"▓▓▓ Output file: {output_path} ▓▓▓")
        
        else:
            # No visualization mode
            if is_stream_mode:
                print(f"\n▓▓▓ STREAMING ENCRYPTION ACTIVE ▓▓▓")
                print(f"▓▓▓ MONITOR IN ANOTHER TERMINAL: ▓▓▓")
                print(f"▓▓▓   watch -n 0.1 'ls -lh {output_path} && echo && tail -c 32 {output_path} | xxd -p' ▓▓▓")
                print(f"\n▓▓▓ SPEAK INTO MICROPHONE TO AFFECT ENCRYPTION ▓▓▓")
                print(f"▓▓▓ Press Ctrl+C to stop ▓▓▓")
                
                # Keep running until interrupted (for continuous mode) or file completes
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\n▓▓▓ Stopping encryption... ▓▓▓")
            else:
                print(f"\n▓▓▓ STANDARD ENCRYPTION COMPLETE ▓▓▓")
                print(f"▓▓▓ Output file: {output_path} ▓▓▓")
        
        # Step 8: Wait for encryption to complete if in streaming mode
        if is_stream_mode and is_file_input:
            # For file input in streaming mode, wait for encryption to complete
            # The encryptor thread will exit when done
            if hasattr(encryptor, 'thread') and encryptor.thread and encryptor.thread.is_alive():
                encryptor.thread.join(timeout=30)  # Wait up to 30 seconds
        
        # Step 9: Final cleanup
        if is_stream_mode:
            stop_encryption_flag.set()
            # Give the thread a moment to clean up
            time.sleep(0.5)
            
            # Jika file original belum dihapus (misalnya karena error), coba hapus lagi
            if args.replace_original and original_file_path and os.path.exists(original_file_path) and not original_file_removed:
                try:
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        os.remove(original_file_path)
                        original_file_removed = True
                        print(f"\n[!] ORIGINAL FILE REPLACED (final cleanup): {original_file_path} -> {output_path}")
                except Exception as e:
                    print(f"\n[!] WARNING: Could not remove original file during cleanup: {str(e)}")
            
            print(f"\n▓▓▓ ENCRYPTION COMPLETE ▓▓▓")
            print(f"▓▓▓ Output file: {output_path} ▓▓▓")
            print(f"▓▓▓ Key file: {args.key} ▓▓▓")
            if os.path.exists(output_path):
                print(f"▓▓▓ Final size: {os.path.getsize(output_path):,} bytes ▓▓▓")
        
        elif encryption_done:
            print(f"\n▓▓▓ ENCRYPTION COMPLETE ▓▓▓")
            print(f"▓▓▓ Output file: {output_path} ▓▓▓")
            print(f"▓▓▓ Key file: {args.key} ▓▓▓")
            print(f"▓▓▓ File size: {os.path.getsize(output_path):,} bytes ▓▓▓")
        
    except KeyboardInterrupt:
        print("\n▓▓▓ INTERRUPTED BY USER ▓▓▓")
        stop_encryption_flag.set()
    except Exception as e:
        print(f"[!] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        stop_encryption_flag.set()
    finally:
        encryption_active = False
        if audio_stream:
            try:
                audio_stream.stop()
                audio_stream.close()
            except:
                pass

if __name__ == "__main__":
    main()
