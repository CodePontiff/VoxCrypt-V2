#!/usr/bin/env python3
"""
VoxCrypt-V2 Encryptor - Secure Audio-Based Encryption Tool
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
SAMPLE_RATE = 48000  # Higher sample rate for more detail
SEED_CHUNK = 4096    # Larger chunk for better audio capture
LIVE_CHUNK = 2048
DISPLAY_LEN = 1024
SALT_SIZE = 32
HKDF_INFO = b"VoxCrypt v2 Key Derivation"
NONCE_SIZE = 12
SMOOTHING = 4        # Minimal smoothing to preserve all audio details

# ========== SENSITIVITY CONFIG ==========
SENSITIVITY_LEVELS = {
    1: {  # Very Low
        'min_rms': 0.01,
        'min_peak': 0.05,
        'record_boost': 2,    # 6dB
        'live_boost': 1.5,    # 3.5dB
        'visual_boost': 1,    # 0dB
        'description': 'VERY LOW - For loud environments'
    },
    2: {  # Low
        'min_rms': 0.005,
        'min_peak': 0.025,
        'record_boost': 4,    # 12dB
        'live_boost': 3,      # 9.5dB
        'visual_boost': 2,    # 6dB
        'description': 'LOW - For normal speaking volume'
    },
    3: {  # Medium (Default)
        'min_rms': 0.001,
        'min_peak': 0.005,
        'record_boost': 8,    # 18dB
        'live_boost': 6,      # 15.6dB
        'visual_boost': 4,    # 12dB
        'description': 'MEDIUM - For quiet speaking'
    },
    4: {  # High
        'min_rms': 0.0005,
        'min_peak': 0.0025,
        'record_boost': 16,   # 24dB
        'live_boost': 12,     # 21.6dB
        'visual_boost': 8,    # 18dB
        'description': 'HIGH - For whispers'
    },
    5: {  # Very High
        'min_rms': 0.0001,
        'min_peak': 0.0005,
        'record_boost': 32,   # 30dB
        'live_boost': 24,     # 27.6dB
        'visual_boost': 16,   # 24dB
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
sensitivity_level = 3  # Default to medium sensitivity

# ========== CRYPTO CORE ==========
class AudioCrypto:
    @staticmethod
    def generate_audio_seed(audio_samples):
        """Generate cryptographic seed from audio with OS entropy mixing"""
        if verbose:
            print("[VERBOSE] Generating audio seed with OS entropy mixing...")
        
        # Apply amplification based on sensitivity level
        boost = SENSITIVITY_LEVELS[sensitivity_level]['record_boost']
        amplified = audio_samples.astype(np.float32) * boost
        
        audio_bytes = amplified.tobytes()
        os_entropy = os.urandom(32)
        seed = hashlib.blake2b(audio_bytes + os_entropy).digest()
        
        if verbose:
            print(f"[VERBOSE] Audio seed generated: {seed.hex()[:16]}...")
            rms = np.sqrt(np.mean(amplified**2))
            peak = np.max(np.abs(amplified))
            print(f"[VERBOSE] Enhanced audio levels - RMS: {rms:.8f}, Peak: {peak:.8f}")
            print(f"[VERBOSE] Using sensitivity level {sensitivity_level}: {SENSITIVITY_LEVELS[sensitivity_level]['description']}")
            
        return seed

    @staticmethod
    def derive_keys(seed, salt=None, info=HKDF_INFO, length=32):
        """HKDF with BLAKE2b for key derivation"""
        if verbose:
            print(f"[VERBOSE] Deriving keys with HKDF (salt: {salt.hex()[:8] if salt else 'None'}...)")
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
        if verbose:
            print("[VERBOSE] Generating X25519 key pair from audio seed...")
        private_key = x25519.X25519PrivateKey.from_private_bytes(
            AudioCrypto.derive_keys(seed, info=b"Key generation salt")
        )
        public_key = private_key.public_key()
        if verbose:
            print("[VERBOSE] Key pair generated successfully")
        return private_key, public_key

    @staticmethod
    def encrypt_data(data, public_key_pem, salt):
        """ChaCha20Poly1305 encryption with ephemeral key pair - FIXED"""
        if verbose:
            print("[VERBOSE] Starting ChaCha20Poly1305 encryption...")
        
        ephemeral_private = x25519.X25519PrivateKey.generate()
        if verbose:
            print("[VERBOSE] Generated ephemeral private key")
        
        # Load the public key from PEM bytes (FIXED)
        public_key = serialization.load_pem_public_key(public_key_pem)
        if verbose:
            print("[VERBOSE] Loaded recipient public key")
        
        shared_key = ephemeral_private.exchange(public_key)
        if verbose:
            print(f"[VERBOSE] Shared key established: {shared_key.hex()[:16]}...")
        
        enc_key = AudioCrypto.derive_keys(shared_key, salt)
        if verbose:
            print(f"[VERBOSE] Encryption key derived: {enc_key.hex()[:16]}...")
        
        cipher = ChaCha20Poly1305(enc_key)
        nonce = os.urandom(NONCE_SIZE)
        if verbose:
            print(f"[VERBOSE] Generated nonce: {nonce.hex()}")
        
        ciphertext = cipher.encrypt(nonce, data, None)
        if verbose:
            print(f"[VERBOSE] Data encrypted: {len(ciphertext)} bytes")
        
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
        
        print(f"»» SPEAK NOW - PRESS ENTER WHEN DONE ««")
        print(f"»» SENSITIVITY LEVEL: {sensitivity_level} ({SENSITIVITY_LEVELS[sensitivity_level]['description']}) ««")
        audio_chunks = []
        
        # Recording settings based on sensitivity level
        boost = SENSITIVITY_LEVELS[sensitivity_level]['record_boost']
        
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE, 
            channels=1, 
            dtype='int16', 
            blocksize=SEED_CHUNK,
            latency='low',  # Lowest latency for maximum sensitivity
            extra_settings=None  # No additional filtering
        )
        stream.start()
        
        try:
            if verbose:
                print(f"[VERBOSE] Audio capture started with sensitivity level {sensitivity_level}...")
                print(f"[VERBOSE] Using {boost}x amplification ({20*np.log10(boost):.1f}dB boost)")
            while True:
                data, _ = stream.read(SEED_CHUNK)
                # Apply amplification based on sensitivity level
                amplified_data = data * boost
                audio_chunks.append(amplified_data.copy().flatten())
                
                # Check for Enter key press
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
            if audio_data.size > 0:
                rms = np.sqrt(np.mean(audio_data.astype(np.float32)**2))
                peak = np.max(np.abs(audio_data))
                print(f"[VERBOSE] Audio levels - RMS: {rms:.8f}, Peak: {peak}")
        return audio_data

    @staticmethod
    def frame_has_voice(frame_i16):
        """Detect if audio frame contains any sound - sensitivity based on level"""
        if frame_i16.size == 0:
            return False
        
        # Get sensitivity thresholds
        min_rms = SENSITIVITY_LEVELS[sensitivity_level]['min_rms']
        min_peak = SENSITIVITY_LEVELS[sensitivity_level]['min_peak']
        boost = SENSITIVITY_LEVELS[sensitivity_level]['live_boost']
        
        # Convert to float and normalize
        audio_float = frame_i16.astype(np.float32) / 32768.0
        
        # Apply amplification based on sensitivity level
        amplified = audio_float * boost
        
        # Calculate RMS (root mean square) for volume detection
        rms = np.sqrt(np.mean(amplified**2))
        
        # Calculate peak amplitude
        peak = np.max(np.abs(amplified))
        
        # Sensitivity-based thresholds
        has_sound = (rms >= min_rms) or (peak >= min_peak)
        
        if verbose and has_sound:
            print(f"[VERBOSE] Sound detected - RMS: {rms:.8f}, Peak: {peak:.8f}")
            
        return has_sound

    @staticmethod
    def calculate_entropy(audio_data):
        """Calculate approximate entropy bits from audio data"""
        if audio_data.size < 100:
            return 0
            
        # Convert to normalized float
        audio_float = audio_data.astype(np.float32) / 32768.0
        
        # Calculate simple entropy approximation
        diff = np.diff(audio_float)
        unique_vals = len(np.unique(np.round(diff, decimals=4)))
        
        # Approximate entropy bits (log2 of unique values)
        entropy_bits = np.log2(max(2, unique_vals))
        
        return entropy_bits

    @staticmethod
    def live_audio_callback(indata, frames, time_info, status):
        """Live audio callback for visualization and salt generation"""
        global latest_audio_frame, current_salt, current_salt_src, entropy_collected, entropy_bits
        
        # Apply amplification based on sensitivity level
        boost = SENSITIVITY_LEVELS[sensitivity_level]['live_boost']
        amplified_data = indata[:, 0].astype(np.float32) * boost
        latest_audio_frame = amplified_data.astype(np.int16)
        
        if AudioHandler.frame_has_voice(latest_audio_frame):
            # Calculate entropy from this frame
            frame_entropy = AudioHandler.calculate_entropy(latest_audio_frame)
            entropy_bits += frame_entropy
            entropy_collected += 1
            
            # Update salt with this high-entropy audio
            current_salt = hashlib.blake2b(latest_audio_frame.tobytes()).digest()
            current_salt_src = f"mic-voice ({entropy_bits:.1f} bits)"
            
            if verbose and entropy_collected % 10 == 0:
                print(f"[VERBOSE] Entropy collected: {entropy_bits:.1f} bits from {entropy_collected} frames")
                print(f"[VERBOSE] Current salt: {current_salt.hex()[:8]}...")
        else:
            current_salt = b""
            current_salt_src = "static"

# ========== FILE OPERATIONS ==========
class SecureFile:
    @staticmethod
    def encrypt_file(input_path, output_path, public_key_pem):
        """Encrypt file with authenticated format - FIXED"""
        if verbose:
            print(f"[VERBOSE] Reading input file: {input_path}")
        with open(input_path, 'rb') as f:
            data = f.read()
        
        if verbose:
            print(f"[VERBOSE] File read: {len(data)} bytes")
        
        salt = os.urandom(SALT_SIZE)
        if verbose:
            print(f"[VERBOSE] Generated salt: {salt.hex()[:8]}...")
        
        encrypted = AudioCrypto.encrypt_data(data, public_key_pem, salt)
        
        hmac_key = AudioCrypto.derive_keys(salt, info=b"HMAC key")
        h = hmac.HMAC(hmac_key, hashes.BLAKE2b(64), backend=default_backend())

        data_to_hmac = encrypted['ephemeral_pub'] + encrypted['nonce'] + encrypted['ciphertext']
        h.update(data_to_hmac)
        hmac_value = h.finalize()
        
        if verbose:
            print(f"[VERBOSE] HMAC computed: {hmac_value.hex()[:8]}...")
            print(f"[VERBOSE] Writing encrypted file: {output_path}")
        
        with open(output_path, 'wb') as f:
            f.write(b'VXC3H')  # Format marker
            f.write(salt)
            f.write(hmac_value)
            f.write(encrypted['ephemeral_pub'])
            f.write(encrypted['nonce'])
            f.write(encrypted['ciphertext'])
        
        if verbose:
            print("[VERBOSE] File encryption completed successfully")

    @staticmethod
    def encrypt_file_from_data(data, output_path, public_key_pem):
        """Encrypt raw data with authenticated format (for text input) - FIXED"""
        if verbose:
            print(f"[VERBOSE] Encrypting text data: {len(data)} bytes")
        
        salt = os.urandom(SALT_SIZE)
        if verbose:
            print(f"[VERBOSE] Generated salt: {salt.hex()[:8]}...")
        
        encrypted = AudioCrypto.encrypt_data(data, public_key_pem, salt)
        
        hmac_key = AudioCrypto.derive_keys(salt, info=b"HMAC key")
        h = hmac.HMAC(hmac_key, hashes.BLAKE2b(64), backend=default_backend())

        data_to_hmac = encrypted['ephemeral_pub'] + encrypted['nonce'] + encrypted['ciphertext']
        h.update(data_to_hmac)
        hmac_value = h.finalize()
        
        if verbose:
            print(f"[VERBOSE] HMAC computed: {hmac_value.hex()[:8]}...")
            print(f"[VERBOSE] Writing encrypted file: {output_path}")
        
        with open(output_path, 'wb') as f:
            f.write(b'VXC3H')  # Format marker
            f.write(salt)
            f.write(hmac_value)
            f.write(encrypted['ephemeral_pub'])
            f.write(encrypted['nonce'])
            f.write(encrypted['ciphertext'])
        
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
    
    # Convert to float and normalize
    display = audio_int16.astype(np.float32) / 32768.0
    
    # Apply amplification based on sensitivity level for visualization
    visual_boost = SENSITIVITY_LEVELS[sensitivity_level]['visual_boost']
    display = display * visual_boost
    
    # Minimal smoothing to preserve all audio details
    if len(display) >= SMOOTHING:
        kernel = np.ones(SMOOTHING) / SMOOTHING
        display = np.convolve(display, kernel, mode='same')
    
    x_new = np.linspace(0, 1, target_len)
    
    return x_new, np.interp(x_new, np.linspace(0, 1, len(display)), display)

def update_cyber_visual(_frame_idx):
    """Update the cyberpunk visualization with verbose details"""
    global user_finalized, dot_animation_state, should_exit
    
    # Increment animation state (cycles 0-3)
    dot_animation_state = (dot_animation_state + 1) % 4
    
    # Check for Enter key press
    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        _ = sys.stdin.readline()
        user_finalized = True
        should_exit = True
        plt.close()  # Close the visualization window
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
    
    # Create animated recording status
    recording_status = "RECORDING" + "." * dot_animation_state
    if encryption_done:
        status_text = "ENCRYPTION COMPLETE"
    else:
        status_text = recording_status
    
    status = [
        f"▓▓▓ ADJUSTABLE ENTROPY COLLECTION ▓▓▓",
        f"» SENSITIVITY: LEVEL {sensitivity_level} - {SENSITIVITY_LEVELS[sensitivity_level]['description']}",
        f"» SALT: {current_salt.hex()[:12]}..." if current_salt else "» SALT: [SYSTEM DEFAULT]",
        f"» KEY SOURCE: {current_salt_src.upper()}",
        f"» ENTROPY COLLECTED: {entropy_bits:.1f} bits",
        f"» STATUS: {status_text}"
    ]
    
    if encryption_done:
        status.extend([
            "",
            "▓▓▓ MISSION SUMMARY ▓▓▓",
            f"» OUTPUT: {base_name}.vxc",
            f"» TOTAL ENTROPY: {entropy_bits:.1f} bits"
        ])
        
        if verbose:
            status.extend([
                "",
                "▓▓▓ LIVE CRYPTO DETAILS ▓▓▓",
                f"» CURRENT SALT: {current_salt.hex()[:16]}..." if current_salt else "» NO ACTIVE SALT"
            ])
    
    update_cyber_visual.info_txt.set_text("\n".join(status))
    
    return [update_cyber_visual.segments, update_cyber_visual.glow, update_cyber_visual.info_txt]

# ========== MAIN APPLICATION ==========
def main():
    global encryption_active, user_finalized, args, verbose, encryption_done, base_name, audio_stream, sensitivity_level
    
    parser = argparse.ArgumentParser(
        description="VoxCrypt Ultimate - Audio-Based Secure Encryption with Adjustable Sensitivity",
        epilog="Examples:\n"
               "  Encrypt file: voxcrypt -I secret.doc -k key.pem\n"
               "  Encrypt text: voxcrypt -i \"secret message\" -k text.pem\n"
               "  High sensitivity: voxcrypt -I file.txt -k key.pem -ss 5\n"
               "  Low sensitivity: voxcrypt -I file.txt -k key.pem -ss 1"
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-i", "--input", help="Text to encrypt")
    group.add_argument("-I", "--input-file", help="File to encrypt")
    group.add_argument("--stream", help="Stream source to encrypt")
    
    parser.add_argument("-k", "--key", help="Output key file", required=True)
    parser.add_argument("--replace-original", help="Replace original file after encryption", action="store_true")
    parser.add_argument("--no-visual", help="Disable visualization", action="store_true")
    parser.add_argument("-v", "--verbose", help="Enable verbose output during process", action="store_true")
    parser.add_argument("-ss", "--sound-sensitivity", 
                       type=int, 
                       choices=[1, 2, 3, 4, 5],
                       default=3,
                       help="Sound sensitivity level (1-5): 1=Very Low, 2=Low, 3=Medium, 4=High, 5=Very High")
    
    args = parser.parse_args()
    verbose = args.verbose
    sensitivity_level = args.sound_sensitivity
    
    try:
        # Initialize
        encryption_active = True
        visualization_enabled = not args.no_visual
        
        if verbose:
            print("[VERBOSE] VoxCrypt Ultimate starting...")
            print(f"[VERBOSE] Arguments: {vars(args)}")
            print(f"[VERBOSE] Using sensitivity level {sensitivity_level}: {SENSITIVITY_LEVELS[sensitivity_level]['description']}")
        
        # Set base name
        if args.input_file:
            base_name = os.path.splitext(args.input_file)[0]
        else:
            base_name = "message"
        
        # Audio seed generation
        if verbose:
            print("[VERBOSE] Starting audio capture...")
        audio_samples = AudioHandler.record_until_enter()
        if audio_samples.size == 0:
            print("[!] ERROR: No audio captured")
            sys.exit(1)
            
        if verbose:
            print("[VERBOSE] Generating cryptographic seed from audio...")
        seed = AudioCrypto.generate_audio_seed(audio_samples)
        
        # Key generation
        if verbose:
            print("[VERBOSE] Generating X25519 key pair...")
        private_key, public_key = AudioCrypto.generate_key_pair(seed)
        pub_key_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        # Save key
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
        
        # Setup visualization
        if visualization_enabled:
            global fig, ax
            if verbose:
                print("[VERBOSE] Initializing visualization...")
            fig, ax = setup_cyberpunk_display()
            
            # Initialize visualization elements
            update_cyber_visual.glow, = ax.plot([], [], 
                color=CYBER_COLORS['blue'],
                linewidth=18,
                alpha=GLOW_ALPHA
            )
            
            update_cyber_visual.info_txt = ax.text(
                0.02, 0.95,
                "INITIALIZING ENTROPY COLLECTION...",
                transform=ax.transAxes,
                fontsize=10,
                fontfamily='monospace',
                color=CYBER_COLORS['cyan'],
                verticalalignment='top'
            )
            
            fig.suptitle(
                f'»» VOXCRYPT ENCRYPTOR - SENSITIVITY LEVEL {sensitivity_level} ««',
                color=CYBER_COLORS['pink'],
                fontsize=14,
                fontweight='bold',
                fontfamily='monospace'
            )
            
            # Start audio stream for visualization and salt
            audio_stream = sd.InputStream(
                samplerate=SAMPLE_RATE, 
                channels=1, 
                dtype='int16',
                blocksize=LIVE_CHUNK, 
                callback=AudioHandler.live_audio_callback,
                latency='low'  # Lower latency for better sensitivity
            )
            audio_stream.start()
            
            # Start animation
            ani = FuncAnimation(
                fig, 
                update_cyber_visual, 
                interval=20, 
                blit=True,
                cache_frame_data=False
            )
            
            print(f"\n▓▓▓ LIVE ENTROPY COLLECTION ACTIVE - SENSITIVITY LEVEL {sensitivity_level} ▓▓▓")
            print(f"▓▓▓ {SENSITIVITY_LEVELS[sensitivity_level]['description']} ▓▓▓")
            print("▓▓▓ PRESS ENTER TO FINALIZE ▓▓▓")
            plt.show()
            
            # Cleanup
            audio_stream.stop()
            audio_stream.close()
        else:
            # Non-visual mode
            print(f"\n▓▓▓ ENCRYPTION IN PROGRESS - SENSITIVITY LEVEL {sensitivity_level} ▓▓▓")
            print(f"▓▓▓ {SENSITIVITY_LEVELS[sensitivity_level]['description']} ▓▓▓")
            print("▓▓▓ PRESS ENTER TO FINALIZE ▓▓▓")
            input()
            user_finalized = True
        
        # Perform encryption
        if args.stream:
            output_path = f"{args.stream}.vxcs"
            print(f"\n▓▓▓ ENCRYPTING STREAM: {args.stream} ▓▓▓")
            if verbose:
                print("[VERBOSE] Stream encryption selected (not implemented)")
            print("Stream encryption not implemented")
        else:
            output_path = f"{base_name}.vxc"
            
            if args.input:
                if verbose:
                    print(f"[VERBOSE] Encrypting text input: {args.input}")
                data = args.input.encode()
                SecureFile.encrypt_file_from_data(data, output_path, pub_key_pem)
            else:
                if verbose:
                    print(f"[VERBOSE] Encrypting file: {args.input_file}")
                SecureFile.encrypt_file(args.input_file, output_path, pub_key_pem)
            
            if args.replace_original and args.input_file:
                if verbose:
                    print(f"[VERBOSE] Removing original file: {args.input_file}")
                os.remove(args.input_file)
                
        encryption_done = True
        print(f"\n▓▓▓ ENCRYPTION COMPLETE ▓▓▓")
        print(f"» Output: {output_path}")
        print(f"» Total entropy collected: {entropy_bits:.1f} bits")
        
        if verbose:
            print("[VERBOSE] Encryption process completed successfully")
        
        # Wait for user to press Enter before exiting
        print("\n▓▓▓ PRESS ENTER TO EXIT ▓▓▓")
        input()
        
    except Exception as e:
        print(f"[!] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Wait for user to press Enter even on error
        print("\n▓▓▓ PRESS ENTER TO EXIT ▓▓▓")
        input()
        
        sys.exit(1)
    finally:
        encryption_active = False
        # Clean up audio stream if it exists
        if audio_stream:
            try:
                audio_stream.stop()
                audio_stream.close()
            except:
                pass

if __name__ == "__main__":
    main()
