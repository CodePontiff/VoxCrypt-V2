# VoxCrypt - Audio-Based Secure Encryption
 
 
 ## 🔐 About
 
 VoxCrypt is an innovative encryption tool that uses your voice as a cryptographic seed, combined with modern encryption algorithms for secure file protection. It features:
 
 - **Audio-derived keys** - Your voice generates unique encryption keys
 - **Dual encryption modes** - Choose between static file or continuous stream encryption
 - **Military-grade crypto** - XChaCha20-Poly1305 + X25519 + HKDF-BLAKE2b
 - **Cyberpunk visualization** - Real-time audio waveform display
 
 ## 📦 Installation
```
git clone https://github.com/yourusername/VoxCrypt.git
cd VoxCrypt
pip install -r requirements.txt
```
```
Requirements:

Python 3.8+

cryptography library

sounddevice for audio capture

matplotlib for visualization
```

🚀 Usage
Encryption
```
# Encrypt a file (creates document.pdf.vxc)
python VoxCrypt.py -I document.pdf -k doc_key.pem

# Encrypt a stream (creates data.log.vxcs)
python VoxCrypt.py --stream data.log -k stream_key.pem

# Encrypt text (creates message.vxc)
python VoxCrypt.py -i "Secret message" -k msg_key.pem
Decryption

# Decrypt files
python VoxCrypt_decryptor.py -i file.vxc -k key.pem -o output.txt

# Decrypt streams 
python VoxCrypt_decryptor.py -i log.vxcs -k stream_key.pem -o original.log
```

🔧 Technical Details
Encryption Modes
Mode	Extension	Best For	Key Feature
Static	.vxc	Files, text	Single encryption
Stream	.vxcs	Logs, databases	Chunked with live salt
Cryptographic Stack
Key Exchange: X25519 elliptic curve

Encryption: XChaCha20-Poly1305 (256-bit)

Key Derivation: HKDF with BLAKE2b

Authentication: HMAC-BLAKE2b

🛡️ Security Features
Perfect Forward Secrecy - Ephemeral session keys

Authenticated Encryption - HMAC verified decryption

Voice Activity Detection - Dynamic salt generation

Entropy Mixing - Combines audio with OS randomness

🌟 Example Workflow
Generate encrypted file:

bash
python VoxCrypt.py -I secret.docx -k secret.pem
Securely transfer:

secret.docx.vxc (encrypted file)

secret.pem (private key)

Decrypt on recipient side:

bash
python VoxCrypt_decryptor.py -i secret.docx.vxc -k secret.pem -o secret.docx
📜 License
MIT License - See LICENSE for details.

⚠️ Disclaimer
This is experimental software. For truly sensitive data, use professionally audited tools like GPG or VeraCrypt.

"Your voice is your key" 🔑🗝️
