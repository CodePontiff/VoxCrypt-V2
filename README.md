
# VoxCrypt V2 🔊🔐

✝️ "Lux in tenebris lucet, et tenebrae eam non comprehenderunt."

**VoxCrypt V2** is an audio-based encryption tool with a cyberpunk-style visualization.
This release introduces major improvements over V1, especially in **cryptography** and support for **dynamic encryption**.

## ✨ Features

* 🎙️ **Audio-seeded encryption**: Keys are derived from voice input + OS entropy.
* 🔑 **X25519 keypair** generated directly from audio.
* 🛡️ **ChaCha20-Poly1305** for strong and fast encryption.
* 🎨 **Cyberpunk waveform visualization** (can be disabled with `--no-visual`).
* 📂 **Encrypt files or plaintext** directly from the command line.
* ⚡ **Streaming mode** via `--stream` (experimental, untested).
* 📜 Authenticated output format with header, salt, HMAC, and ephemeral public key.
* 🔓 **Decryption support** via `voxdecrypt.py`, works for both text and binary files.

---

## 🚀 Usage

### 🔐 Encryption (VoxCrypt_encryptor.py)

```
Encryptor Command Line Options:

| Option                              | Description                                                                             |
|-------------------------------------|-----------------------------------------------------------------------------------------|
| `-i, --input`                       | Encrypt plain text directly from CLI input.                                             |
| `-I, --input-file`                  | Encrypt a file (e.g., `.txt`, `.jpg`, `.mp3`).                                          |
| `--stream`                          | Encrypt data from a stream source (⚠️ experimental, not fully tested).                  |
| `-k, --key`                         | Specify the key file (required).                                                        |
| `--replace-original`                | Replace the original file after encryption (⚠️ **will overwrite**, backup recommended). |
| `--no-visual`                       | Disable visualization (no waveform or graphical output).                                |
| `-v, --verbose`                     | Enable verbose output for detailed process logging.                                     |
| `-ss, --sound-sensitivity`          | Sound sensitivity for audio salt entropy (1 - 5 very low, low, medium, high, very high).|
```

#### 1. Encrypt a File

```bash
python voxcrypt.py -I secret.doc -k mykey.pem
```

* Produces `secret.vxc` (encrypted file).
* Private key saved in `mykey.pem`.

#### 2. Encrypt Plaintext

```bash
python VoxCrypt_encryptor.py -i "secret message" -k textkey.pem
```

* Produces `message.vxc`.

#### 3. Encrypt with Cyberpunk Visualization

```bash
python VoxCrypt_encryptor.py -I notes.txt -k key.pem
```

* Press **Enter** to start audio recording.
* Press **Enter** again to finish.

#### 4. Encrypt Without Visualization

```bash
python VoxCrypt_encryptor.py -I data.bin -k key.pem --no-visual
```

#### 5. Encrypt Dynamic Type Files (Experimental, Untested)

```bash
python VoxCrypt_encryptor.py --stream camera_feed -k streamkey.pem
```

⚠️ This feature is **experimental and untested**. Use at your own risk.

#### ⚠️ Important Warning: `--replace-original`

You can force encryption to overwrite the original file with:

```bash
python VoxCrypt_decryptor.py -i secret.doc -k key.pem --replace-original
```

⚠️ **Warning:**
Using `--replace-original` will **overwrite your original file permanently**.
It is strongly recommended to make a **backup copy** first.
You will need the correct private key to restore the file later.

---

### 🔓 Decryption (VoxCrypt_decryptor.py)

The decryptor works with **both text and binary files**. It verifies HMAC, loads the ephemeral public key, and restores the original plaintext.

```
Decryptor Command Line Options:

| Option                  | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `-i, --input`            | Input encrypted file (`.vxc`) to decrypt.                                  |
| `-k, --key`              | Private key file used for decryption.                                      |
| `-o, --output`           | Output file path where the decrypted content will be saved.                |
```

Example output:

```
▓▓▓ DECRYPTING: secret.vxc (2048 bytes) ▓▓▓
▓▓▓ USING KEY: mykey.pem ▓▓▓
Detected VXC3H format (static file)
Reading file components...
HMAC verified successfully
Decryption successful!
✓ Detected file type: PDF
✓ Decryption completed successfully
▓▓▓ SUCCESS: Decrypted to recovered.doc ▓▓▓
```

#### Notes

* The decryptor automatically detects whether the decrypted data is **text or binary**.
* It attempts to identify file signatures (JPEG, PNG, PDF, ZIP, MP3, etc.) for convenience.
* If verification fails, the decryptor cleans up any partial output file.

---

## 📦 Dummy

The example_backups folder is dummy test subject that you can experience with them:

```
example_backups/
 ├── chill_guy.jpg
 ├── rick_roll.mp3
 └── lorem_ipsum.txt
```

---

## 🛠️ Requirements

* Python 3.8+
* Dependencies:

  * `cryptography>=41.0.0`
  * `pycryptodome>=3.19.0`
  * `numpy>=1.24.0`
  * `matplotlib>=3.7.0`

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔮 Roadmap:

* [ ] Fully implement and test `--stream` mode.

## 📷 Screenshoot:

<img width="1342" height="143" alt="image" src="https://github.com/user-attachments/assets/3077528a-13be-4f37-8658-87eddfc33038" />
<img width="1236" height="664" alt="image" src="https://github.com/user-attachments/assets/9cbac172-4110-49ac-b97e-01b6065a3ad0" />
<img width="675" height="443" alt="image" src="https://github.com/user-attachments/assets/ec2c66b7-7b60-4b46-bdac-9d130af9f32d" />

---

## ⚠️ Disclaimer

VoxCrypt is an experimental project. Do not use it for production or highly sensitive data without a full security audit.

---
