import numpy as np
from PIL import Image
import hashlib
from scipy.fftpack import dct, idct
import logging
import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)

# --- Configuration and Setup ---
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def preprocess_image(image):
    """Preprocess image to ensure consistent format and size"""
    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Ensure dimensions are multiples of 8
    width, height = image.size
    new_width = width - (width % 8)
    new_height = height - (height % 8)
    if new_width != width or new_height != height:
        image = image.crop((0, 0, new_width, new_height))

    return image

def dct2(a):
    """2D Discrete Cosine Transform"""
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

def idct2(a):
    """2D Inverse Discrete Cosine Transform"""
    return idct(idct(a.T, norm='ortho').T, norm='ortho')

def hash_security_code(security_code):
    """Hashes the security code using SHA256"""
    return hashlib.sha256(security_code.encode('utf-8')).hexdigest()[:32]

def allowed_file(filename):
    """Checks if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def embed_message(image, message, hashed_security_code):
    """Embeds a message into an image using improved DCT method."""
    try:
        # Preprocess the image
        image = preprocess_image(image)

        # Convert to YCbCr and get components
        ycbcr = image.convert('YCbCr')
        y, cb, cr = ycbcr.split()
        y_array = np.array(y, dtype=np.float32)
        height, width = y_array.shape

        # Use a simplified quantization matrix for better stability
        Q = np.ones((8, 8)) * 16
        Q[0:4, 0:4] = np.array([
            [8, 8, 8, 8],
            [8, 8, 8, 8],
            [8, 8, 8, 8],
            [8, 8, 8, 8]
        ])

        # Prepare data with security code, message, and multiple delimiters
        start_delimiter = b'\xAA\xAA\xAA\xAA'
        end_delimiter = b'\xFF\xFF\xFF\xFF'
        data_to_hide = (start_delimiter +
                       hashed_security_code.encode('utf-8') +
                       message.encode('utf-8') +
                       end_delimiter)

        data_bits = ''.join(format(byte, '08b') for byte in data_to_hide)

        # Calculate maximum capacity
        max_bits = (height // 8) * (width // 8) * 8  # Increased capacity
        if len(data_bits) > max_bits:
            raise ValueError(f"Message too long. Maximum message length is {(max_bits - 256 - 64) // 8} bytes")

        bit_index = 0
        # Process each 8x8 block
        for i in range(0, height - height % 8, 8):
            for j in range(0, width - width % 8, 8):
                if bit_index >= len(data_bits):
                    break

                block = y_array[i:i+8, j:j+8]
                dct_block = dct2(block)
                quantized = np.round(dct_block / Q)

                # Use more coefficients in mid-frequency range
                positions = [(1,1), (2,1), (1,2), (2,2), (3,1), (1,3), (2,3), (3,2)]
                for pos in positions:
                    if bit_index < len(data_bits):
                        bit = int(data_bits[bit_index])
                        # Ensure coefficient is even/odd based on bit
                        current_val = int(quantized[pos])
                        if bit == 0 and current_val % 2 == 1:
                            quantized[pos] = current_val - 1
                        elif bit == 1 and current_val % 2 == 0:
                            quantized[pos] = current_val + 1
                        bit_index += 1

                # Reconstruct block
                dct_block = quantized * Q
                y_array[i:i+8, j:j+8] = idct2(dct_block)

        # Ensure pixel values are in valid range
        y_array = np.clip(y_array, 0, 255)

        # Reconstruct image
        y_embedded = Image.fromarray(y_array.astype(np.uint8))
        embedded_image = Image.merge('YCbCr', (y_embedded, cb, cr)).convert('RGB')
        return embedded_image

    except Exception as e:
        logging.error(f"Error in embed_message: {str(e)}")
        raise

def extract_message(image, hashed_security_code):
    """Extracts a message from an image using improved DCT method."""
    try:
        # Preprocess the image
        image = preprocess_image(image)

        # Convert to YCbCr and get Y channel
        ycbcr = image.convert('YCbCr')
        y, _, _ = ycbcr.split()
        y_array = np.array(y, dtype=np.float32)
        height, width = y_array.shape

        # Use same quantization matrix as embedding
        Q = np.ones((8, 8)) * 16
        Q[0:4, 0:4] = np.array([
            [8, 8, 8, 8],
            [8, 8, 8, 8],
            [8, 8, 8, 8],
            [8, 8, 8, 8]
        ])

        extracted_bits = []
        # Process each 8x8 block
        for i in range(0, height - height % 8, 8):
            for j in range(0, width - width % 8, 8):
                block = y_array[i:i+8, j:j+8]
                dct_block = dct2(block)
                quantized = np.round(dct_block / Q)

                # Extract from same positions as embedding
                positions = [(1,1), (2,1), (1,2), (2,2), (3,1), (1,3), (2,3), (3,2)]        
                for pos in positions:
                    bit = int(quantized[pos]) % 2
                    extracted_bits.append(str(bit))

        # Convert bits to bytes
        extracted_bits = ''.join(extracted_bits)
        extracted_bytes = bytearray()
        for i in range(0, len(extracted_bits), 8):
            if i + 8 <= len(extracted_bits):
                byte = int(extracted_bits[i:i+8], 2)
                extracted_bytes.append(byte)

        # Look for start delimiter
        start_delimiter = b'\xAA\xAA\xAA\xAA'
        start_pos = extracted_bytes.find(start_delimiter)
        if start_pos == -1:
            return {"success": False, "error": "Could not find start of message"}

        # Extract security code
        security_start = start_pos + len(start_delimiter)
        extracted_security = extracted_bytes[security_start:security_start + 32].decode('utf-8')

        if extracted_security != hashed_security_code:
            return {"success": False, "error": "Invalid security code"}

        # Find end delimiter and extract message
        message_start = security_start + 32
        end_delimiter = b'\xFF\xFF\xFF\xFF'
        end_pos = extracted_bytes.find(end_delimiter, message_start)

        if end_pos == -1:
            return {"success": False, "error": "Could not find end of message"}

        message = extracted_bytes[message_start:end_pos].decode('utf-8')
        return {"success": True, "message": message}

    except Exception as e:
        logging.error(f"Error in extract_message: {str(e)}")
        return {"success": False, "error": f"Error during message extraction: {str(e)}"}

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/app')  # Route for the app page
def app_page():
    return render_template('app.html')


@app.route('/encrypt', methods=['POST'])
def encrypt():
    """Handles the encryption process."""
    original_file_path = None
    try:
        if 'encryptImage' not in request.files:
            return jsonify({"success": False, "error": "No image file"}), 400

        file = request.files['encryptImage']
        if file.filename == '':
            return jsonify({"success": False, "error": "No selected file"}), 400

        if not allowed_file(file.filename):
            return jsonify({"success": False, "error": "File type not allowed"}), 400

        filename = secure_filename(file.filename)
        original_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(original_file_path)

        # Open and preprocess image
        image = Image.open(original_file_path)
        security_code = request.form['encryptSecurityCode']
        hashed_security_code = hash_security_code(security_code)
        message = request.form['encryptMessage']

        embedded_image = embed_message(image, message, hashed_security_code)
        encrypted_filename = f"encrypted_{filename}"
        encrypted_file_path = os.path.join(app.config['UPLOAD_FOLDER'], encrypted_filename)
        # Save as PNG to avoid compression artifacts
        embedded_image.save(encrypted_file_path, 'PNG', quality=100)

        return jsonify({"success": True, "filename": encrypted_filename})

    except Exception as e:
        logging.exception("Encryption error:")
        return jsonify({"success": False, "error": str(e)}), 500

    finally:
        if original_file_path and os.path.exists(original_file_path):
            os.remove(original_file_path)

@app.route('/decrypt', methods=['POST'])
def decrypt():
    """Handles the decryption process."""
    temp_filepath = None
    try:
        if 'decryptImage' not in request.files:
            return jsonify({"success": False, "error": "No image file"}), 400

        file = request.files['decryptImage']
        if file.filename == '':
            return jsonify({"success": False, "error": "No selected file"}), 400

        if not allowed_file(file.filename):
            return jsonify({"success": False, "error": "File type not allowed"}), 400

        security_code = request.form.get('decryptSecurityCode')
        if not security_code:
            return jsonify({"success": False, "error": "Missing security code"}), 400

        hashed_security_code = hash_security_code(security_code)

        filename = secure_filename(file.filename)
        temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_filepath)

        image = Image.open(temp_filepath)
        result = extract_message(image, hashed_security_code)

        if not result["success"]:
            logging.warning(f"Decryption failed: {result['error']}")

        return jsonify(result)

    except Exception as e:
        logging.exception("Decryption error:")
        return jsonify({"success": False, "error": str(e)}), 500

    finally:
        if temp_filepath and os.path.exists(temp_filepath):
            os.remove(temp_filepath)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    
@app.route('/contact', methods=['POST'])
def contact():
    """Handles the contact form submission."""
    try:
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')

        # Basic validation (you can add more)
        if not name or not email or not message:
            return jsonify({'success': False, 'error': 'All fields are required'}), 400

        # Log the contact form submission. This is the important part.
        logging.info(f'Contact Form Submission: Name: {name}, Email: {email}, Message: {message}')

        # In a real application, you would send an email here or save to a database.
        # For this example, we're just logging.

        return jsonify({'success': True, 'message': 'Message received!'})

    except Exception as e:
        logging.exception("Error in contact form submission")
        return jsonify({'success': False, 'error': 'An error occurred'}), 500


if __name__ == '__main__':
    app.run(debug=True)