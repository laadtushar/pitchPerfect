from ppadb.client import Client as AdbClient
import time
from PIL import Image
import numpy as np
import cv2
import pytesseract
import openai
from dotenv import load_dotenv
import os
import cv2
import numpy as np
import re
import xml.etree.ElementTree as ET

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def get_ui_hierarchy(device):
    """
    Dumps the current UI hierarchy to a local file 'window_dump.xml'
    and returns the file path.
    """
    try:
        device.shell("uiautomator dump /sdcard/window_dump.xml")
        device.pull("/sdcard/window_dump.xml", "window_dump.xml")
        return "window_dump.xml"
    except Exception as e:
        print(f"Error getting UI hierarchy: {e}")
        return None


def parse_bounds(bounds_str):
    # bounds="[x1,y1][x2,y2]"
    match = re.match(r'\[(\d+),(\d+)\]\[(\d+),(\d+)\]', bounds_str)
    if match:
        x1, y1, x2, y2 = map(int, match.groups())
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        return center_x, center_y
    return None


def find_element_coordinates(xml_file, keyword):
    """
    Parses the XML file to find the first element whose content-desc OR text
    contains the keyword (case-insensitive).
    Returns (x, y) coordinates of the center, or (None, None).
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        for node in root.iter():
            desc = node.attrib.get('content-desc', '')
            text = node.attrib.get('text', '')
            
            # Check content-desc
            if desc and keyword.lower() in desc.lower():
                bounds = node.attrib.get('bounds', '')
                if bounds:
                    return parse_bounds(bounds)
            
            # Check text
            if text and keyword.lower() in text.lower():
                bounds = node.attrib.get('bounds', '')
                if bounds:
                    return parse_bounds(bounds)
                    
    except Exception as e:
        print(f"Error parsing XML: {e}")
    return None, None



def find_icon(
    screenshot_path,
    template_path,
    approx_x=None,
    approx_y=None,
    margin_x=100,
    margin_y=100,
    min_matches=10,
    threshold=0.8,
    scales=[0.9, 1.0, 1.1],
):
    img = cv2.imread(screenshot_path, cv2.IMREAD_COLOR)
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)

    if img is None:
        print("Error: Could not load screenshot.")
        return None, None

    if template is None:
        print("Error: Could not load template.")
        return None, None

    if approx_x is not None and approx_y is not None:
        H, W = img.shape[:2]
        x_start = max(0, approx_x - margin_x)
        y_start = max(0, approx_y - margin_y)
        x_end = min(W, approx_x + margin_x)
        y_end = min(H, approx_y + margin_y)
        cropped_img = img[y_start:y_end, x_start:x_end]
        offset_x, offset_y = x_start, y_start
    else:
        cropped_img = img
        offset_x, offset_y = 0, 0

    scene_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(template_gray, None)
    kp2, des2 = orb.detectAndCompute(scene_gray, None)

    if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda m: m.distance)

        if len(matches) > min_matches:
            # Compute homography
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(
                -1, 1, 2
            )
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(
                -1, 1, 2
            )
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is not None:
                h_t, w_t = template_gray.shape
                pts = np.float32([[0, 0], [w_t, 0], [w_t, h_t], [0, h_t]]).reshape(
                    -1, 1, 2
                )
                dst_corners = cv2.perspectiveTransform(pts, M)

                center_x_cropped = int(np.mean(dst_corners[:, 0, 0]))
                center_y_cropped = int(np.mean(dst_corners[:, 0, 1]))
                center_x = center_x_cropped + offset_x
                center_y = center_y_cropped + offset_y
                return center_x, center_y

    # Fallback: Multi-Scale Template Matching
    img_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    w_t, h_t = template_gray.shape[::-1]

    for scale in scales:
        resized_template = cv2.resize(
            template_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
        res = cv2.matchTemplate(img_gray, resized_template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)

        if len(loc[0]) != 0:
            top_left = (loc[1][0], loc[0][0])
            tw, th = resized_template.shape[::-1]
            center_x_cropped = top_left[0] + tw // 2
            center_y_cropped = top_left[1] + th // 2
            center_x = center_x_cropped + offset_x
            center_y = center_y_cropped + offset_y
            return center_x, center_y

    # If no match found
    return None, None


def type_text_slow(device, text, per_char_delay=0.1):
    """
    Simulates typing text character by character.
    Slower, but you can see it appear on screen.
    """
    for char in text:
        # Handle space character, since 'input text " "' can be problematic
        # '%s' is recognized as a space by ADB shell.
        if char == " ":
            char = "%s"
        # You may also need to handle special characters or quotes
        device.shell(f"input text {char}")
        time.sleep(per_char_delay)


# Use to connect directly
def connect_device(user_ip_address="127.0.0.1"):
    adb = AdbClient(host=user_ip_address, port=5037)
    devices = adb.devices()

    if len(devices) == 0:
        print("No devices connected")
        return None
    device = devices[0]
    print(f"Connected to {device.serial}")
    return device


# Use to connect remotely (works for wireless ADB connections from Docker)
def connect_device_remote(device_ip="192.168.0.176:43553"):
    """
    Connect to a wireless ADB device from Docker container.
    device_ip should be in format "IP:PORT" (e.g., "192.168.0.176:43553")
    Will retry multiple times and prompt user to connect device.
    """
    import time
    
    # Parse IP and port from device_ip
    if ":" in device_ip:
        ip, port = device_ip.rsplit(":", 1)
        port = int(port)
    else:
        ip = device_ip
        port = 5555
    
    # Connect to HOST's ADB server via host.docker.internal
    adb = AdbClient(host="host.docker.internal", port=5037)
    
    max_retries = 5
    retry_delay = 10  # seconds
    
    for attempt in range(max_retries):
        devices = adb.devices()
        
        if len(devices) > 0:
            device = devices[0]
            print(f"Connected to {device.serial}")
            return device
        
        # No device found - prompt user
        print(f"\n{'='*60}")
        print(f"NO DEVICE CONNECTED (Attempt {attempt + 1}/{max_retries})")
        print(f"{'='*60}")
        print(f"\nPlease run this command on your HOST machine (not in Docker):")
        print(f"\n    adb connect {device_ip}")
        print(f"\nWaiting {retry_delay} seconds before retrying...")
        print(f"{'='*60}\n")
        
        if attempt < max_retries - 1:
            time.sleep(retry_delay)
    
    print("\n" + "="*60)
    print("FAILED: Could not connect to device after multiple attempts.")
    print("Make sure:")
    print("  1. Your phone has USB debugging enabled")
    print("  2. Wireless debugging is enabled (Settings > Developer Options)")
    print("  3. Run 'adb connect YOUR_IP:PORT' on your host machine")
    print("="*60 + "\n")
    return None


def capture_screenshot(device, filename):
    result = device.screencap()
    with open("images/" + str(filename) + ".png", "wb") as fp:
        fp.write(result)
    return "images/" + str(filename) + ".png"


def tap(device, x, y):
    device.shell(f"input tap {x} {y}")


def press_back(device):
    """
    Simulates pressing the BACK button on Android (Keycode 4).
    Useful for hiding the keyboard.
    """
    device.shell("input keyevent 4")
    time.sleep(1)


def input_text(device, text):
    # input text requires replacing space with %s
    text = text.replace(" ", "%s")
    # Escape single quotes and other special chars for shell
    text = text.replace("'", r"\'")
    text = text.replace('"', r'\"')
    text = text.replace('!', r'\!')
    print("text to be written: ", text)
    device.shell(f'input text "{text}"')
    # Optional: Press back to hide keyboard? No, let the main loop decide.


def swipe(device, x1, y1, x2, y2, duration=500):
    device.shell(f"input swipe {x1} {y1} {x2} {y2} {duration}")


def extract_text_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text


def do_comparision(profile_image, sample_images):
    """
    Returns an average distance score for the best match among the sample_images.
    A lower score indicates a better match.
    If no matches found, returns a high value (indicating poor match).
    """
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(profile_image, None)
    if des1 is None or len(des1) == 0:
        return float("inf")  # No features in profile image

    best_score = float("inf")
    for sample_image in sample_images:
        kp2, des2 = orb.detectAndCompute(sample_image, None)
        if des2 is None or len(des2) == 0:
            continue
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        if len(matches) == 0:
            continue

        matches = sorted(matches, key=lambda x: x.distance)
        score = sum([match.distance for match in matches]) / len(matches)
        if score < best_score:
            best_score = score

    return best_score if best_score != float("inf") else float("inf")


def generate_comment(profile_text):
    """
    Generate a Hinge opening message using algorithm-aware strategy.
    Optimizes for reply probability while maintaining authentic personality.
    Raises CommentGenerationError if all models fail.
    """
    
    system_prompt = """You are an AI agent sending first messages on Hinge for a 24-year-old man.

Your goal is to maximise reply likelihood without misrepresenting personality, then filter for depth over time.

**Core Personality**
- Calm, intelligent, emotionally steady
- Observant, not performative
- Dry, understated wit
- Selective attention
- Curious without chasing
- Serious intent without heaviness

**Market Position**
- Age 24, but emotionally more mature than peers
- Signals long-term orientation subtly
- Avoids appearing intense, heavy, or analytical early
- Lets depth emerge progressively

**Messaging Rules**
- One idea per message
- Short sentences
- No emojis
- No exclamation points
- Curiosity > commentary
- ≤ 140 characters ideal

**Opening Message Strategy**
1. Notice something specific
2. Interpret lightly (not deeply)
3. End without a question OR with a very easy one

**Approved Opener Examples**
- "This profile feels intentional."
- "You come across as calm. That stood out."
- "This feels refreshingly unforced."
- "Quietly interesting profile."

**Fail-Safe Rule**
If unsure: Say less. Be warmer. Let her lean in."""

    user_prompt = f"""Based on this profile, write ONE short opening message (under 140 chars).
Notice something specific. Be warm, not clever. Low cognitive load.

Profile:
{profile_text}

Your opener (one line only):"""
    
    # Try gpt-4o first (more accessible), then gpt-3.5-turbo as fallback
    models_to_try = ["gpt-4o", "gpt-3.5-turbo"]
    
    for model in models_to_try:
        try:
            print(f"Trying to generate comment with {model}...")
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=60,
                temperature=0.7,
            )
            comment = response.choices[0].message["content"].strip()
            # Remove any quotes if the model adds them
            comment = comment.replace('"', '').replace("'", "")
            # Ensure it's short
            if len(comment) > 150:
                comment = comment[:147] + "..."
            print(f"Generated comment using {model}: {comment}")
            return comment
        except Exception as e:
            print(f"Error with {model}: {e}")
            continue
    
    # If all models fail, raise an exception to stop the bot
    raise Exception("CRITICAL: All AI models failed to generate comment. Check OpenAI API quota/billing.")


def get_screen_resolution(device):
    output = device.shell("wm size")
    print("screen size: ", output)
    resolution = output.strip().split(":")[1].strip()
    width, height = map(int, resolution.split("x"))
    return width, height


def open_hinge(device):
    package_name = "co.match.android.matchhinge"
    device.shell(f"monkey -p {package_name} -c android.intent.category.LAUNCHER 1")
    time.sleep(5)
