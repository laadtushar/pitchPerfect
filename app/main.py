# app/main.py

import asyncio
import time
import os
import openai
import cv2
import uuid
from dotenv import load_dotenv
from multiprocessing import Process, freeze_support, set_start_method
from ppadb.client import Client as AdbClient

# Import your prompt engine weight updater
from prompt_engine import update_template_weights
from config import OPENAI_API_KEY

# Import your existing helper functions
from helper_functions import (
    connect_device,
    connect_device_remote,
    get_screen_resolution,
    open_hinge,
    swipe,
    capture_screenshot,
    extract_text_from_image,  # If you want to keep your original OCR or unify with text_analyzer
    do_comparision,
    find_icon,
    generate_comment,  # If you're using the advanced prompt_engine, you can rename or unify
    tap,
    input_text,
)

# Import data store logic for success-rate tracking
from data_store import (
    store_generated_comment,
    store_feedback,
    calculate_template_success_rates,
)

openai.api_key = OPENAI_API_KEY


# async def main():
def main():
    # Use wireless device connection
    device_ip = os.getenv("DEVICE_IP", "").strip()
    
    # Check if DEVICE_IP is set
    if not device_ip:
        print("\n" + "="*60)
        print("DEVICE_IP NOT SET")
        print("="*60)
        print("\nTo find your device IP and port:")
        print("  1. On your phone: Settings > Developer Options > Wireless Debugging")
        print("  2. Note the IP address and Port shown")
        print("  3. Add to your .env file: DEVICE_IP=YOUR_IP:PORT")
        print("     Example: DEVICE_IP=192.168.0.176:43553")
        print("\nThen on your HOST machine, run:")
        print("     adb connect YOUR_IP:PORT")
        print("="*60 + "\n")
        raise Exception("DEVICE_IP not configured in .env file")
    
    device = connect_device_remote(device_ip)
    if not device:
        raise Exception("Failed to connect to device. See instructions above.")

    width, height = get_screen_resolution(device)

    # Approximate coordinates based on actual Hinge UI (1080x2340 screen)
    # Heart button is at ~60% height (y=1398), ~85% width (x=918)
    x_select_like_button_approx = int(width * 0.85)
    y_select_like_button_approx = int(height * 0.60)

    x_select_comment_button_approx = 540
    y_select_comment_button_approx = 1755

    x_select_done_button_approx = int(width * 0.85)
    y_select_done_button_approx = int(height * 0.50)

    # Send button is at ~73% height (y=1704), ~75% width
    # Previous 0.80 was hitting "Cancel" button at y=1874
    x_send_like_button = int(width * 0.75)
    y_send_like_button = int(height * 0.73)

    x_dislike_button_approx = int(width * 0.15)
    y_dislike_button_approx = int(height * 0.85)

    x1_swipe = int(width * 0.15)
    x2_swipe = x1_swipe

    y1_swipe = int(height * 0.5)
    y2_swipe = int(y1_swipe * 0.75)

    # Load sample images for matching criteria (like/dislike)
    like_images = [
        cv2.imread(path) for path in ["images/like2.jpeg"] if os.path.exists(path)
    ]
    dislike_images = [
        cv2.imread(path) for path in ["images/dislike.jpeg"] if os.path.exists(path)
    ]

    open_hinge(device=device)
    time.sleep(5)

    previous_profile_text = ""

    # Optionally, run once at the start: recalc success rates & update template weights
    success_rates = calculate_template_success_rates()
    update_template_weights(success_rates)

    # Import new helper functions
    from helper_functions import get_ui_hierarchy, find_element_coordinates, press_back

    for _ in range(10):
        # Swipe to next profile (swipe up)
        swipe(device, x1_swipe, y1_swipe, x2_swipe, y2_swipe)
        time.sleep(1)
        
        # Scroll UP to top of profile so heart button is visible
        # Swipe DOWN (from top to bottom) to scroll up
        swipe(device, int(width * 0.5), int(height * 0.3), int(width * 0.5), int(height * 0.7), duration=300)
        time.sleep(0.5)
        
        screenshot_path = capture_screenshot(device, "screen")


        # OCR for text extraction (or direct from text_analyzer, whichever you prefer)
        current_profile_text = extract_text_from_image(screenshot_path).strip()
        if not current_profile_text:
            print("Warning: OCR returned empty text.")

        profile_image = cv2.imread(screenshot_path)

        # Compare with sample images
        match_like = do_comparision(profile_image, like_images)
        match_dislike = do_comparision(profile_image, dislike_images)

        print("Calculated scores => Like:", match_like, "Dislike:", match_dislike)

        # Smart Navigation: Find Like Button
        x_select_like_button = None
        y_select_like_button = None
        
        # Scroll-to-Find Loop: Try up to 3 times
        max_retries = 3
        for attempt in range(max_retries):
            print(f"Searching for Like button (Attempt {attempt+1}/{max_retries})...")
            xml_file = get_ui_hierarchy(device)
            if xml_file:
                # Search for "Like photo" or just "Like"
                center = find_element_coordinates(xml_file, "Like photo")
                if not center:
                    center = find_element_coordinates(xml_file, "Like")
                
                if center:
                    x_select_like_button, y_select_like_button = center
                    print(f"Found 'Like photo' at {center}")
                    break # Found it!
            
            # If not found, scroll down slightly to reveal it
            if attempt < max_retries - 1:
                print("Like button not found. Scrolling down slightly...")
                # Swipe UP (from bottom to top) to scroll DOWN
                swipe(device, int(width*0.5), int(height*0.7), int(width*0.5), int(height*0.5), duration=300)
                time.sleep(1)

        # Fallback to coordinates if still not found
        if x_select_like_button is None:
            print("Like button not found after scrolling. Using approximate coordinates.")
            x_select_like_button = x_select_like_button_approx
            y_select_like_button = y_select_like_button_approx

        # Decision-making logic - ALWAYS LIKE for now
        # Generate a comment using GPT-4o (will raise exception if API fails)
        comment = generate_comment(current_profile_text)
        print(f"Generated Comment: {comment}")

        # Create a comment_id to track feedback
        comment_id = str(uuid.uuid4())

        # Optionally store the generated comment for analytics
        store_generated_comment(
            comment_id=comment_id,
            profile_text=current_profile_text,
            generated_comment=comment,
            style_used="unknown",
        )

        # Tap Like
        print(f"Tapping Like at {x_select_like_button}, {y_select_like_button}")
        tap(device, x_select_like_button, y_select_like_button)
        
        # Wait for "Add a comment" dialog to appear
        time.sleep(2)

        # Smart Navigation: Find Comment Box and Send Button
        print("Searching for Comment Dialog elements...")
        xml_file_dialog = get_ui_hierarchy(device)
        
        x_comment_box = 540
        y_comment_box = int(height * 0.65)
        
        x_send_btn = x_send_like_button
        y_send_btn = y_send_like_button

        if xml_file_dialog:
            # Try to find comment input
            center_box = find_element_coordinates(xml_file_dialog, "Add a comment")
            if center_box:
                x_comment_box, y_comment_box = center_box
                print(f"Found 'Add a comment' box at {center_box}")
            
            # Try to find Send button (Priority Like, Rose, or specific text)
            # In dialog dump we saw 'Send priority like' and 'Send a Rose'
            # We try a few distinct keywords
            center_send = None
            for key in ["Send priority like", "Send like", "Send a Rose", "Send"]:
                center_send = find_element_coordinates(xml_file_dialog, key)
                if center_send:
                    print(f"Found Send button ('{key}') at {center_send}")
                    break
            
            if center_send:
                x_send_btn, y_send_btn = center_send

        # Tap the "Add a comment" text field to focus it
        print(f"Tapping Comment Box at {x_comment_box}, {y_comment_box}")
        tap(device, x_comment_box, y_comment_box)
        time.sleep(1)

        # Input the comment
        print(f"Typing comment: {comment}")
        input_text(device, comment)
        time.sleep(1)
        
        # KEYBOARD HANDLING: Press Back to hide keyboard so Send button is visible
        print("Hiding keyboard...")
        press_back(device)
        time.sleep(1)

        # Re-fetch UI hierarchy after keyboard is hidden
        print("Searching for Send button after hiding keyboard...")
        xml_file_dialog_send = get_ui_hierarchy(device)
        
        if xml_file_dialog_send:
            # Try to find Send button (Priority Like, Rose, or specific text)
            center_send = None
            for key in ["Send priority like", "Send like", "Send a Rose", "Send"]:
                center_send = find_element_coordinates(xml_file_dialog_send, key)
                if center_send:
                    print(f"Found Send button ('{key}') at {center_send}")
                    break
            
            if center_send:
                x_send_btn, y_send_btn = center_send
                # Tap Send
                print(f"Tapping Send at {x_send_btn}, {y_send_btn}")
                tap(device, x_send_btn, y_send_btn)
            else:
                print("Send button NOT found. Tapping Cancel to exit dialog.")
                # Try to find Cancel button
                center_cancel = find_element_coordinates(xml_file_dialog_send, "Cancel")
                if center_cancel:
                    tap(device, center_cancel[0], center_cancel[1])
                else:
                    # Generic cancel location if not found (bottom center-ish)
                    tap(device, 540, 1870) 

        else:
            print("Could not dump UI for dialog. Tapping Send approx.")
            tap(device, x_send_btn, y_send_btn)

        previous_profile_text = current_profile_text
        time.sleep(3)  # Wait for animation to finish
    
    print("Main loop finished.")

    # After processing 10 profiles, re-check success rates, update template weights
    success_rates = calculate_template_success_rates()
    update_template_weights(success_rates)
    print("Final success rates:", success_rates)
    print("Main loop finished.")


def test():
    height = 1080
    width = 2340
    device = connect_device()
    comment = "Hi"

    x_select_comment_button_approx = 540
    y_select_comment_button_approx = 1755

    swipe(device, width * 0.50, height * 0.70, width * 0.55, height * 0.70)
    tap(device, x_select_comment_button_approx, y_select_comment_button_approx)
    # Type the comment
    input_text(device, comment)


if __name__ == "__main__":
    # async run of main
    # Windows fix for multiprocessing
    # freeze_support()
    # set_start_method("spawn", force=True)
    # asyncio.run(main())

    # Test for checking only input text
    # test()

    main()
