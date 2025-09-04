def convert_hsv_to_opencv(h, s, v):
    # Hue conversion: scale from 0-360 to 0-179
    cv_h = int((h / 360) * 179)
    
    # Saturation conversion: scale from 0-100 to 0-255
    cv_s = int((s / 100) * 255)

    # Value conversion: scale from 0-100 to 0-255
    cv_v = int((v / 100) * 255)

    return cv_h, cv_s, cv_v

if __name__ == "__main__":
    import sys
    
    # Check if sufficient arguments are provided
    if len(sys.argv) != 4:
        print("Usage: python hsv_converter.py <H> <S> <V>")
        print("H (0-360), S (0-100), V (0-100)")
        sys.exit(1)

    # Parse command line arguments
    try:
        h = float(sys.argv[1])
        s = float(sys.argv[2])
        v = float(sys.argv[3])

        if not (0 <= h <= 360) or not (0 <= s <= 100) or not (0 <= v <= 100):
            raise ValueError("Values out of range")

        # Convert and display the OpenCV HSV values
        opencv_hsv = convert_hsv_to_opencv(h, s, v)
        print(f"OpenCV HSV: {opencv_hsv}")

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
