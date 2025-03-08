import cv2
import numpy as np

def detect_shapes(image_path):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not open or read image: {image_path}")
        return
    
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use Canny edge detection to find edges
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours from the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through the contours and detect shapes
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

        if len(approx) == 3:
            shape = "Triangle"
            cv2.drawContours(img, [approx], -1, (0, 255, 0), 3)
        elif len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.95 <= aspect_ratio <= 1.05:
                shape = "Square"
                cv2.drawContours(img, [approx], -1, (0, 0, 255), 3)
            else:
                shape = "Rectangle"
                cv2.drawContours(img, [approx], -1, (255, 0, 0), 3)
        elif len(approx) == 5:
            shape = "Pentagon"
            cv2.drawContours(img, [approx], -1, (255, 255, 0), 3)
        elif len(approx) > 5:
            shape = "Circle"
            cv2.drawContours(img, [approx], -1, (0, 255, 255), 3)
        else:
            shape = "Undefined"
            continue

        # Add text label for each shape detected
        x, y = approx.ravel()[0], approx.ravel()[1]
        cv2.putText(img, shape, (x - 50, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Display the output image in a new window
    cv2.imshow('Detected Shapes', img)

    # Wait for a key press to close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Provide the image path to the function
detect_shapes('images.png')  # Replace with your local image file path
