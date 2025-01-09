import cv2
import numpy as np
import os


def create_mask(image_path, output_path = None, is_element = True):
    """
    Sets the transparent background of images with a transparent background to white, and other areas to black.
    
    :param image_path: Input image path
    :param output_path: Output image path
    """
    # Load the image, keeping the Alpha channel
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    # Check if there is an Alpha channel
    # print(image.shape)
    if len(image.shape) == 3 and image.shape[2] == 4:
        # Extract the Alpha channel
        alpha_channel = image[:, :, 3]
        
        # If it's not an element, print debug info
        #if is_element == False:
            #print('check 1', image.shape)
        
        # Create a white image (255 is white)
        # white_background = np.ones_like(alpha_channel) * 255
        
        # Set transparent areas to white, others to black
        if is_element == True:
            # mask = cv2.bitwise_not(alpha_channel)  # Invert the Alpha channel
            mask = np.where(alpha_channel == 0, 255, 0).astype(np.uint8)
        else:
            #print('=======>', alpha_channel.shape)
            mask = np.where(alpha_channel == 0, 0, 255).astype(np.uint8)
        # result = cv2.merge((mask, mask, mask, white_background))
    elif len(image.shape) == 3 and image.shape[2] == 3:
        # For three-channel input, define transparent areas as pure black (0,0,0)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = np.where(gray_image == 0, 255 if is_element else 0, 0 if is_element else 255).astype(np.uint8)
    elif len(image.shape) == 2:
        mask = np.where(image == 0, 255 if is_element else 0, 0 if is_element else 255).astype(np.uint8)
        
    else:
        raise ValueError("Input image must be either 1 or 3 or 4 channels")
    # Save the result
    # cv2.imwrite(output_path, mask)

    return mask


def check_black_overlap(image1, image2, rotate=None, x=0, y=0):
    """ 
    Checks for overlapping black areas in two black and white images, with translation and rotation applied to the second image
    
    :param image1: First black and white image in numpy array format
    :param image2: Second black and white image in numpy array format
    :param rotate: Rotation angle (in degrees) for the second image
    :param x: X-coordinate of the second image center relative to the first image's top-left corner
    :param y: Y-coordinate of the second image center relative to the first image's top-left corner
    :return: 1 if overlap exists, 0 if no overlap
    """
    # Ensure the input images are single-channel
    if len(image1.shape) != 2 or len(image2.shape) != 2:
        raise ValueError("input image must be black and white!")

    # Apply rotation to the second image if specified
    if rotate is not None:
        # Get the size of the second image
        h2, w2 = image2.shape
        
        # Calculate the bounding box size after rotation
        angle_rad = np.deg2rad(rotate)
        new_w = int(abs(w2 * np.cos(angle_rad)) + abs(h2 * np.sin(angle_rad)))
        new_h = int(abs(h2 * np.cos(angle_rad)) + abs(w2 * np.sin(angle_rad)))

        # Create a large enough canvas to fit the rotated image
        padded_image2 = cv2.copyMakeBorder(image2, 
                                           (new_h - h2) // 2, (new_h - h2 + 1) // 2, 
                                           (new_w - w2) // 2, (new_w - w2 + 1) // 2, 
                                           cv2.BORDER_CONSTANT, value=255)

        # Get the center of the new image
        h2_padded, w2_padded = padded_image2.shape
        center = (w2_padded // 2, h2_padded // 2)

        # Generate rotation matrix and rotate the image
        rotation_matrix = cv2.getRotationMatrix2D(center, rotate, 1.0)
        image2 = cv2.warpAffine(padded_image2, rotation_matrix, (w2_padded, h2_padded), borderValue=255)

    # Get the size of both images
    h1, w1 = image1.shape
    h2, w2 = image2.shape

    # Find the maximum size
    max_height = max(h1, h2)
    max_width = max(w1, w2)

    # Pad both images to the same size with white color (255)
    image1_padded = cv2.copyMakeBorder(image1, 0, max_height - h1, 0, max_width - w1, cv2.BORDER_CONSTANT, value=255)
    image2_padded = cv2.copyMakeBorder(image2, 0, max_height - h2, 0, max_width - w2, cv2.BORDER_CONSTANT, value=255)

    # Calculate the translation required for image2 based on x, y coordinates
    x_shift = x - w2 // 2  # Translate based on the center of image2
    y_shift = y - h2 // 2  # Translate based on the center of image2

    # Create a translation matrix
    translation_matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])

    # Apply the translation to the second image
    translated_image2 = cv2.warpAffine(image2_padded, translation_matrix, (max_width, max_height), borderValue=255)

    # Find the black areas (pixel value 0) overlap between both images
    mask1 = (image1_padded == 0).astype(np.uint8)
    mask2 = (translated_image2 == 0).astype(np.uint8)
    overlap = cv2.bitwise_and(mask1, mask2)

    # Check if there is any black pixel in the overlap region
    if np.any(overlap):  # If there is at least one True pixel (i.e. overlap)
        return 1
    else:
        return 0

def euclidean_distance(coord1, coord2):
    """
    Computes the Euclidean distance between two n-dimensional coordinates

    Parameters:
    coord1 (ndarray): The first coordinate (n-dimensional array)
    coord2 (ndarray): The second coordinate (n-dimensional array)

    Returns:
    float: The Euclidean distance between the two coordinates
    """
    coord1 = np.array(coord1)
    coord2 = np.array(coord2)
    distance = np.linalg.norm(coord1 - coord2)  # Using NumPy to compute the Euclidean distance

    return distance

def interactive_canny(image_name):
    """
    Interactive window to adjust Canny edge detection parameters in real-time
    
    Parameters:
        image_name (str): Name of the image file in the current directory
    """
    def nothing(x):
        pass

    # Read image
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, image_name)
    
    if not os.path.exists(image_path):
        print(f"Error: Image '{image_name}' not found in current directory")
        return
    
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Create window with trackbars
    cv2.namedWindow('Canny Edge Detection')
    cv2.createTrackbar('Low Threshold', 'Canny Edge Detection', 0, 255, nothing)
    cv2.createTrackbar('Hh Threshold', 'Canny Edge Detection', 0, 255, nothing)

    while True:
        # Get current positions of trackbars
        low = cv2.getTrackbarPos('Low Threshold', 'Canny Edge Detection')
        high = cv2.getTrackbarPos('Hh Threshold', 'Canny Edge Detection')

        # Apply Canny edge detection
        edges = cv2.Canny(gray, low, high)
        
        # Show the image
        cv2.imshow('Canny Edge Detection', edges)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Save the final result
            output_name = 'canny_' + image_name
            output_path = os.path.join(current_dir, output_name)
            cv2.imwrite(output_path, edges)
            break

    cv2.destroyAllWindows()

import cv2
import numpy as np

def visualize_image_transform(image1_path, image2_path, center_x, center_y, rotation_angle, output_path):
    # Load the first image
    image1 = cv2.imread(image1_path, cv2.IMREAD_UNCHANGED)
    if image1 is None:
        raise FileNotFoundError(f"Unable to load image: {image1_path}")

    # Load the second image
    image2 = cv2.imread(image2_path, cv2.IMREAD_UNCHANGED)
    if image2 is None:
        raise FileNotFoundError(f"Unable to load image: {image2_path}")

    # If the first image is single-channel or two-channel, convert to 4 channels (with alpha)
    if len(image1.shape) == 2:  # Single-channel grayscale image
        alpha_channel = np.ones((image1.shape[0], image1.shape[1], 1), dtype=np.uint8) * 255
        image1 = cv2.merge((image1, image1, image1, alpha_channel))
    elif image1.shape[2] == 1:  # Two-channel (assumed grayscale+alpha)
        gray = image1[:, :, 0]
        alpha_channel = image1[:, :, 1]
        image1 = cv2.merge((gray, gray, gray, alpha_channel))
    elif image1.shape[2] == 3:  # Three-channel
        alpha_channel = np.ones((image1.shape[0], image1.shape[1], 1), dtype=np.uint8) * 255
        image1 = np.concatenate((image1, alpha_channel), axis=2)

    # Ensure the second image has an alpha channel
    if image2.shape[2] != 4:
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2BGRA)

    # Get the size of the second image
    h2, w2 = image2.shape[:2]

    # Calculate the bounding box size after rotation
    angle_rad = np.deg2rad(rotation_angle)
    new_w = int(abs(w2 * np.cos(angle_rad)) + abs(h2 * np.sin(angle_rad)))
    new_h = int(abs(h2 * np.cos(angle_rad)) + abs(w2 * np.sin(angle_rad)))

    # Ensure non-negative padding sizes
    padding_top = max(0, (new_h - h2) // 2)
    padding_bottom = max(0, (new_h - h2 + 1) // 2)
    padding_left = max(0, (new_w - w2) // 2)
    padding_right = max(0, (new_w - w2 + 1) // 2)

    # Create a large enough canvas to fit the rotated image
    padded_image2 = cv2.copyMakeBorder(image2, 
                                       padding_top, padding_bottom, 
                                       padding_left, padding_right, 
                                       cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))

    # Get the center of the padded image
    h2_padded, w2_padded = padded_image2.shape[:2]
    center = (w2_padded // 2, h2_padded // 2)

    # Get the rotation matrix and rotate the image
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    rotated_image = cv2.warpAffine(padded_image2, rotation_matrix, (w2_padded, h2_padded), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    # Get the bounding box of the non-transparent areas in the rotated image
    non_transparent_area = np.where(rotated_image[:, :, 3] > 0)
    min_x = min(non_transparent_area[1])
    max_x = max(non_transparent_area[1])
    min_y = min(non_transparent_area[0])
    max_y = max(non_transparent_area[0])

    # Crop the rotated image to its non-transparent area
    cropped_rotated_image = rotated_image[min_y:max_y+1, min_x:max_x+1]

    # Get the size of the cropped rotated image
    rotated_h, rotated_w = cropped_rotated_image.shape[:2]

    # Calculate the final canvas size
    h1, w1 = image1.shape[:2]
    x_min = min(0, center_x - rotated_w // 2)
    y_min = min(0, center_y - rotated_h // 2)
    x_max = max(w1, center_x + rotated_w // 2)
    y_max = max(h1, center_y + rotated_h // 2)

    new_width = x_max - x_min
    new_height = y_max - y_min

    # Create a new blank image with a transparent background
    result_image = np.zeros((new_height, new_width, 4), dtype=np.uint8)

    # Paste the first image into the result image
    result_image[-y_min:h1 - y_min, -x_min:w1 - x_min] = image1

    # Calculate the position of the cropped rotated image
    x_pos = center_x - rotated_w // 2 - x_min
    y_pos = center_y - rotated_h // 2 - y_min

    # Paste the cropped rotated image into the result image
    for i in range(rotated_h):
        for j in range(rotated_w):
            if 0 <= i + y_pos < new_height and 0 <= j + x_pos < new_width:
                if cropped_rotated_image[i, j][3] > 0:  # Check the alpha channel
                    result_image[i + y_pos, j + x_pos] = cropped_rotated_image[i, j]

    # Save the result image
    output_full_path = f"output_visualize/test_visualization_{output_path}.png"
    cv2.imwrite(output_full_path, result_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    print(f"Result image saved at {output_full_path}")

def get_image_center(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    if image is None:
        raise FileNotFoundError(f"Unable to load image: {image_path}")
    
    # Get the height and width of the image
    height, width = image.shape[:2]
    
    # Calculate the center coordinates
    center_x = width // 2
    center_y = height // 2
    
    return center_x, center_y

def get_image_centers(image_paths):
    centers = []  # Used to store center coordinates

    # Iterate through each image path
    for image_path in image_paths:
        # Load the image
        image = cv2.imread(image_path)
        
        if image is None:
            raise FileNotFoundError(f"Unable to load image: {image_path}")
        
        # Get the height and width of the image
        height, width = image.shape[:2]
        
        # Calculate the center coordinates
        center_x = width // 2
        center_y = height // 2
        
        # Add the center coordinates to the list
        centers.append([center_x, center_y])
    
    # Convert the list to a numpy array and return
    return np.array(centers)