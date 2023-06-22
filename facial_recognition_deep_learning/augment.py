import os
import cv2

def augment_images(folder_path):
    # Get a list of subfolders within the given folder path
    subfolders = [subfolder for subfolder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, subfolder))]

    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_path, subfolder)
        images = [image for image in os.listdir(subfolder_path) if image.endswith(('.jpg', '.jpeg', '.png'))]

        for image_name in images:
            image_path = os.path.join(subfolder_path, image_name)

            # Load the image
            image = cv2.imread(image_path)

            # Perform rotation
            angle = 45
            rows, cols, _ = image.shape
            rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))

            # Save the rotated image
            rotated_image_path = os.path.join(subfolder_path, f"rotated_{image_name}")
            cv2.imwrite(rotated_image_path, rotated_image)

            # Perform horizontal flip
            flipped_image = cv2.flip(image, 1)

            # Save the flipped image
            flipped_image_path = os.path.join(subfolder_path, f"flipped_{image_name}")
            cv2.imwrite(flipped_image_path, flipped_image)

            print(f"Augmented and saved {image_name}")

        print(f"Augmentation complete for folder: {subfolder_path}")

    print("Augmentation complete for all folders.")

# Example usage
folder_path = os.path.join('data')
augment_images(folder_path)
