import cv2
import os

def make_video_from_images(image_folder, output_video_file, fps=30):
    # Get list of image files from the folder
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
    image_files.sort()  # Ensure the images are processed in sorted order

    # throw out first frame because it is sized wrong
    image_files = image_files[1:]

    if not image_files:
        raise ValueError("No PNG images found in the specified folder.")

    # Read the ~~first~~ second image to get the width and height
    first_image_path = os.path.join(image_folder, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, layers = first_image.shape

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 file
    video = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))

    # Write each image to the video
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        img = cv2.imread(image_path)
        video.write(img)

    # Release the VideoWriter object
    cv2.destroyAllWindows()
    video.release()

    print(f"Video saved as {output_video_file}")

# Example usage
image_folder = 'results_1/'
output_video_file = 'results_1.mp4'
make_video_from_images(image_folder, output_video_file, fps=4)
# 1 uses 3 fps
# 3 uses 4 fps
# 5 uses 2 fps
