# importing OpenCV library
import cv2
import os

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH, 'static/upload/')


def capture_image():
    # initialize the camera
    # If you have multiple cameras connected with
    # the current device, assign a value in cam_port
    # variable according to that
    cam_port = 0
    cam = cv2.VideoCapture(cam_port)

    while True:
        print ("capturing..")
        # reading the input using the camera
        result, image = cam.read()

        # If image will detected without any error,
        # show result
        if result:
            # showing result, it takes frame name and image
            # output
            cv2.imshow("GeeksForGeeks", image)

            # Wait for 'c' key to be pressed to capture the image
            if cv2.waitKey(1) & 0xFF == ord('c'):
                # Generate filename for the captured image
                filename = "GeeksForGeeks.png"
                path_save = os.path.join(UPLOAD_PATH, filename)
                # saving image in local storage
                cv2.imwrite(path_save, image)
                print("Image captured!")
                cam.release()
                cv2.destroyAllWindows()
                return path_save, filename


        # If captured image is corrupted, moving to else part
        else:
            print("No image detected. Please! try again")

    # Release the camera and close OpenCV windows
        

# capture_image()
# Example usage:
# path_save, filename = capture_image()
# print("Path:", path_save)
# print("Filename:", filename)

