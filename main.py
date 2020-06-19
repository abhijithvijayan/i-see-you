import os
import cv2
import uuid
import time
import click
import pickle
import face_recognition


def prepare_image(cameraImage):
    image_encoded = face_recognition.face_encodings(cameraImage)

    if(len(image_encoded) == 0):
        print("Error : Face features cannot be extracted from this file.")
        exit(1)

    return image_encoded[0]


def store_image_for_reference(image):
    uid = uuid.uuid4().hex

    db_file = open('persist/' + uid, 'ab')
    pickle.dump(image, db_file)

    return uid


def create_out_dir():
    os.mkdir('persist')


# returns boolean
def out_dir_exist():
    return os.path.exists('persist')


def get_image_identifier(image_to_match):
    if not out_dir_exist():
        create_out_dir()
        return None

    images = os.listdir('persist')
    """
    iterate through all the files and try matching
    If no matches, store as new
    """
    for image in images:
        image_meta_file = open('persist/' + image, 'rb')
        existing_encoded_image = pickle.load(image_meta_file)

        result = face_recognition.compare_faces(
            [image_to_match], existing_encoded_image)
        if result[0] == True:
            return image

    return None


def get_image_from_cam():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Press Space to capture")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Unable to open video capture.")
            break
        cv2.imshow("Press Space to capture", frame)

        k = cv2.waitKey(1)
        if k % 256 == 32:
            cam.release()
            print("Image Captured...")
            return frame


@click.command()
@click.option("--autosave", default=False, help="Auto save new user.")
@click.option("--listdir", default=False, help="List all existing users.")
@click.option("--showid", default=False, help="Show ID of authenticated user on success.")
@click.option("--image", default=None, help="Authenticate from an image instead of webcam")
def main(autosave, listdir, showid, image):
    time_start = time.time()

    if listdir:
        if not out_dir_exist():
            create_out_dir()

        images = os.listdir('persist')
        if len(images) <= 0:
            print('Empty Directory!')
        else:
            for _image in images:
                print(_image)
    else:
        frame = None
        if image is not None:
            frame = face_recognition.load_image_file("images/{}".format(image))
        else:
            frame = get_image_from_cam()

        if frame is not None:
            image_to_be_matched_encoded = prepare_image(frame)
            uuid = get_image_identifier(image_to_be_matched_encoded)

            if not uuid:
                print('Access Denied!')
                if autosave:
                    print('Saving new user')
                    uuid = store_image_for_reference(
                        image_to_be_matched_encoded)
                    if showid:
                        print("Unique Identifier: ", uuid)
            else:
                print('Access Granted')
                if showid:
                    print("Unique Identifier: ", uuid)

    print('Took %s seconds' % (int(time.time() - time_start)))


if __name__ == '__main__':
    main()
