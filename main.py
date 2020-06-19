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


def get_image_identifier(image_to_match):
    if not os.path.exists('persist'):
        os.mkdir('persist')
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
            print('Access Granted')
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
def main(autosave):
    time_start = time.time()
    frame = get_image_from_cam()
    if frame is not None:
        image_to_be_matched_encoded = prepare_image(frame)
        uuid = get_image_identifier(image_to_be_matched_encoded)

        if not uuid:
            print('Access Denied!')
            if autosave:
                print('Saving new user')
                store_image_for_reference(image_to_be_matched_encoded)

    print('Took %s seconds' % (int(time.time() - time_start)))
    # print("Unique Identifier : ", uuid)


if __name__ == '__main__':
    main()
