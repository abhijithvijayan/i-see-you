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
        return None

    return image_encoded[0]


def store_image_for_reference(image, username):
    # generate a uuid for the new image
    uid = uuid.uuid4().hex
    # `uuid_username` format
    filename = "{0}_{1}".format(uid, username.replace(
        'Default: ', '').replace(' ', '%'))

    out_file = open('persist/' + filename, 'ab')
    pickle.dump(image, out_file)

    return filename


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
        # open image in binary format
        image_meta_file = open('persist/' + image, 'rb')
        existing_encoded_image = pickle.load(image_meta_file)

        # compare with existing encoded
        result = face_recognition.compare_faces(
            [image_to_match], existing_encoded_image)

        # match is found
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


def show_user_info(user):
    person = user.split('_')
    uuid = person[0]
    username = person[1].replace('%', ' ')
    print("Username: {}".format(username))
    print("Unique Identifier: {}".format(uuid))


def authenticate(autosave, showid, image=None):
    if image is not None:
        # get the encoded version
        image_to_be_matched_encoded = prepare_image(image)
        # encoding successful
        if image_to_be_matched_encoded is not None:
            user = get_image_identifier(image_to_be_matched_encoded)

            if not user:
                print('---- Access Denied! ----')
                if autosave:
                    print('Saving new user...')
                    # prompt to get a username
                    username = click.prompt(
                        'Enter username: ', default='Default: No username', type=str)
                    user = store_image_for_reference(
                        image_to_be_matched_encoded, username)
                    if showid:
                        show_user_info(user)
            else:
                print('---- Access Granted ----')
                if showid:
                    show_user_info(user)
        print()


@click.command()
@click.option("--autosave", default=False, help="Auto save new user.")
@click.option("--listdir", default=False, help="List all existing users.")
@click.option("--showid", default=False, help="Show ID of authenticated user on success.")
@click.option("--image", default=None, help="Authenticate from an image instead of webcam")
@click.option("--multiple", default=False, help="Try with multiple images in folder")
def main(autosave, listdir, showid, image, multiple):
    time_start = time.time()

    if listdir:
        if not out_dir_exist():
            create_out_dir()

        images = os.listdir('persist')
        if len(images) <= 0:
            print('Empty Directory!')
        else:
            for _image in images:
                # Trim only id, ignore `Name` part
                print(_image.split('_')[0])
    else:
        if image is not None:
            # get image from path
            frame = face_recognition.load_image_file("images/{}".format(image))
            authenticate(autosave, showid, frame)
        else:
            if not multiple:
                # get single image from webcam
                frame = get_image_from_cam()
                authenticate(autosave, showid, frame)
            else:
                images = os.listdir('images')
                if len(images) <= 0:
                    print('Empty Directory!')
                else:
                    for _image in images:
                        # load each image and perform authentication
                        print("Using image: {}".format(_image))
                        frame = face_recognition.load_image_file(
                            "images/{}".format(_image))
                        authenticate(autosave, showid, frame)

    print('Took %s seconds' % (int(time.time() - time_start)))


if __name__ == '__main__':
    main()
