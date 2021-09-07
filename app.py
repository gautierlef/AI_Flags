import os
import requests
from PIL import ExifTags
import streamlit as st

REPO_DIR = 'https://github.com/Maathess/Projet_annuel_flag'
MODEL_FILE = 'dogs_online_resnet50_cpu.pkl'

#st.set_page_config(page_title="Flag Predictor - Maathess", page_icon="🏴")

st.write("# Flag predictor")
st.markdown('By <a href="https://github.com/Maathess/Projet_annuel_flag" target="_blank">Maathess</a>', unsafe_allow_html=True)

st.write("This project is about predicting flag using trained MLP model")

file_data = st.file_uploader("Select an image", type=["jpg"])


def download_file(url):
    with st.spinner('Downloading model...'):
        # from https://stackoverflow.com/a/16696317
        local_filename = url.split('/')[-1]
        # NOTE the stream=True parameter below
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    # If you have chunk encoded response uncomment if
                    # and set chunk_size parameter to None.
                    # if chunk:
                    f.write(chunk)
        return local_filename


def fix_rotation(file_data):
    # check EXIF data to see if has rotation data from iOS. If so, fix it.
    try:
        image = PILImage.create(file_data)
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        exif = dict(image.getexif().items())

        rot = 0
        if exif[orientation] == 3:
            rot = 180
        elif exif[orientation] == 6:
            rot = 270
        elif exif[orientation] == 8:
            rot = 90

        if rot != 0:
            st.write(f"Rotating image {rot} degrees (you're probably on iOS)...")
            image = image.rotate(rot, expand=True)
            # This step is necessary because image.rotate returns a PIL.Image, not PILImage, the fastai derived class.
            image.__class__ = PILImage

    except (AttributeError, KeyError, IndexError):
        pass  # image didn't have EXIF data

    return image


# cache the model so it only gets loaded once
@st.cache(allow_output_mutation=True)
def get_model():
    if not os.path.isfile(MODEL_FILE):
        _ = download_file(f'{REPO_DIR}/models/{MODEL_FILE}')

    learn = load_learner(MODEL_FILE)
    return learn


learn = get_model()

if file_data is not None:
    with st.spinner('Classifying...'):
        # load the image from uploader; fix rotation for iOS devices if necessary
        img = fix_rotation(file_data)

        st.write('## Your Image')
        st.image(img, width=200)

        # classify
        pred, pred_idx, probs = learn.predict(img)
        top5_preds = sorted(list(zip(learn.dls.vocab, list(probs.numpy()))), key=lambda x: x[1], reverse=True)[:5]

        # prepare output
        out_text = '<table><tr> <th>Breed</th> <th>Confidence</th> <th>Example</th> </tr>'

        for pred in top5_preds:
            example = REPO_DIR + '/example_dogs/' + pred[0].replace(" ", "").lower() + ".jpg"
            out_text += '<tr>' + \
                        f'<td>{pred[0]}</td>' + \
                        f'<td>{100 * pred[1]:.02f}%</td>' + \
                        f'<td><img src="{example}" height="150" /></td>' + \
                        '</tr>'
        out_text += '</table><br><br>'

        st.write('## What the model thinks')
        st.markdown(out_text, unsafe_allow_html=True)