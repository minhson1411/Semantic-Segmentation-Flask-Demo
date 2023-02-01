from flask import Flask, render_template, send_from_directory, url_for
from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
from inference import SegmentModel
#Load model
sg = SegmentModel()

#Define flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'gnvml'
app.config['UPLOADED_PHOTOS_DEST'] = 'uploads'

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)


class UploadForm(FlaskForm):
    photo = FileField(
        validators=[
            FileAllowed(photos, 'Only images are allowed'),
            FileRequired('File field should not be empty')
        ]
    )
    submit = SubmitField('Upload')

@app.route('/uploads/<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOADED_PHOTOS_DEST'], filename)


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    form = UploadForm()
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_url = url_for('get_file', filename=filename)
        seg_image, label = sg.get_prediction(image_path=file_url[1:])
        #Save segmentation image
        seg_image.save(file_url[1:])
    else:
        file_url = None
        label = None


    return render_template('index.html', form=form, file_url=file_url, label = label)

if __name__=='__main__':
    app.run(host="0.0.0.0", port = 5000, debug=True)