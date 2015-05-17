import tornado
from tornado.options import define, options
import tornado.web
import tornado.httpserver
import tornado.ioloop
import os.path
import os
from dataModelsForImageStorage import SkinImage
from ImageProcessor import ImageFilter
from ImageProcessor import ClassifyImages

__SKINIMAGES__ = os.path.join(os.path.dirname(__file__), 'skinImages/')

define("port", default=8000, help="Run the application on the given port", type=int)


class IndexHandler(tornado.web.RequestHandler):
    def get(self, *args, **kwargs):
        return self.render('index.html', title="Home - Final Year Project", message='')

    def post(self, *args, **kwargs):
        if dict(self.request.files).__len__() == 0:
            self.render('index.html', message="No image detected. Please try again", title="Image upload error")
        else:
            skinPicture = self.request.files['skin_image'][0]
            skinPictureName = skinPicture['filename']
            skinPictureExtension = os.path.splitext(skinPictureName)[1]

            #Save Skin Image Information to database
            skinImage = SkinImage(patientAge=self.get_argument('patient_age'),
                                  patientLocation=self.get_argument('patient_location'),
                                  skinConditionDescription=self.get_argument('condition_description'),
                                  skinImageName=skinPictureName,
                                  skinImageExtension=skinPictureExtension)

            inserted_image_id = skinImage.saveSkinImage()
            if inserted_image_id > 0:
                #Create the image folder
                image_path = os.path.join(os.path.dirname(__file__), 'skinImages/' + inserted_image_id.__str__())
                print image_path
                os.makedirs(image_path)

                #Upload the image
                open(__SKINIMAGES__ + inserted_image_id.__str__() + '/' + skinPictureName, 'w').write(
                    skinPicture['body'])
                self.render('account.html', title="Image successfully uploaded", message="Image Upload successfully")
            else:
                self.render('account.html', title="Image information not saved",
                            message="Image information not saved because of an error")


class DiagnoseImageHandler(tornado.web.RequestHandler):
    def get(self, skinImageId):
        #Generate histogram for image and store necessary data to a file
        __HEALTHY_SKIN_TRAINING_DATA_PATH__ = os.path.join(
            os.path.join(os.path.dirname(__file__), 'ImageProcessor/training_data/healthy'))

        __DISEASED_SKIN_TRAINING_DATA_PATH__ = os.path.join(
            os.path.join(os.path.dirname(__file__), 'ImageProcessor/training_data/deseased'))

        __HEALTHY_SKIN_TEST_DATA_PATH__ = os.path.join(
            os.path.join(os.path.dirname(__file__), 'ImageProcessor/test data/healthy'))

        __DISEASED_SKIN_TEST_DATA_PATH__ = os.path.join(
            os.path.join(os.path.dirname(__file__), 'ImageProcessor/test data/deseased'))

        __TO_DIAGNOSE_IMAGE_PATH__ = os.path.join(
            os.path.join(os.path.dirname(__file__), 'skinImages/' + skinImageId.__str__()))

        skin_histogram = ImageFilter.get_histograms(__DISEASED_SKIN_TRAINING_DATA_PATH__,
                                                    __HEALTHY_SKIN_TRAINING_DATA_PATH__, __TO_DIAGNOSE_IMAGE_PATH__)
        data_classes = ImageFilter.define_data_classes(skin_histogram["trainingDiseasedData"],
                                                       skin_histogram["trainingHealthyData"],
                                                       skin_histogram["testDiseasedData"],
                                                       skin_histogram["testHealthyData"])
        concatenated_data = ImageFilter.concatenate_data(skin_histogram["trainingHealthyData"],
                                                         data_classes["trainingHealthyClasses"],
                                                         skin_histogram["trainingDiseasedData"],
                                                         data_classes['trainingDiseasedClasses'],
                                                         skin_histogram['testDiseasedData'],
                                                         data_classes["testDiseasedClasses"],
                                                         skin_histogram['testHealthyData'],
                                                         data_classes["testHealthyClasses"],
                                                         )

        ImageFilter.save_data_to_folder('data', concatenated_data)

        #classify image
        image_classification = ClassifyImages.classify_image(2)
        message = ''
        if image_classification == 1:
            message += "fungal infection Detected"
        elif image_classification == 0:
            message += "fungal infection not Detected"

        skinImageInfo = SkinImage(skinImageId=skinImageId).getImageName()
        self.render('diagnosedImage.html', title='Image Diagnosis Results', message=message, imageId=skinImageId,
                    imageInfo=skinImageInfo, image_classification=image_classification)


class ViewUploadedImagesHandler(tornado.web.RequestHandler):
    def get(self, *args, **kwargs):
        imagesRetrieved = SkinImage().retrieveImages()
        self.render('viewImages.html', title='Uploaded Images', message='', images=imagesRetrieved)


class ViewSingleUploadedImageHandler(tornado.web.RequestHandler):
    def get(self, *args, **kwargs):
        imageRetrieved = SkinImage().retrieveImages()
        self.render('viewImage.html', title='Single Uploaded Image', message='', images=imageRetrieved)


class contact_form(tornado.web.RequestHandler):
    def get(self, *args, **kwargs):
        return self.render('contact.html', title="Contact - Final Year Project", message='')


if __name__ == "__main__":
    tornado.options.parse_command_line()
    app = tornado.web.Application(
        handlers=[
            (r"/", IndexHandler),
            (r"/images", ViewUploadedImagesHandler),
            (r"/single_image", ViewSingleUploadedImageHandler),
            (r"/diagnose/([0-9]+)", DiagnoseImageHandler),
            (r"/skinImages/(.*)", tornado.web.StaticFileHandler,
             {"path": os.path.join(os.path.dirname(__file__), "skinImages")}),
            (r"/contact", contact_form),
        ],
        cookie_secret="alkfa';438u98u54;aodfliahg0;.,;'oiau98kjin",
        template_path=os.path.join(os.path.dirname(__file__), "templates"),
        static_path=os.path.join(os.path.dirname(__file__), "static"),
        debug=True
    )
    tornado.httpserver.HTTPServer(app).listen(options.port)
    tornado.ioloop.IOLoop.instance().start()
