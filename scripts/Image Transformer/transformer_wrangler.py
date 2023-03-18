"""
A class to handle the data wrangling for the image transformer method of the BirdCLEF 2023 Kaggle Challenge.

This class takes sound clips, splits them to an appropriate length, and prepares them to be passed to a transformer neural network.

Authors:
Justin Butler & Garrett Toews

"""

class TransformerData:
    def __init__(self, sound_clips):
        '''
        Initializer class for the Transformer.

        Creates a class instance and passes the array of sound clips to the preparatory methods.

        @TODO: Handle pre-split clips AND long sound files
        '''
        self.data = self.convert(sound_clips)
    
    def sound_to_image(self, clip):
        '''
        Convert the image file into a power spectral density
        '''
        pass
    
    def resize_image(self, img):
        '''
        Resize image into array of 16x16 images
        '''
        pass

    def flatten_image(self, img):
        '''
        Flatten the 16x16 images into vector
        '''
        pass

    def encode(self, vector):
        '''
        Prepare the vector for the Transformer
        @TODO: Figure out what the hell this means
        '''
        pass

    def convert(self, sound_clips):
        '''
        Driving function that passes each clip through the preparatory methods.
        '''
        converted_data=[]
        for clip in sound_clips:
            image = self.sound_to_image(clip)
            tiny_image=self.resize_image(image)
            image_vector=self.flatten_image(tiny_image)
            converted_data.append(self.encode(image_vector))
        return converted_data

def main():
    '''
    Main to handle the operation of the code.
    '''
    pass

if __name__ == "main":
    main()