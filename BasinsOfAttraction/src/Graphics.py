#@PydevCodeAnalysisIgnore
import Image

class ImageGenerator(object):
    
    @staticmethod
    def generate_image(data_list, file_name, type):
        width = len(data_list)
        height = len(data_list[0])

        image = Image.new("RGB", (width, height))
        pixels = image.load()

        for i, row in enumerate(reversed(data_list)): # reversed in order to have list from up to down so generated image won't be flipped
            for j, item in enumerate(row):
                if item == 1:
                    pixels[i, j] = (255, 0, 0)
                else:
                    pixels[i, j] = (0, 0, 255)

        image.save(file_name + ".png", type)

