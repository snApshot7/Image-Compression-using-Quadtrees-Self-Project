import numpy as np 
import cv2
from PIL import Image, ImageDraw

MAX_DEPTH = 8
DETAIL_THRESHOLD = 13
SIZE_MULT = 1

def average_colour(image):
    image_arr = np.asarray(image)

    avg_color_per_row = np.average(image_arr, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0) 

    return (int(avg_color[0]), int(avg_color[1]), int(avg_color[2]))

def weighted_average(hist):
    total = sum(hist)
    error = value = 0

    if total > 0:
        value = sum(i * x for i, x in enumerate(hist)) / total
        error = sum(x * (value - i) ** 2 for i, x in enumerate(hist)) / total
        error = error ** 0.5

    return error

def get_detail(hist):
    red_detail = weighted_average(hist[:256])
    green_detail = weighted_average(hist[256:512])
    blue_detail = weighted_average(hist[512:768])

    detail_intensity = red_detail * 0.2989 + green_detail * 0.5870 + blue_detail * 0.1140

    return detail_intensity

class Quadrant():
    def __init__(self, image, bbox, depth):
        self.bbox = bbox
        self.depth = depth
        self.children = None
        self.leaf = False

        image = image.crop(bbox)
        hist = image.histogram()

        self.detail = get_detail(hist)
        self.colour = average_colour(image)

    def split_quadrant(self, image):
        left, top, width, height = self.bbox

        middle_x = left + (width - left) / 2
        middle_y = top + (height - top) / 2

        upper_left = Quadrant(image, (left, top, middle_x, middle_y), self.depth+1)
        upper_right = Quadrant(image, (middle_x, top, width, middle_y), self.depth+1)
        bottom_left = Quadrant(image, (left, middle_y, middle_x, height), self.depth+1)
        bottom_right = Quadrant(image, (middle_x, middle_y, width, height), self.depth+1)

        self.children = [upper_left, upper_right, bottom_left, bottom_right]

class QuadTree():
    def __init__(self, image):
        self.width, self.height = image.size 

        self.max_depth = 0

        self.start(image)
    
    def start(self, image):
        self.root = Quadrant(image, image.getbbox(), 0)
        
        self.build(self.root, image)

    def build(self, root, image):
        if root.depth >= MAX_DEPTH or root.detail <= DETAIL_THRESHOLD:
            if root.depth > self.max_depth:
                self.max_depth = root.depth

            root.leaf = True
            return 
        
        root.split_quadrant(image)

        for children in root.children:
            self.build(children, image)

    def create_image(self, custom_depth, show_lines=False):
        # creating blank image canvas
        image = Image.new('RGB', (self.width, self.height))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, self.width, self.height), (0, 0, 0))

        leaf_quadrants = self.get_leaf_quadrants(custom_depth)

        for quadrant in leaf_quadrants:
            if show_lines:
                draw.rectangle(quadrant.bbox, quadrant.colour, outline=(0, 0, 0))
            else:
                draw.rectangle(quadrant.bbox, quadrant.colour)

        return image

    def get_leaf_quadrants(self, depth):
        if depth > self.max_depth:
            raise ValueError('A depth larger than the trees depth was given')

        quandrants = []

        self.recursive_search(self, self.root, depth, quandrants.append)

        return quandrants

    def recursive_search(self, tree, quadrant, max_depth, append_leaf):
        if quadrant.leaf == True or quadrant.depth == max_depth:
            append_leaf(quadrant)

        elif quadrant.children != None:
            for child in quadrant.children:
                self.recursive_search(tree, child, max_depth, append_leaf)

    def create_gif(self, file_name, duration=1000, loop=0, show_lines=False):
        gif = []
        end_product_image = self.create_image(self.max_depth, show_lines=show_lines)

        for i in range(self.max_depth):
            image = self.create_image(i, show_lines=show_lines)
            gif.append(image)

        for _ in range(4):
            gif.append(end_product_image)

        gif[0].save(
            file_name,
            save_all=True,
            append_images=gif[1:],
            duration=duration, loop=loop)
        

if __name__ == '__main__':
    
    image_path = "/home/kartik/Documents/image_compression/img1.jpeg"

    image = Image.open(image_path)
    image = image.resize((image.size[0] * SIZE_MULT, image.size[1] * SIZE_MULT))

    quadtree = QuadTree(image)

    depth = 7
    image = quadtree.create_image(depth, show_lines=False)
    quadtree.create_gif("img1.gif", show_lines=True)
    
    image.save("img1_quadtree.jpg")
