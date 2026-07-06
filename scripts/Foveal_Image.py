from PIL import Image
from FovealModel import FovealModel


img = Image.open("img/visual_scene.png")

# Apply the foveal model to the image with a fixation point at (128, 128)
#  should change this point based on the image
model = FovealModel(fovea_radius=40, peripheral_radius=90)
foveated_img = model.apply(img, fixation=(128, 128))
    

foveated_img.show()

# Saves the image to the folder "img"
foveated_img.save("img/foveated_scene.png")
print("Saved img/foveated_scene.png")
