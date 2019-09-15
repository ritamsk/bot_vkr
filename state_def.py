import cv2
import numpy as np

def get_state(model, IMG_SIZE, image):

    img = cv2.resize(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), (IMG_SIZE[0], IMG_SIZE[1]))
    data = [img[:,:,:1]]
    # model_out = model.predict([data])[0]
    model_out = model.predict([data])
    print(model_out[0][0])
    return round(model_out[0][0])


#model, IMG_SIZE = state.build_neural_network()
#get_state(image = cv2.imread("images/gameplay/frame3.jpg"), model=model, IMG_SIZE=IMG_SIZE)