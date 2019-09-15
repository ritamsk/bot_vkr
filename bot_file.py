import pyautogui
from state_def import  get_state

actionsList = ['jump', 'roll']
botControl = ['up', 'down']

def getKey(action):
    global botControl, actionsList
    action = actionsList.index(actionsList[action])
    curkey = botControl[action]
    #print(curkey)
    return curkey

def botAgent(action):
    k = getKey(action)
    pyautogui.press(k)

def check_crash(model, IMG_SIZE, image):
    alive = get_state(model, IMG_SIZE,image)
    if not alive:
        pyautogui.press('r')
    return alive




'''''
time.sleep(2)
while True:
    botAgent(actionsList[0])
'''''
