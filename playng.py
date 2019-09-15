import pyscreenshot as pyscreen

import bot_file as bot
from score_def import get_score


def get_reward(alive, prev_reward):
    if alive:
        return prev_reward + 5
    else:
        return prev_reward - 10




def Play(SVMclf, state_model, IMG_SIZE, prev_reward):

    image = pyscreen.grab()

    alive = bot.check_crash(state_model, IMG_SIZE, image)
    score = get_score(SVMclf, image)
    reward = get_reward(alive,prev_reward)

    return image, score, reward, alive






