from random import sample
import matplotlib.pyplot as plt
from utilities import get_some_filter


def have_a_look_at(dataframe, activity_code, signal, axis, num_actors=1, actors=None):
    if actors == None:
        actors = sample(range(0, 25), num_actors)
    label = ["subject_" + str(actor) for actor in actors]
    data = get_some_filter(dataframe, actors, activity_code)

    for i in actors:
        axis.plot(data.loc[data["subject"] == i][signal][100:300])
        axis.legend(label)
        axis.set_title(activity_code[0])
