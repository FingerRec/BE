import numpy as np
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
# sns.set()

# palette = np.array(sns.color_palette("hls", 10))
patch = mpatches.Patch(color='red', label='red')
plt.legend(handles=[patch])
plt.savefig("../../../experiments/visualization/color_paletters.png")