import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data
df=pd.DataFrame({'x_values': range(1,6), 'y1_values': [0.058133807696741646,0.060340506808064055,0.06399633509199538,0.057888992337139546,0.07133520474341175], 'y2_values': [0.07159354036856107,0.08309064132977127,0.0898850169391,0.08259715223330791,0.07491614618735545], 'y3_values': [0.00894187779434, 0.00894187779434, 0.00372578241431, 0.0, 0.00745156482861]})

# multiple line plots
plt.plot( 'x_values', 'y1_values', data=df, marker='o', color='skyblue', linewidth=4)
plt.plot( 'x_values', 'y2_values', data=df, marker='+', color='olive', linewidth=2)
plt.plot( 'x_values', 'y3_values', data=df, marker='x', color='olive', linewidth=2, linestyle='dashed', label="toto")

# show legend
#plt.legend()

# show graph

plt.savefig('turn.png')
