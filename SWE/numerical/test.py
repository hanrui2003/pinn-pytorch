from clawpack.pyclaw import examples
from clawpack.pyclaw import plot

claw = examples.dam_break.setup()
claw.run()
plot.html_plot()
