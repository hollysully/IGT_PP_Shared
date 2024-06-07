library(plyr)
library(patchwork)

# NOTE: this code is from:  https://www.r-bloggers.com/2014/10/my-commonly-done-ggplot2-graphs/ 

# read in data
data <- read.csv("/Users/tuo09169/Dropbox/1_Comp_Modelling/1_IGT_PlayPass/IGT_PP_Shared/1_TADS_long/Data/3_LongFormat/TADS_child_long_IGT.csv")


# make a basic plot for the whole group
tspag = ggplot(data, aes(x=age, y=Arew_qnormed)) + 
  geom_line() + guides(colour=FALSE) + xlab("Observation Time Point") +
  ylab("Arew_qnormed") + 
  theme_bw() # Apply a black-and-white theme
spag = tspag + aes(colour = factor(id)) 
spag

# add a black fit line
sspag = spag + geom_smooth(se=FALSE, colour="black", size=2)
sspag



##############################################################################
# now make separate plots for each group
##############################################################################


# Create two subsets of the data based on 'MatHxDepress'
MatHxDepress <- subset(data, MatHxDepress == 1)
NoMatHxDepress <- subset(data, MatHxDepress == 0)



# Plot for Group 1 (MatHxDepress = 1) in red
plot_MatHxDepress <- ggplot(MatHxDepress, aes(x = age, y = Arew_qnormed, group = id)) +
  geom_line(color = "pink", alpha = 0.5) +  # Individual lines in pink
  geom_smooth(aes(group = 1), method = "lm", se = FALSE, color = "red", size = 2) +  # Overall fit line in red
  guides(colour = FALSE) +
  xlab("Observation Time Point") +
  ylab("Arew_qnormed") +
  theme_bw()

# Plot for Group 2 (MatHxDepress = 0) in blue
plot_NoMatHxDepress <- ggplot(NoMatHxDepress, aes(x = age, y = Arew_qnormed, group = id)) +
  geom_line(color = "lightgreen", alpha = 0.5) +  # Individual lines in light blue
  geom_smooth(aes(group = 1), method = "lm", se = FALSE, color = "darkgreen", size = 2) +  # Overall fit line in dark blue
  guides(colour = FALSE) +
  xlab("Observation Time Point") +
  ylab("Arew_qnormed") +
  theme_bw()



combined_plot <- plot_NoMatHxDepress + plot_MatHxDepress

print(combined_plot)





