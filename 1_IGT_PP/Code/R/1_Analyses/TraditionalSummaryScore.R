library(dplyr)
library(tidyverse) 

Sess1 <- read.csv("/Users/tuo09169/Dropbox/1_Comp Modelling/1_PlayPass_IGT/Data/1_Preprocessed/Sess1_IGT.csv")
Sess2 <- read.csv("/Users/tuo09169/Dropbox/1_Comp Modelling/1_PlayPass_IGT/Data/1_Preprocessed/Sess2_IGT.csv")

names(Sess2)



# NOTE:  
# AB = playorpass - Version 1
# BD = playorpass - Version 2

# create empty dataframes for 
PlayDecks <- data.frame(Subject=integer(),
                      Version=integer(),
                      Play1=integer(), 
                      Play2=integer(),
                      Play3=integer(),
                      Play4=integer())

Sess2 <- mutate(Sess2, ExperimentName = recode(ExperimentName, 'playorpass - Version 1' = 1, 'playorpass - Version 2' = 2))

# fill PlayDecks with %Play for each deck and the version subs played
subj_ids <- unique(Sess2$Subject) 
n_subj <- length(subj_ids)


# Subject loop
for (i in 1:n_subj) {
  # subset & check the AB data
  tmp <- subset(Sess2, Subject==subj_ids[i])
  Ply1 <- (sum(tmp$stim==1 & tmp$ydata==1))/sum(tmp$stim==1)
  Ply2 <- (sum(tmp$stim==2 & tmp$ydata==1))/sum(tmp$stim==2)
  Ply3 <- (sum(tmp$stim==3 & tmp$ydata==1))/sum(tmp$stim==3)
  Ply4 <- (sum(tmp$stim==4 & tmp$ydata==1))/sum(tmp$stim==4)
  Version <- unique(tmp$ExperimentName)
  Subject <- tmp$Subject[i]
  tmp_row <- cbind(Subject, Version, Ply1, Ply2, Ply3, Ply4)
  PlayDecks <- rbind(PlayDecks, tmp_row)
}



# # Subject loop
# for (i in 1:n_subj) {
#   # subset & check the AB data
#   tmp <- subset(Sess2, Subject==subj_ids[i])
#   PlyA <- (sum(tmp$cardname=="A" & tmp$ydata==1))/sum(tmp$cardname=="A")
#   PlyB <- (sum(tmp$cardname=="B" & tmp$ydata==1))/sum(tmp$cardname=="B")
#   PlyC <- (sum(tmp$cardname=="C" & tmp$ydata==1))/sum(tmp$cardname=="C")
#   PlyD <- (sum(tmp$cardname=="D" & tmp$ydata==1))/sum(tmp$cardname=="D")
#   Version <- unique(tmp$ExperimentName)
#   Subject <- tmp$Subject[i]
#   tmp_row <- cbind(Subject, Version, PlyA, PlyB, PlyC, PlyD)
#   PlayDecks <- rbind(PlayDecks, tmp_row)
# }



path_out <- "/Users/tuo09169/Desktop/"
PlayDecks_dat = paste(path_out, 'PlayDecks.csv', sep = '')
write.csv(PlayDecks,PlayDecks_dat)








  
  
  
  
  Sess2_currAB <- subset(Sess2, ExperimentName== "playorpass - Version 1")
  Sess2_currBD <- subset(Sess2, ExperimentName== "playorpass - Version 2")  
  
