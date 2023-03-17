library(dplyr)
library(tidyverse) 
library(haven)
library(lubridate)
library(readxl)
library(here)

# test comment 


#import Play or Pass raw data
# NOTE: before I saved these as .xlsx files, I converted 'SessionDate' to format:  2015-10-23, to work with lubridate
# `here(...)` finds the root project directory, and then creates the file path given arguements
AB <- read_excel(here("Data", "0_Raw", "AB", "Merged_AB.xlsx"))
BD <- read_excel(here("Data", "0_Raw", "BD", "Merged_BD.xlsx"))

names(AB)
length(unique(AB$Subject)) #44
length(unique(BD$Subject)) #44

AB <- as.data.frame(AB)
# Remove every row (3 per sub) where AB$Procedure != "Knowledge"
# currently, 5412 rows. subtract 3 knowledge row x 44 subs = 5280 rows remaining
AB <- subset(AB, Procedure!="Knowledge")

BD <- as.data.frame(BD)
# Remove every row (3 per sub) where Procedure != "Knowledge"
# currently, 5412 rows. subtract 3 knowledge row x 44 subs = 5280 rows remaining
BD <- subset(BD, Procedure!="Knowledge")



#Note: PlayPass IGT task subject numbers are 2049—2099

# import questionnaire data
Sess1_Q <- read_sav(here("Data", "0_Raw", "MergedQuest_3.21.16-Session1.sav"))
Sess2_Q <- read_sav(here("Data", "0_Raw", "MergedQuest_3.21.16-Session2.sav"))
# subset out just subjects from this study
Sess1_Q <- subset(Sess1_Q, ID >= 2049)
Sess2_Q <- subset(Sess2_Q, ID >= 2049)

length(unique(Sess1_Q$ID)) #51
length(unique(Sess2_Q$ID)) #41


Version <- read_sav(here("Data", "0_Raw", "IGT_VersionTrackingSheet.sav"))
names(Version)
Version <- subset(Version, SubjectID >= 2049)
Version <- subset(Version, SubjectID != 2059) # took out this sub b/c they played old IGT
Version <- subset(Version, SubjectID != 2083) # took out this sub b/c they had no task data

# create a subject list 
IGT_PP_Subs <- unique(Version$SubjectID) 
# should be 49 subs long


# look for differences in the Sess 1 sub numbers in the task vs. questionnaire data
setdiff((unique(AB$Subject)), (unique(Sess1_Q$ID)))   # diff is 4050
# set x to index all rows where Subject == 4050
x <- which(AB$Subject==4050) 
# change subject ID 4050 to 2050
AB$Subject[x] <- 2050



# look for differences in the Sess 2 sub numbers in the task vs. questionnaire data
setdiff((unique(BD$Subject)), (unique(Sess2_Q$ID)))  # diffs are 20783  2090  2093  2096  2098
# change sub 20783 to 2078 b/c their task participation date matches sub 2078 questionnaire participation date for Sess2
x <- which(BD$Subject==20783) 
BD$Subject[x] <- 2078
  
  
  

setdiff((unique(Sess1_Q$ID)), (unique(AB$Subject))) # 2059 2083 2090 2093 2094 2096 2098
# sub 2059 completed the old IGT task and so is not part of this study
# I believe subs 2083 2090 2093 2094 2096 2098 completed BD at Sess 1 and did not complete IGT at Sess 2



################################################################################
################################################################################
# recode values in IGT data
################################################################################
################################################################################


# Recode the cardname from letters to numbers and for the BD Version 2 
# to match the corresponding cards from AB Version 1 in a new variable called stim

# AB = playorpass - Version 1
# BD = playorpass - Version 2
# 
# Version AB >> Version BD
# 
# A >> B:  1
# B >> D:  2
# C >> A:  3
# D >> C:  4


AB$cardname
AB <- mutate(AB, stim = recode(cardname, 'A'= 1, 'B' = 2, 'C' = 3, 'D' = 4))


BD$cardname
BD <- mutate(BD, stim = recode(cardname, 'A'= 3, 'B' = 1, 'C' = 4, 'D' = 2))





# create a new variable called Srewlos that recodes for the sign of the  reward/loss in absmoney1
# webpage on conditional recoding:  https://rpubs.com/prlicari13/541675

AB$Srewlos[AB$absmoney1 < 0 ] <- -1
AB$Srewlos[AB$absmoney1 > 0 ] <- 1
AB$Srewlos[AB$absmoney1 == 0 ] <- 0
# Change all the na's to 0
AB$Srewlos[is.na(AB$Srewlos)] <- 0


BD$Srewlos[BD$absmoney1 < 0 ] <- -1
BD$Srewlos[BD$absmoney1 > 0 ] <- 1
BD$Srewlos[BD$absmoney1 == 0 ] <- 0
# Change all the na's to 0
BD$Srewlos[is.na(BD$Srewlos)] <- 0



# now recode absmoney1 into a new variable called rewlos that is the value of absmoney1
# Scale by 100 like in the original model (since outcomes go up to the 1000's)
AB$rewlos <- AB$absmoney1 / 100
AB$rewlos[is.na(AB$rewlos)] <- 0

BD$rewlos <- BD$absmoney1 / 100
BD$rewlos[is.na(BD$rewlos)] <- 0



# recode card.RESP (where Play (1) or Pass(3))to a new variable called ydata, 
# where 1 for play or 2 for pass, & NAs are recoded to 0

AB$ydata[AB$card.RESP == 3 ] <- 2
AB$ydata[AB$card.RESP == 1 ] <- 1
# Change all the na's to 2
AB$ydata[is.na(AB$ydata)] <- 2


BD$ydata[BD$card.RESP == 3 ] <- 2
BD$ydata[BD$card.RESP == 1 ] <- 1
# Change all the na's to 2
BD$ydata[is.na(BD$ydata)] <- 2




# ################################################################################
# ################################################################################
# # check task dates
# ################################################################################
# ################################################################################
# 
# check_sess <- as.data.frame(array(NA, c((length(IGT_PP_Subs)), 3)))
# 
# # Now get the dates of task completion to confirm info in Versions is correct
# # Getting subject IDs and number of subjects from data
# subj_ids <- IGT_PP_Subs
# n_subj <- length(subj_ids)
# # Subject loop
# for (i in 1:n_subj) {
#   check_sess[i,1] <- subj_ids[i]
#   # Subset out current subject
#   tmpAB <- subset(AB, Subject==subj_ids[i])
#   # Get the SessionDate
#   check_sess[i,2] <- ymd(tmpAB$SessionDate[1])
#   tmpBD <- subset(BD, Subject==subj_ids[i])
#   # Get the SessionDate
#   check_sess[i,3] <- ymd(tmpBD$SessionDate[1])
# }


# websites re: processing date info
# https://lubridate.tidyverse.org/
# https://data.library.virginia.edu/working-with-dates-and-time-in-r-using-the-lubridate-package/
# https://rawgit.com/rstudio/cheatsheets/main/lubridate.pdf

# NOTE: lubridate stores dates as numbers like 16731, 
# which I think represents the date as # of days since 1970-01-01
# use as_date to convert back to date format
# as_date(check_sess[i,3]) 

# # create a variable for the ellapsed time between sessions
# names(check_sess) <- c("ID", "AB_date", "BD_date")
# check_sess[,4] <- check_sess$AB_date - check_sess$BD_date
# names(check_sess)[4] <- "AB_BD_diff"

# create empty dataframes for the Sess 1 & Sess 2 task data
Sess1_IGT <- Sess2_IGT <- data.frame(Subject=integer(),
                        ExperimentName=character(),
                        absmoney1=integer(),
                        rewlos=integer(),
                        Srewlos=integer(),
                        card.RESP=integer(),
                        ydata=integer(),
                        cardname=integer(),
                        stim=integer()
                        )
                    




# fill Sess1_IGT & Sess2_IGT with each subject's appropriate task data (based on AB_BD_diff)

subj_ids <- IGT_PP_Subs
n_subj <- length(subj_ids)

# Subject loop
for (i in 1:n_subj) {
  # subset & check the AB data
  tmpAB <- subset(AB, Subject==subj_ids[i])
  AB_date <- ymd(tmpAB$SessionDate[1])
  if (nrow(tmpAB) > 0) {
    tmpAB <- tmpAB[,c("Subject", "ExperimentName", "absmoney1", "rewlos", "Srewlos", "card.RESP", "ydata", "cardname", "stim")]
    AB_dat <- TRUE } 
  else if (nrow(tmpAB) == 0) {
      AB_dat <- FALSE }
  # subset & check the BD data
  tmpBD <- subset(BD, Subject==subj_ids[i])
  BD_date <- ymd(tmpBD$SessionDate[1])
  if (nrow(tmpBD) > 0) {
    tmpBD <- tmpBD[,c("Subject", "ExperimentName", "absmoney1", "rewlos", "Srewlos", "card.RESP", "ydata", "cardname", "stim")]
    BD_dat <- TRUE }
  else if (nrow(tmpBD) == 0) {
      BD_dat <- FALSE }
  # so now I have the sub's AB & BD data, a date for each session, & a TRUE/FALSE for whether they have data
  # pick which data belongs to Sess1 & Sess2 dataframes based on AB_date & BD_date & TRUE/FALSE for whether they have data
  if (AB_dat == TRUE && BD_dat == TRUE) {
    if (AB_date < BD_date) {
      Sess1_IGT <- rbind(Sess1_IGT, tmpAB)
      Sess2_IGT <- rbind(Sess2_IGT, tmpBD) }
    else if (BD_date < AB_date) {
      Sess1_IGT <- rbind(Sess1_IGT, tmpBD)
      Sess2_IGT <- rbind(Sess2_IGT, tmpAB) }}
  else if (AB_dat == TRUE && BD_dat == FALSE) {
    Sess1_IGT <- rbind(Sess1_IGT, tmpAB) }
  else if (AB_dat == FALSE && BD_dat == TRUE) {
    Sess1_IGT <- rbind(Sess1_IGT, tmpBD) }
}


length(unique(Sess1_IGT$Subject)) # 49 subs
length(unique(Sess2_IGT$Subject)) # 39 subs

setdiff((unique(IGT_PP_Subs)), (unique(Sess1_IGT$Subject))) # no differences

unique(Sess2_IGT$ExperimentName)





############################################################################################################
# Save IGT dataframes as .csv files
############################################################################################################

Sess1_IGT_dat = here("Data", "1_Preprocessed", "Sess1_IGT.csv")
write.csv(Sess1_IGT,Sess1_IGT_dat)

Sess2_IGT_dat = here("Data", "1_Preprocessed", "Sess2_IGT.csv")
write.csv(Sess2_IGT,Sess2_IGT_dat)






############################################################################################################
# code for making stan-ready lists from just the sess 1 data or just the sess 2 data
############################################################################################################

for (session in 1:2) {
  if (session == 1) {
    Sess_data <- Sess1_IGT
  } else if (session == 2) {
    Sess_data <- Sess2_IGT    
  }
  
  # Individual Subjects
  subjList <- unique(Sess_data$Subject) # list of subject IDs from the session data specified by Sess_data
  numSubjs <- length(subjList)        # number of subjects
  
  # Trials per subject in behavioral data (0 when missing)
  Tsubj <- as.vector(rep(0, numSubjs)) 
  
  # now use these for loops to fill the array 'Tsubj' with the number of trials, when there are trials, otherwise the value stays 0
  
  
  # Trials per subject in behavioral data
  # number of trials for each subject
  for ( i in 1:numSubjs )  {
    curSubj  <- subjList[i]
    Tsubj[i] <- sum(Sess_data$Subject==curSubj)
  }
  maxTrials <- max(Tsubj)
  
  
  
  # Behavioral data arrays
  RLmatrix <- SRLmatrix <- stim <- array(-1, c(numSubjs, maxTrials))
  Ydata <- array(-1, c(numSubjs, maxTrials))
  
  
  # Filling arrays with raw data
  for (i in 1:numSubjs) {
    currID   <- subjList[i]
    tmp_dat  <- subset(Sess_data, Subject==currID)
    stim[i,]  <- tmp_dat$stim
    Ydata[i,] <- tmp_dat$ydata
    RLmatrix[i,] <- tmp_dat$rewlos
    SRLmatrix[i,] <- tmp_dat$Srewlos
  }  
  
  
  dataList <- list(
    N       = numSubjs,
    T       = maxTrials,
    Tsubj   = Tsubj,
    stim    = stim,
    Srewlos = SRLmatrix,     
    rewlos  = RLmatrix,
    ydata   = Ydata,
    subjID  = subjList
  )
  
  # save Sess 1 or 2 lists as .rds
  saveRDS(dataList, file = here("Data", "1_Preprocessed", paste0("Sess", session, ".rds")))
}











# I'm here, this is old code for preprocessing the raw data for running a joint ORL model

############################################################################################################
# code for making test-retest stan-ready lists
# NONE OF THIS CODE HAS BEEN UPDATED. THIS IS CODE FROM SCRIPT;
# /Users/tuo09169/Dropbox/1_Comp Modelling/1_IGT_CompModel/1_Decision_Test_ReTest/TestRetest_IGT/Code/R/0_Preprocessing/1_preprocess_joint_retest.R
############################################################################################################

# Individual Subjects
subjList <- unique(comb_dat$subjID) # list of subject IDs
numSubjs <- length(subjList)        # number of subjects

# Trials per subject in behavioral data (0 when missing)
Tsubj <- array(0, c(numSubjs, 2)) # a 2-column array, filled with all 0's
# now use these for loops to fill the array 'Tsubj' with the number of trials, when there are trials, otherwise the value stays 0
for (i in 1:numSubjs) {
  for (s in 1:2) {
    curSubj <- subjList[i]
    # this next line fills Tsubj[1,s] with the sum of all the rows where subjID==curSubj & session==s; 
    # otherwise leaves 0 in that spot in the Tsubj array
    Tsubj[i,s] <- sum(comb_dat$subjID==curSubj & comb_dat$session==s) #Sum of Vector Elements--this is how you get the trial count
  }
}
maxTrials <- max(Tsubj)

# Behavioral data arrays
# this creates 3 identical arrays of [1:50, 1:100, 1:2]; the array is filled with -1 values
# array names are RLmatrix, SRLmatrix, choice
RLmatrix <- SRLmatrix <- choice <- array(-1, c(numSubjs, maxTrials,2))

# Loop through and format into 3 arrays: choice, RLmatrix, SRLmatrix
for (i in 1:numSubjs) {
  for (s in 1:2) {
    if (Tsubj[i,s]>0) {
      tmp_dat <- comb_dat %>%
        filter(subjID==subjList[i] & session==s) %>% # By filtering out each subjID like this, the order 
        arrange(trial)                               # of subjList is preserved in the arrays
      choice[i,,s]    <- tmp_dat$choice
      RLmatrix[i,,s]  <- tmp_dat$gain - tmp_dat$loss
      SRLmatrix[i,,s] <- sign(RLmatrix[i,,s])
    }
  }
}



# Put in stan-ready list
stan_dat <- list(N        = numSubjs,     # single value of 50
                 T        = maxTrials,    # single value of 100
                 Tsubj    = Tsubj,        # 2-columns: 50 rows, 1 column for each session w/ trial nums
                 choice   = choice,       # the choice array made above
                 outcome  = RLmatrix/100, # divide outcomes by 100    # the RLmatrix made above
                 sign_out = SRLmatrix,    # the SLRmatrix made above
                 subjIDs  = subjList)     # subjList contains the subject IDs, and the order is retained in the loop above (see comment) 

saveRDS(stan_dat, file = "1_Preprocessed/stan_ready_ORL_joint_retest.rds")

