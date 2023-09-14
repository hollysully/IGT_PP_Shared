library(dplyr)
library(tidyverse) 
library(haven)
library(lubridate)
library(readxl)


# the here package relates somehow to using "projects", which I set up when putting this on github
# open a new session of R, & in the top right hand corner, open the project file:
# /Users/tuo09169/Dropbox/1_Comp_Modelling/1_IGT_PlayPass/IGT_PP_Shared/IGT_PP_Shared.Rproj
# then, declare where the script is relative to the project root directory
here::i_am("1_SIGT_PP/Code/R/0_Preprocess/Preprocess_SIGT.R")

# here is a website that explains how here works:
# https://here.r-lib.org/articles/here.html#declare-the-location-of-the-current-script-1
# another helpful website:  http://jenrichmond.rbind.io/post/how-to-use-the-here-package/

library(here)
# the here() command tells you where the project root directory is
here()



#import IGT_PP & SIGT_PP merged .xlsx files:   (Note: info here about how I made the merged .xlsx files: https://osf.io/a9b5r/wiki/home/ )
# NOTE: before I saved these as .xlsx files, I converted 'SessionDate' to format:  2015-10-23, to work with lubridate
# `here(...)` finds the root project directory, and then creates the file path given arguements
IGT_AB <- as.data.frame(read_excel(here("1_SIGT_PP", "Data", "0_Raw", "IGT_AB_Merged.xlsx")))
IGT_BD <- as.data.frame(read_excel(here("1_SIGT_PP", "Data", "0_Raw", "IGT_BD_Merged.xlsx")))
SIGT_AB <- as.data.frame(read_excel(here("1_SIGT_PP", "Data", "0_Raw", "SIGT_AB_Merge_all.xlsx")))
SIGT_BD <- as.data.frame(read_excel(here("1_SIGT_PP", "Data", "0_Raw", "SIGT_BD_Merged_all.xlsx")))



# Remove every row (3 per sub) where IGT_AB$Procedure != "Knowledge"
IGT_AB <- subset(IGT_AB, Procedure!="Knowledge") 
IGT_BD <- subset(IGT_BD, Procedure!="Knowledge") 
SIGT_AB <- subset(SIGT_AB, Procedure!="Knowledge") 
SIGT_BD <- subset(SIGT_BD, Procedure!="Knowledge") 


################################################################################
# Remove the second set of task data for 2 subjects that completed the IGT twice
################################################################################

# sub 7156 completed IGT version 2 & SIGT on the same day, but then has data for IGT version 1 a week later
# so I am deleting the task data for sub 7156 from the IGT_AB dataframe (IGT version 1) 
IGT_AB <- subset(IGT_AB, Subject!=7156) 
  
# sub 7034 played IGT version 2 at 7:39pm & IGT version 1 at 8:20pm
# so, I am removing their IGT version 1 data
IGT_AB <- subset(IGT_AB, Subject!=7034) 



################################################################################
# import questionnaire data
################################################################################
Q_item <- read_sav(here("1_SIGT_PP", "Data", "0_Raw", "FINAL Questionnaire Data- Item Level 9.11.18.sav"))
Q_scales <- read_sav(here("1_SIGT_PP", "Data", "0_Raw", "FINAL Questionnaire Data- Summary Scales Only - 9.11.18.sav"))

length(unique(Q_scales$Subject)) # 199

# create a subject list 
SIGT_PP_Subs <- unique(Q_scales$Subject) 
# should be 199 subs long



################################################################################
################################################################################
# recode values in IGT & SIGT data
################################################################################
################################################################################
# FYI, the money lists are saved here:  /Users/tuo09169/Dropbox/1_Comp_Modelling/1_IGT_PlayPass/IGT_PP_Shared/1_SIGT_PP/z.Notes/MoneyLists_NewIGT.xlsx
#      the social lists are saved here:  /Users/tuo09169/Dropbox/1_Comp_Modelling/1_IGT_PlayPass/IGT_PP_Shared/1_SIGT_PP/z.Notes/SIGT List Orders.xlsx



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


IGT_AB$cardname
IGT_AB <- mutate(IGT_AB, stim = recode(cardname, 'A'= 1, 'B' = 2, 'C' = 3, 'D' = 4))

IGT_BD$cardname
IGT_BD <- mutate(IGT_BD, stim = recode(cardname, 'A'= 3, 'B' = 1, 'C' = 4, 'D' = 2))



SIGT_AB$cardname
SIGT_AB <- mutate(SIGT_AB, stim = recode(cardname, 'A'= 1, 'B' = 2, 'C' = 3, 'D' = 4))

SIGT_BD$cardname
SIGT_BD <- mutate(SIGT_BD, stim = recode(cardname, 'A'= 3, 'B' = 1, 'C' = 4, 'D' = 2))





################################################################################
################################################################################
# combine task versions into single IGT data & single SIGT data
################################################################################
################################################################################

# combine the two versions for IGT
IGT <- rbind(IGT_AB,IGT_BD)
length(unique(IGT$Subject)) #198

# combine the two versions for SIGT
SIGT <- rbind(SIGT_AB,SIGT_BD)
length(unique(SIGT$Subject)) #205 




################################################################################
################################################################################
# explore & define which sub nums are included in all data
################################################################################
################################################################################


# look for differences in sub numbers between the two tasks 
setdiff((unique(IGT$Subject)), (unique(SIGT$Subject))) # 7137 7145 4074 7081
setdiff((unique(SIGT$Subject)), (unique(IGT$Subject))) # 1066 1069 7047    1 1065 1076 7108 1073 7165 7451 1111


# get the IDs common to both tasks:  194 subjects
commonIDs_task <- Reduce(intersect, list((unique(IGT$Subject)), (unique(SIGT$Subject))))

# get the IDs common to both tasks and the questionnaires:  193 subjects
commonIDs_all <- Reduce(intersect, list((unique(IGT$Subject)), (unique(SIGT$Subject)), (unique(Q_scales$Subject))))




################################################################################
################################################################################
# check for complete task data
################################################################################
################################################################################


# check for complete IGT task data in all subjects:  sub 7081 does not have complete data
IGT_subnums <- sort(unique(IGT$Subject))
for (i in 1:(length(unique(IGT$Subject)))) {
  # subset the subject's data
  tmp <- subset(IGT, Subject==IGT_subnums[i])
  # print(tmp$Subject[1])
  # print(max(tmp$Trialorder))
  if (max(tmp$Trialorder) != 122) { # note: max Trialorder should be 122 
    #because there are 120 trials, but 3 "knowledege" trials bumped the Trialorder count 
    #even though they have been excluded from the current dataframe, 
    #but the last knowledge trial was truncated from dataframe
    print(tmp$Subject[1])
  }
}


# check for complete SIGT task data in all subjects:  
SIGT_subnums <- sort(unique(SIGT$Subject))
for (i in 1:(length(unique(SIGT$Subject)))) {
  # subset the subject's data
  tmp <- subset(SIGT, Subject==SIGT_subnums[i])
  #print(tmp$Subject[1])
  #print(max(tmp$Trialorder))
  if (max(tmp$Trialorder) != 122) { # note: max Trialorder should be 122 
    #because there are 120 trials, but 3 "knowledege" trials bumped the Trialorder count 
    #even though they have been excluded from the current dataframe, 
    #but the last knowledge trial was truncated from dataframe
    print(tmp$Subject[1]) 
    }
}



xx <- which(commonIDs_all==7081) # sub 7081 did not have complete IGT task data, but
# x returns an empty index, b/c sub num 7081 is not included in commonIDs_all 
# & so doesn't need to be excluded from the final list of subject numbers




################################################################################
################################################################################
# check dates tasks were completed
################################################################################
################################################################################

####  based on code below, sub 7018 will be removed from final subs b/c they completed the two tasks months apart


### NOTE:  if I want to also check the time that tasks were completed, I would need to use code like: 
#  parse_date_time(IGT$SessionStartDateTimeUtc, orders = c("%Y-%m-%d %H:%M:%S"))

subnums <- sort(commonIDs_all)

check_sess <- as.data.frame(array(NA, c((length(subnums)), 5)))

# Now get the dates of task completion to confirm info in Versions is correct
# Getting subject IDs and number of subjects from data

n_subj <- length(subnums)
# Subject loop
for (i in 1:n_subj) {
  check_sess[i,1] <- subnums[i]
  # Subset out current subject
  tmp_IGT <- subset(IGT, Subject==subnums[i])
  # Get the SessionDate
  check_sess[i,2] <- ymd(tmp_IGT$SessionDate[1])
  check_sess[i,3] <- as.character(tmp_IGT$SessionDate[1])
  tmp_SIGT <- subset(SIGT, Subject==subnums[i])
  # Get the SessionDate
  check_sess[i,4] <- ymd(tmp_SIGT$SessionDate[1])
  check_sess[i,5] <- as.character(tmp_SIGT$SessionDate[1])
}


# websites re: processing date info
# https://lubridate.tidyverse.org/
# https://data.library.virginia.edu/working-with-dates-and-time-in-r-using-the-lubridate-package/
# https://rawgit.com/rstudio/cheatsheets/main/lubridate.pdf

# NOTE: lubridate stores dates as numbers like 16731,
# which I think represents the date as # of days since 1970-01-01
# use as_date to convert back to date format
# as_date(check_sess[i,3])

# create a variable for the ellapsed time between sessions
names(check_sess) <- c("ID", "IGT_date_ymd", "IGT_date","SIGT_date_ymd", "SIGT_date")
check_sess[,6] <- check_sess$SIGT_date_ymd - check_sess$IGT_date_ymd
names(check_sess)[6] <- "SIGT_IGT_diff"




####  since sub 7018 completed the two tasks months apart, add this sub num to x to be removed from final sub list
x <- which(commonIDs_all== 7018)


################################################################################
################################################################################
# create the final list of subject numbers (n = 192)
################################################################################
################################################################################
subnums <- sort(commonIDs_all)
subnums <- subnums[-x] # this excludes sub 7018 b/c they completed the SIGT & IGT months apart
# for a final total of 192 subs





################################################################################
################################################################################
# subset IGT & SIGT by the final subject list
################################################################################
################################################################################

IGT <- IGT[IGT$Subject %in% subnums,] # from 23692 rows to 23040 rows (23040/ 192 = 120)
SIGT <- SIGT[SIGT$Subject %in% subnums,] # from 24600 rows to 23040 rows (23040/ 192 = 120)


sort(unique(IGT$Subject))
sort(unique(SIGT$Subject))


# ### quick check to validate the recoding by checking values in .csv files
# outdir = here("1_SIGT_PP", "Data", "0_Raw")
# 
# # save the .csv to check the recode makes sense
# filename1 = paste(outdir,'IGT_check.csv',sep = '')
# filename2 = paste(outdir,'SIGT_check.csv',sep = '')
# write.csv(IGT,filename1)
# write.csv(SIGT,filename2)








################################################################################
################################################################################
# create new variables that will be used for the ORL model
################################################################################
################################################################################

# create a new variable called Srewlos that recodes for the sign of the  reward/loss in absmoney1
# webpage on conditional recoding:  https://rpubs.com/prlicari13/541675

IGT$Srewlos[IGT$absmoney1 < 0 ] <- -1
IGT$Srewlos[IGT$absmoney1 > 0 ] <- 1
IGT$Srewlos[IGT$absmoney1 == 0 ] <- 0
# Change all the na's to 0
IGT$Srewlos[is.na(IGT$Srewlos)] <- 0


# now recode absmoney1 into a new variable called rewlos that is the value of absmoney1
# Scale by 100 like in the original model (since outcomes go up to the 1000's)
IGT$rewlos <- IGT$absmoney1 / 100
IGT$rewlos[is.na(IGT$rewlos)] <- 0





# Because the "neutral" face outcome in SIGT used a very low % morph of a happy face rather than a truly neutral face
# However, the neutral face is only presented when subs "play" & get a neutral outcome,
# when subs "pass" they recieve no facial feedback at all
# so this presents a challenge in terms of how to code the rewlos and the Srewlos variables for these two conditions:
# code as "neutral face" neutral? code "neutral face" as positive? 
# (see email with subject line:  SIGT: values for neutral stim)

# So for SIGT, we decided to always code the pass outcome as truly neutral, 
# but to code the "neutral face" outcomes in 3 different ways (3 diff analyses):

# 1. Code rewlos as the actual % facial morph used, but code Srewlos for neutral outcome as 0
# 2. Set the rewlos value for the neutral outcome as 0 & shift all the other rewlos values accordingly (i.e. subtracting 25 as Tom suggested)
# 3. Code rewlos as the actual % facial morph used, and allow Srewlos to take on the actual positive sign according to the true value for rewlos 
#    (which would then make the only true 0 rewlos/sign outcome the pass outcome)  


# NOTE: 
# pass >> SIGT$winlose1 = NA 
# play, but neutral outcome >> SIGT$winlose1 == "No change"


################################################################################
# SIGT code scheme 1:  rewlos1 = % facial morph/ 10; Srewlos1 = 0
################################################################################

# recode SocReward1 into a new variable called rewlos1 that is the % morph of the emotional faces
# dropping the leading zeros & scaling by 10
SIGT$rewlos1 <- as.numeric(substr((SIGT$SocReward1),10,12)) /10
# the "no outcome" following a pass is the only true 0 outcome
SIGT$rewlos1[is.na(SIGT$SocReward1)] <- 0

# create a new variable called Srewlos1 that recodes for the sign of the  reward/loss in absmoney1
# webpage on conditional recoding:  https://rpubs.com/prlicari13/541675
SIGT$Srewlos1[SIGT$winlose1 == "You lose" ] <- -1
SIGT$Srewlos1[SIGT$winlose1 == "You win" ] <- 1
SIGT$Srewlos1[SIGT$winlose1 == "No change" ] <- 0
SIGT$Srewlos1[is.na(SIGT$winlose1)] <- 0



################################################################################
# SIGT code scheme 2:  rewlos2 = (% facial morph -25)/ 10; Srewlos2 = 0
################################################################################

# recode SocReward1 into a new variable called rewlos2 that is the % morph of the emotional faces, 
# also subtracting 25 from the value, which sets the neutral face to 0 & shifting the other facial morph values 
# finally, scaling by 10
SIGT$rewlos2 <- ((as.numeric(substr((SIGT$SocReward1),10,12))) -25) / 10
# the "no outcome" following a pass is set to 0, so same value as the nuetral face outcome
SIGT$rewlos2[is.na(SIGT$SocReward1)] <- 0

# create a new variable called Srewlos2 that recodes for the sign of the  reward/loss in absmoney1
# webpage on conditional recoding:  https://rpubs.com/prlicari13/541675
SIGT$Srewlos2[SIGT$winlose1 == "You lose" ] <- -1
SIGT$Srewlos2[SIGT$winlose1 == "You win" ] <- 1
SIGT$Srewlos2[SIGT$winlose1 == "No change" ] <- 0
SIGT$Srewlos2[is.na(SIGT$winlose1)] <- 0


################################################################################
# SIGT code scheme 3:  rewlos3 = % facial morph / 10; Srewlos3 = 1
################################################################################

# recode SocReward1 into a new variable called rewlos3 that is the % morph of the emotional faces
# scaling by 10
SIGT$rewlos3 <- as.numeric(substr((SIGT$SocReward1),10,12)) /10
# the "no outcome" following a pass is the only true 0 outcome
SIGT$rewlos3[is.na(SIGT$SocReward1)] <- 0

# create a new variable called Srewlos2 that recodes for the sign of the  reward/loss in absmoney1
# webpage on conditional recoding:  https://rpubs.com/prlicari13/541675
SIGT$Srewlos3[SIGT$winlose1 == "You lose" ] <- -1
SIGT$Srewlos3[SIGT$winlose1 == "You win" ] <- 1
SIGT$Srewlos3[SIGT$winlose1 == "No change" ] <- 1
SIGT$Srewlos3[is.na(SIGT$winlose1)] <- 0






# recode card.RESP (where Play (1) or Pass(3))to a new variable called ydata, 
# where 1 for play or 2 for pass, & NAs are recoded to 0

IGT$ydata[IGT$card.RESP == 3 ] <- 2
IGT$ydata[IGT$card.RESP == 1 ] <- 1
# Change all the na's to 2
IGT$ydata[is.na(IGT$ydata)] <- 2



SIGT$ydata[SIGT$card.RESP == 3 ] <- 2
SIGT$ydata[SIGT$card.RESP == 1 ] <- 1
# Change all the na's to 2
SIGT$ydata[is.na(SIGT$ydata)] <- 2





# create empty dataframes for the IGT_pre, SIGT_v1_pre, SIGT_v2_pre, SIGT_v3_pre data
IGT_pre <- data.frame(Subject=integer(),
                      ExperimentName=character(),
                      absmoney1=integer(),
                      rewlos=integer(),
                      Srewlos=integer(),
                      card.RESP=integer(),
                      ydata=integer(),
                      cardname=integer(),
                      stim=integer()
)



SIGT_v1_pre <- SIGT_v2_pre <- SIGT_v3_pre <- data.frame(Subject=integer(),
                                                        ExperimentName=character(),
                                                        SocReward1=integer(),
                                                        rewlos=integer(),
                                                        Srewlos=integer(),
                                                        card.RESP=integer(),
                                                        ydata=integer(),
                                                        cardname=integer(),
                                                        stim=integer()
)




# fill IGT_pre, SIGT_v1_pre, SIGT_v2_pre, SIGT_v3_pre dataframes with each subject's appropriate version of data 

subj_ids <- subnums
n_subj <- length(subj_ids)

# Subject loop
for (i in 1:n_subj) {
  # subset the subject's IGT data
  tmpIGT <- subset(IGT, Subject==subj_ids[i])
  tmpIGT1 <- tmpIGT[,c("Subject", "ExperimentName", "absmoney1", "rewlos", "Srewlos", "card.RESP", "ydata", "cardname", "stim")]
  # add the subject's IGT data to dataframe IGT_pre
  IGT_pre <- rbind(IGT_pre, tmpIGT1)
  
  
  # subset the subject's SIGT data
  tmpSIGT <- subset(SIGT, Subject==subj_ids[i])
  tmpSIGT1 <- tmpSIGT[,c("Subject", "ExperimentName", "SocReward1", "rewlos1", "Srewlos1", "card.RESP", "ydata", "cardname", "stim")]
  tmpSIGT2 <- tmpSIGT[,c("Subject", "ExperimentName", "SocReward1", "rewlos2", "Srewlos2", "card.RESP", "ydata", "cardname", "stim")]
  tmpSIGT3 <- tmpSIGT[,c("Subject", "ExperimentName", "SocReward1", "rewlos3", "Srewlos3", "card.RESP", "ydata", "cardname", "stim")]
  # add the subject's SIGT data to dataframes SIGT_v1_pre, SIGT_v2_pre, SIGT_v3_pre 
  SIGT_v1_pre <- rbind(SIGT_v1_pre, tmpSIGT1)
  SIGT_v2_pre <- rbind(SIGT_v2_pre, tmpSIGT2)
  SIGT_v3_pre <- rbind(SIGT_v3_pre, tmpSIGT3)
}




############################################################################################################
# Save the preprocessed data as .csv files
############################################################################################################

### quick check to validate the recoding by checking values in .csv files

outdir = here("1_SIGT_PP", "Data", "1_Preprocessed")
  
# save the .csv files of pre-processed data to check the recodes, etc makes sense
IGT_preproc = paste(outdir,'/', 'IGT_preproc.csv',sep = '')
write.csv(IGT_pre,IGT_preproc) 

SIGT_v1_preproc = paste(outdir,'/', 'SIGT_v1_preproc.csv',sep = '')
write.csv(SIGT_v1_pre,SIGT_v1_preproc) 

SIGT_v2_preproc = paste(outdir,'/', 'SIGT_v2_preproc.csv',sep = '')
write.csv(SIGT_v2_pre,SIGT_v2_preproc) 

SIGT_v3_preproc = paste(outdir,'/', 'SIGT_v3_preproc.csv',sep = '')
write.csv(SIGT_v3_pre,SIGT_v3_preproc) 



############################################################################################################
# Now, rename SIGT columns to be consistent with IGT columns
############################################################################################################
names(SIGT_v1_pre) <- names(SIGT_v2_pre) <- names(SIGT_v3_pre) <- c("Subject", "ExperimentName", "absmoney1", "rewlos", "Srewlos", "card.RESP", "ydata", "cardname", "stim")




############################################################################################################
############################################################################################################
# code for making stan-ready lists 
############################################################################################################
############################################################################################################


for (df in 1:4) {
  if (df == 1) {
    input_data <- IGT_pre
    input_name <- "IGT_pre"
  } else if (df == 2) {
    input_data <- SIGT_v1_pre
    input_name <- "SIGT_v1_pre"
  } else if (df == 3) {
    input_data <- SIGT_v2_pre
    input_name <- "SIGT_v2_pre"
  } else if (df == 4) {
    input_data <- SIGT_v3_pre
    input_name <- "SIGT_v3_pre"
  }
  
  # Individual Subjects
  subjList <- unique(input_data$Subject) # list of subject IDs from the dataframe specified by input_data
  numSubjs <- length(subjList)        # number of subjects
  
  # Trials per subject in behavioral data (0 when missing)
  Tsubj <- as.vector(rep(0, numSubjs)) 
  
  # now use these for loops to fill the array 'Tsubj' with the number of trials, when there are trials, otherwise the value stays 0
  
  
  # Trials per subject in behavioral data
  # number of trials for each subject
  for ( i in 1:numSubjs )  {
    curSubj  <- subjList[i]
    Tsubj[i] <- sum(input_data$Subject==curSubj)
  }
  maxTrials <- max(Tsubj)
  
  
  
  # Behavioral data arrays
  RLmatrix <- SRLmatrix <- stim <- array(-1, c(numSubjs, maxTrials))
  Ydata <- array(-1, c(numSubjs, maxTrials))
  
  
  # Filling arrays with raw data
  for (i in 1:numSubjs) {
    currID   <- subjList[i]
    tmp_dat  <- subset(input_data, Subject==currID)
    stim[i,]  <- tmp_dat$stim
    Ydata[i,] <- tmp_dat$ydata
    RLmatrix[i,] <- tmp_dat$rewlos
    SRLmatrix[i,] <- tmp_dat$Srewlos
  }  
  
  
  dataList <- list(
    N       = numSubjs,
    T       = maxTrials,
    Tsubj   = Tsubj,
    card    = stim,
    sign = SRLmatrix,     
    outcome  = RLmatrix,
    choice   = Ydata,
    subjID  = subjList
  )
  
  # save each stan ready data list as .rds
  saveRDS(dataList, file = here("1_SIGT_PP", "Data", "1_Preprocessed", paste0(input_name, ".rds")))
          }



# to read the .rds file back into R, from the lab server: 
data_current <- readRDS(here("1_SIGT_PP", "Data", "1_Preprocessed", "IGT_pre.rds"))

