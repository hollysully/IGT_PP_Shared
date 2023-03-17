library(dplyr)
library(foreach)
library(gdata)
library(haven)

setwd("~/Dropbox/Box/LAP/Blair/")

# Get all participant files for PA task (all start with "PA")
file_names <- list.files("Data/0_Raw/Passive_Avoidance/", pattern = "^PA")

# Get record and subject IDs for complete dataset
survey_dat <- read.xls("Data/0_Raw/Survey/S_DATABASE.xlsx") %>%
  select(record_id, ssid_number) %>%
  full_join(read_sav("Data/0_Raw/Survey/SPSS_Big_DB_ToSend.sav") %>%
              select(RCID, Subj),
            by = c("record_id" = "RCID")) %>%
  mutate(subj_id = ifelse(is.na(ssid_number), Subj, ssid_number)) %>%
  select(record_id, subj_id)

# Reverse scored items on all relevant scales
sdq_reversed <- c(7, 11, 14, 21, 25)
icu_reversed <- c(1, 3, 5, 8, 13, 14, 15, 16, 17, 19, 23, 24)

# Read in item-level data and rename for convenience
item_dat <- read.xls("Data/0_Raw/Survey/PdetailsScales.xlsx") %>%
  # Renaming
  rename(subj_id = subject_id_suq) %>%
  rename_with(function(x) paste0("AUDIT_", 1:10),
              how_often_did_you_have_a_d:has_a_relative_friend_doct) %>%
  rename_with(function(x) paste0("CUDIT_", 1:8),
              how_often_did_you_use_cann:have_you_ever_thought_abou) %>%
  rename(smoking = have_you_ever_smoked_cigar) %>%
  rename_with(function(x) paste0("SDQ_", 1:25),
              i_try_to_be_nice_to_other:i_finish_the_work_i_m_doin) %>%
  rename_with(function(x) paste0("ICU_", 1:24),
              i_express_my_feelings_openly:i_do_things_to_make_others) %>%
  rename_with(function(x) paste0("Conners_", 1:10),
              fidgeting:interrups_others_for_examp) %>%
  rename_with(function(x) paste0("RPQ_", 1:23),
              yelled:yell_to_do) %>%
  rename_with(function(x) paste0("SCARED_", 1:41),
              when2_i_feel_frightened_it:i_am_shy2) %>%
  rename_with(function(x) paste0("ARI_", 1:7),
              i_am_easily_annoyed_by_oth:overall_my_irritability_ca) %>%
  rename_with(function(x) paste0("MFQ_", 1:33),
              i_felt_miserable_or_unhapp:i_slept_a_lot_more_than_us) %>%
  # Pre-processing
  mutate_at(vars(AUDIT_2:AUDIT_10), ~ifelse(AUDIT_1==0, 0, .)) %>%
  mutate_at(vars(CUDIT_2:CUDIT_8), ~ifelse(CUDIT_1==0, 0, .)) %>%
  mutate_at(vars(paste0("SDQ_", sdq_reversed)), ~(2-.)) %>%
  mutate_at(vars(paste0("ICU_", icu_reversed)), ~(3-.)) %>%
  mutate(CUDIT_2 = ifelse(CUDIT_2==5, 4, CUDIT_2)) %>% # Scored as a 5 when should only go to 4
  full_join(survey_dat, by = "record_id") %>%
  rename(subj_id = subj_id.y) %>%
  select(record_id, subj_id, contains("AUDIT_"),
         contains("CUDIT_"), smoking, contains("SDQ_"),
         contains("ICU_"), contains("Conners_"),
         contains("RPQ_"), contains("SCARED_"),
         contains("ARI_"), contains("MFQ_")) %>%
  mutate(n_NAs = apply(., 1, function(x) sum(is.na(x)))) %>%
  filter(n_NAs < 142 & !(subj_id %in% c(1185, 2438))) # subjects have no behavioral data, remove for now

# Read PA data in, into long format
inc_ids <- NULL
pa_dat <- foreach(i=file_names, .combine = "rbind") %do% {
  cur_id <- substr(strsplit(i, "_")[[1]][2], 1, 4)
  if (!(cur_id %in% inc_ids)) {
    inc_ids <- c(inc_ids, cur_id)
    tmp <- read.xls(paste0("Data/0_Raw/Passive_Avoidance/", i), sheet = "DATA2") %>% 
      select(Subject, Trial, ObjectNumber, Amount, Resp)
  } else {
    tmp <- NULL
  }
  return(tmp)
} 

# Make stimulus ID consistent and combine with item-level data
all_dat <- pa_dat %>%
  mutate(stimulus = case_when(ObjectNumber=="1" ~ 1,
                              ObjectNumber=="2" ~ 2,
                              ObjectNumber=="3" ~ 3,
                              ObjectNumber=="4" ~ 4,
                              ObjectNumber=="e1" ~ 1,
                              ObjectNumber=="e2" ~ 2,
                              ObjectNumber=="e3" ~ 3,
                              ObjectNumber=="e4" ~ 4,
                              ObjectNumber=="e9" ~ 1,
                              ObjectNumber=="10" ~ 2,
                              ObjectNumber=="11" ~ 3,
                              ObjectNumber=="12" ~ 4)) %>%
  full_join(item_dat, 
             by = c("Subject" = "subj_id"))

# Individual Subjects
subjList <- unique(all_dat$Subject)  # list of subjects x blocks
numSubjs <- length(subjList)        # number of subjects

# Trials per subject in behavioral data
Tsubj <- as.vector(rep(0, numSubjs)) # number of trials for each subject
for ( i in 1:numSubjs )  {
  curSubj  <- subjList[i]
  Tsubj[i] <- sum(all_dat$Subject==curSubj)
}
maxTrials <- max(Tsubj)

# Behavioral data arrays
RLmatrix <- SRLmatrix <- stim <- array(-1, c(numSubjs, maxTrials))
Ydata <- array(-1, c(numSubjs, maxTrials))

# Survey data, #items per questionnaire 
n_items <- c(5,  # sdq_hyperactivity 
             5,  # sdq_conduct_problems
             5,  # sdq_prosocial (will reverse to indicate anti-social)
             24, # ICU
             10, # conners_adhd_total_score (30 if ADHD index and items scored 0-3)
             23, # rpq_total
             7,  # ARI
             10, # AUDIT 
             8,  # CUDIT 
             33, # MFQ
             5,  # sdq_emotional_problems
             41) # SCARED total

# Number response options, same order as above
n_resp <- c(3, 3, 3, 4, 4, 3, 3, 5, 5, 3, 3, 3)

# Survey data arrays
y1  <- array(-1, c(numSubjs, n_items[1], 2))
y2  <- array(-1, c(numSubjs, n_items[2], 2))
y3  <- array(-1, c(numSubjs, n_items[3], 2))
y4  <- array(-1, c(numSubjs, n_items[4], 2))
y5  <- array(-1, c(numSubjs, n_items[5], 2))
y6  <- array(-1, c(numSubjs, n_items[6], 2))
y7  <- array(-1, c(numSubjs, n_items[7], 2))
y8  <- array(-1, c(numSubjs, n_items[8], 2))
y9  <- array(-1, c(numSubjs, n_items[9], 2))
y10 <- array(-1, c(numSubjs, n_items[10], 2))
y11 <- array(-1, c(numSubjs, n_items[11], 2))
y12 <- array(-1, c(numSubjs, n_items[12], 2))

# Filling arrays with raw data
for (i in 1:numSubjs) {
  currID   <- subjList[i]
  tmp_dat  <- subset(all_dat, Subject==currID)
  
  # Whether participant completed item or not
  y1[i,,1]  <- as.numeric(!is.na(tmp_dat[1,paste0("SDQ_", c(2, 10, 15, 21, 25))]))
  y2[i,,1]  <- as.numeric(!is.na(tmp_dat[1,paste0("SDQ_", c(5, 7, 12, 18, 22))]))
  y3[i,,1]  <- as.numeric(!is.na(tmp_dat[1,paste0("SDQ_", c(1, 4, 9, 17, 20))]))
  y4[i,,1]  <- as.numeric(!is.na(tmp_dat[1,paste0("ICU_", 1:24)]))
  y5[i,,1]  <- as.numeric(!is.na(tmp_dat[1,paste0("Conners_", 1:10)]))
  y6[i,,1]  <- as.numeric(!is.na(tmp_dat[1,paste0("RPQ_", 1:23)]))
  y7[i,,1]  <- as.numeric(!is.na(tmp_dat[1,paste0("ARI_", 1:7)]))
  y8[i,,1]  <- as.numeric(!is.na(tmp_dat[1,paste0("AUDIT_", 1:10)]))
  y9[i,,1]  <- as.numeric(!is.na(tmp_dat[1,paste0("CUDIT_", 1:8)]))
  y10[i,,1] <- as.numeric(!is.na(tmp_dat[1,paste0("MFQ_", 1:33)]))
  y11[i,,1] <- as.numeric(!is.na(tmp_dat[1,paste0("SDQ_", c(3, 8, 13, 16, 24))]))
  y12[i,,1] <- as.numeric(!is.na(tmp_dat[1,paste0("SCARED_", 1:41)]))
  
  # Add 1 to all responses to start from 1 (as opposed to 0)
  y1[i,,2]  <- as.numeric(tmp_dat[1,paste0("SDQ_", c(2, 10, 15, 21, 25))] + 1)
  y2[i,,2]  <- as.numeric(tmp_dat[1,paste0("SDQ_", c(5, 7, 12, 18, 22))] + 1)
  y3[i,,2]  <- (4 - as.numeric(tmp_dat[1,paste0("SDQ_", c(1, 4, 9, 17, 20))] + 1)) # reverse to indicate anti-social
  y4[i,,2]  <- as.numeric(tmp_dat[1,paste0("ICU_", 1:24)] + 1)
  y5[i,,2]  <- as.numeric(tmp_dat[1,paste0("Conners_", 1:10)] + 1)
  y6[i,,2]  <- as.numeric(tmp_dat[1,paste0("RPQ_", 1:23)] + 1)
  y7[i,,2]  <- as.numeric(tmp_dat[1,paste0("ARI_", 1:7)] + 1)
  y8[i,,2]  <- as.numeric(tmp_dat[1,paste0("AUDIT_", 1:10)] + 1)
  y9[i,,2]  <- as.numeric(tmp_dat[1,paste0("CUDIT_", 1:8)] + 1)
  y10[i,,2] <- as.numeric(tmp_dat[1,paste0("MFQ_", 1:33)] + 1)
  y11[i,,2] <- as.numeric(tmp_dat[1,paste0("SDQ_", c(3, 8, 13, 16, 24))] + 1)
  y12[i,,2] <- as.numeric(tmp_dat[1,paste0("SCARED_", 1:41)] + 1)
  
  # Replace NA responses with 0, skipped in loglik iterating 
  y1[is.na(y1)]   <- 0
  y2[is.na(y2)]   <- 0
  y3[is.na(y3)]   <- 0
  y4[is.na(y4)]   <- 0
  y5[is.na(y5)]   <- 0
  y6[is.na(y6)]   <- 0
  y7[is.na(y7)]   <- 0
  y8[is.na(y8)]   <- 0
  y9[is.na(y9)]   <- 0
  y10[is.na(y10)] <- 0
  y11[is.na(y11)] <- 0
  y12[is.na(y12)] <- 0
  
  stim[i,]  <- tmp_dat$stimulus
  Ydata[i,] <- tmp_dat$Resp+1
  RLmatrix[i,] <- ifelse(tmp_dat$Resp==1, tmp_dat$Amount, 0)
    
  for (t in 1:maxTrials) {
    # For binarizing 
    if ( RLmatrix[i,t] > 0 ) {
      SRLmatrix[i,t] <- 1
    } else if ( RLmatrix[i,t] == 0 ) {
      SRLmatrix[i,t] <- 0
    } else {
      SRLmatrix[i,t] <- -1
    }
  }
}

dataList <- list(
  N       = numSubjs,
  T       = maxTrials,
  Tsubj   = Tsubj,
  N_items = n_items,
  N_resp  = n_resp,
  y1      = y1,
  y2      = y2,
  y3      = y3,
  y4      = y4,
  y5      = y5,
  y6      = y6,
  y7      = y7,
  y8      = y8,
  y9      = y9,
  y10     = y10,
  y11     = y11,
  y12     = y12,
  stim    = stim,
  Srewlos = SRLmatrix,     
  rewlos  = RLmatrix,
  ydata   = Ydata,
  subjID  = subjList
)

saveRDS(dataList, file = "Data/1_Preprocessed/stan_ready_PA_item_fullInfo_Q12_v1.rds")
