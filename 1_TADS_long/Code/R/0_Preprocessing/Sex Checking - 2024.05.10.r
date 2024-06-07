library(haven)
library(tidyverse)

df_check <- read_sav('C:/Users/tuf22063/OneDrive - Temple University/TADS Data/Finalized Datasets Spring 2022/Demographic Info/FINALIZED_Demoinfo_7.1.22_KD.sav')

### Age checking
df_check <- df_check[c("id", grep("IGT", names(df_check), value = T)[c(1, 4, 7, 10, 13)])]

id_check <- data.frame(plyr::count(df_check, "id"))

subset(id_check, freq != 1)

### Checking sex

### Parse only the sex variables
df_check_sex <- df_check[c("id", grep("sex", names(df_check), ignore.case = T, value = T))]

### Restrict the data to conflicting variable values
df_check_sex_prob <- subset(df_check_sex, Sex_T1Child != T1_Sex_Child)

# Check cross-tabs between the 'Sex_T1Child' variable against the other sex variables
plyr::count(df_check_sex_prob, c("Sex_T1Child", "T1_Sex_Child"))

plyr::count(df_check_sex_prob, c("Sex_T1Child", "T2_Sex_Child"))

plyr::count(df_check_sex_prob, c("Sex_T1Child", "T3_Sex_Child"))

plyr::count(df_check_sex_prob, c("Sex_T1Child", "T4_Sex_Child"))

plyr::count(df_check_sex_prob, c("Sex_T1Child", "T5_Sex_Child"))

##### Check cross-tabs between the 'T1_Sex_Child' variable against the other sex variables

plyr::count(df_check_sex_prob, "T1_Sex_Child")

plyr::count(df_check_sex_prob, c("T1_Sex_Child", "T2_Sex_Child"))

plyr::count(df_check_sex_prob, c("T1_Sex_Child", "T3_Sex_Child"))

plyr::count(df_check_sex_prob, c("T1_Sex_Child", "T4_Sex_Child"))

plyr::count(df_check_sex_prob, c("T1_Sex_Child", "T5_Sex_Child"))

### ==> These are consistent across time

####### Create aggregate sex variable using information across time

df_check_sex <- df_check_sex %>% mutate(ch_sex_fix = 
                                          ifelse(!is.na(T1_Sex_Child), T1_Sex_Child,
                                                 ifelse(!is.na(T2_Sex_Child), T2_Sex_Child,
                                                        ifelse(!is.na(T3_Sex_Child), T3_Sex_Child,
                                                               ifelse(!is.na(T4_Sex_Child), T4_Sex_Child,
                                                                      ifelse(!is.na(T5_Sex_Child), T5_Sex_Child, ""
                                                                      ))))))
### Check frequency count for the fixed variable

plyr::count(df_check_sex, c("ch_sex_fix"))

plyr::count(df_check_sex, c("ch_sex_fix", "T2_Sex_Child"))

plyr::count(df_check_sex, c("ch_sex_fix", "T2_Sex_Child"))

plyr::count(df_check_sex, c("ch_sex_fix", "T3_Sex_Child"))

plyr::count(df_check_sex, c("ch_sex_fix", "T4_Sex_Child"))

plyr::count(df_check_sex, c("ch_sex_fix", "T5_Sex_Child"))

### ==> All consistent across time. However, some potentially missing values.

### Sex variable also from a screening dataset; restricted to only cases that we should have data on.

sex_sum <- read.csv('C:/Users/tuf22063/OneDrive - Temple University/TADS Data/Finalized Datasets Spring 2022/Demographic Info/TADS Participant Sex Check - 10.10.22.csv')

### Merge this information
sex_check <- merge(df_check_sex, sex_sum, by = "id")

### cross tabs for the comparison
plyr::count(sex_check, c("ch_sex_fix", "screen_sex"))

## ==> all match, but the screening dataset has more complete data.
