library(dplyr)
library(tidyverse) 
library(haven)
library(lubridate)
library(readxl)


here::i_am("/Users/tuo09169/Dropbox/1_Comp_Modelling/1_SIGT_PP/")

setwd("/Users/tuo09169/Dropbox/1_Comp_Modelling/1_SIGT_PP/")
library(here)
here()

here::here()

#import IGT_PP & SIGT_PP raw data
# NOTE: before I saved these as .xlsx files, I converted 'SessionDate' to format:  2015-10-23, to work with lubridate
# `here(...)` finds the root project directory, and then creates the file path given arguements
IGT_AB <- read_excel(here("./Data/0_Raw/IGT/AB/IGT_AB_Merged.xlsx"))
IGT_AB <- read_excel(here("1_SIGT_PP", "Data", "0_Raw", "IGT", "AB", "IGT_AB_Merged.xlsx"))
IGT_BD <- read_excel(here("Data", "0_Raw", "IGT", "BD", "IGT_BD_Merged.xlsx"))

data <- read_csv(here("./datafolder/datafile.csv"))

names(IGT_AB)
length(unique(AB$Subject)) #44
length(unique(BD$Subject)) #44


