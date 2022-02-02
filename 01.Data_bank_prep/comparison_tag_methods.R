
# Script used to generate graphs based on the roots length's results

### Importing packages #########################################################
library(openxlsx)
library(ggplot2)
library(dplyr)

### Script #####################################################################

#reading the excel containing the results of the comparison
df <- read.xlsx('01.Data_bank_pretreat/Results/res_comparison_tag.xlsx')

#calculating the means of the times of execution
df %>% summarise(mean_time1 = mean(time1), mean_time2 = mean(time2)) %>% 
  print()

#printing the number of true positives
df %>% count(nb1==true_nb) 
df %>% count(nb2==true_nb) 
