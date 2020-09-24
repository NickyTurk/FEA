install.packages("sp")
library(dplyr)
library(sp)

function_list <- c(7,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,9,10,10,10,10,10,10,10,10,10,11,11,11,11,11,11,11,11,11,12,12,12,12,12,12,12,12,12,13,13,13,13,13,13,13,13,13,14,14,14,14,14,14,14,14,14,15,15,15,15,15,15,15,15,15,16,16,16,16,16,16,16,16,16,17,17,17,17,17,17,17,17,17,18,18,18,18,18,18,18,18,18,19,19,19,19,19,19,19,19,19,20,20,20,20,20,20,20,20,20)
function_files <- list.files(path = "C:/Users/f24n127/Documents/School/FEA/pso/results", pattern = "F\\d{1,2}_100_diff_grouping\\.csv")
large_dim_datasets <- vector("list", length(function_files))
for(i in 1:length(function_files)){
  large_dim_datasets[[i]] <- cbind(function_files[i], read.table(paste("C:/Users/f24n127/Documents/School/FEA/pso/results/",function_files[i], sep=''),header=TRUE, sep=",", stringsAsFactors=TRUE))
}

nr_groups_avg_per_dim <- vector('list', 9)
for(i in 1:length(large_dim_datasets)){
  print(i)
  print(data.frame(large_dim_datasets[i]))
}

large_dim_datasets[['DIMENSION']]

test_dataset2 <- rbind(F7_100_diff_grouping['DIMENSION'],F8_100_diff_grouping['DIMENSION'],F9_100_diff_grouping['DIMENSION'],F10_100_diff_grouping['DIMENSION'],F11_100_diff_grouping['DIMENSION'],F12_100_diff_grouping['DIMENSION'],F13_100_diff_grouping['DIMENSION'],F14_100_diff_grouping['DIMENSION'],F15_100_diff_grouping['DIMENSION'],F16_100_diff_grouping['DIMENSION'],F17_100_diff_grouping['DIMENSION'],F18_100_diff_grouping['DIMENSION'],F19_100_diff_grouping['DIMENSION'],F20_100_diff_grouping['DIMENSION'])
dim_groups <- cbind(function_list,test_dataset, test_dataset2)
plot(dim_groups$DIMENSION, dim_groups$NR_GROUPS, col=dim_groups$function_list)
legend('top', legend = unique(dim_groups$function_list), col = 1:length(unique(function_list)), cex = 0.8, pch = 1, horiz = TRUE)

files <- list.files(path = "C:/Users/f24n127/Documents/School/FEA/pso/results", pattern = "F*_diff_grouping_small_epsilon\\.csv")

datasets <- list()

for(i in 1:length(files)){
  curr_file <- read.table(paste("C:/Users/f24n127/Documents/School/FEA/pso/results/",files[i], sep=''),header=TRUE, sep=",", stringsAsFactors=FALSE)
  datasets <- rbind(datasets, curr_file )
}

datasets <- transform(datasets, DIMENSION = as.numeric(DIMENSION), EPSILON = as.numeric(EPSILON), NR_GROUPS = as.numeric(NR_GROUPS))

for(i in 1:length(unique(datasets$DIMENSION)[1:10])){
  plot(aggregate(datasets[datasets$DIMENSION==unique(datasets$DIMENSION)[i], 'NR_GROUPS'], list(datasets[datasets$DIMENSION == unique(datasets$DIMENSION)[i],'EPSILON']), mean), log='x', pch=20)
  title(paste("Dimension: ", unique(datasets$DIMENSION)[i], sep=""))
}

for(i in 1:length(unique(datasets$FUNCTION)[1:12])){
  plot(aggregate(datasets[datasets$FUNCTION==unique(datasets$FUNCTION)[i], 'NR_GROUPS'], list(datasets[datasets$FUNCTION == unique(datasets$FUNCTION)[i],'EPSILON']), mean), log='x', pch=20)
  title(paste("Function ", unique(datasets$FUNCTION)[i], sep=""))
}
