function_list <- c(7,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,9,10,10,10,10,10,10,10,10,10,11,11,11,11,11,11,11,11,11,12,12,12,12,12,12,12,12,12,13,13,13,13,13,13,13,13,13,14,14,14,14,14,14,14,14,14,15,15,15,15,15,15,15,15,15,16,16,16,16,16,16,16,16,16,17,17,17,17,17,17,17,17,17,18,18,18,18,18,18,18,18,18,19,19,19,19,19,19,19,19,19,20,20,20,20,20,20,20,20,20)

test_dataset2 <- rbind(F7_100_diff_grouping['DIMENSION'],F8_100_diff_grouping['DIMENSION'],F9_100_diff_grouping['DIMENSION'],F10_100_diff_grouping['DIMENSION'],F11_100_diff_grouping['DIMENSION'],F12_100_diff_grouping['DIMENSION'],F13_100_diff_grouping['DIMENSION'],F14_100_diff_grouping['DIMENSION'],F15_100_diff_grouping['DIMENSION'],F16_100_diff_grouping['DIMENSION'],F17_100_diff_grouping['DIMENSION'],F18_100_diff_grouping['DIMENSION'],F19_100_diff_grouping['DIMENSION'],F20_100_diff_grouping['DIMENSION'])
dim_groups <- cbind(function_list,test_dataset, test_dataset2)
plot(dim_groups$DIMENSION, dim_groups$NR_GROUPS, col=dim_groups$function_list)
legend('top', legend = unique(dim_groups$function_list), col = 1:length(unique(function_list)), cex = 0.8, pch = 1, horiz = TRUE)

files <- list.files(path = "~/FEA/results", pattern = "F*_diff_grouping_small_epsilon\\.csv")

datasets <- list()

for(i in 1:length(files)){
  curr_file <- read.table(paste("~/FEA/results/",files[i], sep=''),header=TRUE, sep=",", stringsAsFactors=FALSE)
  datasets <- rbind(datasets, curr_file )
}
  