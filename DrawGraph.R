library(rgl)
library(fMultivar)

table <- read.table("/Users/Kevin/Desktop/Laboratory(CUHK)/SVM/L1SVM/train(2d).dat", header = F)
listx <- as.numeric(unlist(as.list(table["V2"])))
listy<- as.numeric(unlist(as.list(table["V3"])))
listz <- as.numeric(unlist(as.list(table["V4"])))

plot3d(listx[1:1000], listy[1:1000], listz[1:1000], col = "blue", size = 1, xlim = c(0, 1), ylim = c(0, 1), zlim = c(0, 1), xlab = 'x', ylab = 'y', zlab = 'z')
plot3d(listx[1001:2000], listy[1001:2000], listz[1001:2000], col = "green", size = 1, xlim = c(0, 1), ylim = c(0, 1), zlim = c(0, 1), xlab = 'x', ylab = 'y', zlab = 'z')

# planes3d(7.272069, 5.053268, -18.868760, 0, col = "red", alpha = 0.5)

readLines(con="stdin", 1)
