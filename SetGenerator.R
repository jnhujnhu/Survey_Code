Lt <- function(vec) {
    res = rep(0, 3)
    sum = 0
    for(i in 1:length(vec)) {
        sum = sum + log(vec[i])
    }
    for(i in 1:length(vec)) {
        res[i] = log(vec[i]) + 100# - 1/3 * sum
    }
    return(res)
}

Pi <- function(vec) {
    sd = 0.2
    thp = vector(length = 3)
    for(i in 1:length(vec)) {
        thp[i] = rnorm(1, vec[i], sd)
    }
    res = vector(length = 3)
    sum = 0
    for(i in 1:length(thp)) {
        res[i] = exp(thp[i])
        sum = sum + res[i]
    }
    for(i in 1:length(thp)) {
        res[i] = res[i]/sum
    }
    return(res)
}

ori1 = vector(length = 3)
ori1[1] = 0.35
ori1[2] = 0.2
ori1[3] = 0.45

ori2 = vector(length = 3)
ori2[1] = 0.22
ori2[2] = 0.46
ori2[3] = 0.32

theta1 = Lt(ori1)
theta2 = Lt(ori2)

gra1 = vector(length = 2000)
gra2 = vector(length = 2000)
gra3 = vector(length = 2000)
label = vector(length = 2000)

# label 1
for(i in 1:1000) {
    o = Pi(theta1)
    gra1[i] = o[1]
    gra2[i] = o[2]
    gra3[i] = 0
    label[i] = 1
}

# label -1
for(i in 1001:2000) {
    o = Pi(theta2)
    gra1[i] = o[1]
    gra2[i] = o[2]
    gra3[i] = 0
    label[i] = -1
}

df <- data.frame(gra1, gra2, gra3, label)

write.table(df, "train(2d).dat")

# Print Points
# plot3d(gra1, gra2, gra3, col = "red", size = 1, xlab="x", ylab = "y", zlab = "z"
# ,xlim = c(0, 1), ylim = c(0,1), zlim = c(0,1))
# readLines(con="stdin", 1)
