library(sommer)
data("DT_technow")

DT_technow[1:5,1:5]
Md_technow[1:5,1:5]
Mf_technow[1:5,1:5]

# Some formating
DT_technow$dent  <- paste0("D", DT_technow$dent)
DT_technow$flint <- paste0("F", DT_technow$flint)

rownames(Md_technow) <- paste0("D", rownames(Md_technow))
rownames(Mf_technow) <- paste0("F", rownames(Mf_technow))

# Match parent indices required for hybrids
idx.f <- match(DT_technow$flint, rownames(Mf_technow))
idx.d <- match(DT_technow$dent, rownames(Md_technow))

# Generate hybrids
Mdf <- Md_technow[idx.d,] + Mf_technow[idx.f,] # Hybrids with format 0, 1, 2
Mdf[1:10,1:10]
rownames(Mdf) <- DT_technow$hybrid <- paste(DT_technow$dent, DT_technow$flint, sep = ":")

# Use match markers to filtered data (e.g. apply filter again)
hy <- read.table("dentflint_filtered.txt")
m.idx <- colnames(Mdf) %in% colnames(hy)
Mdf <- Mdf[,m.idx]

# Write data
write.table(DT_technow, "Y_data.csv")
write.table(Mdf, "hybrids.txt")
